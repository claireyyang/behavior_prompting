"""
Per-demo strategy sampling for the simplified draw environment: the "how" of a demo, sampled
independently of its geometry.

A strategy here is nothing but a **monotone time-warp over arc length**, plus a permutation and
a set of reversal flags. It cannot express anything else -- there is no way to command a point
that is off the task's canonical polyline. That restriction is the whole point: it is what makes
the invariance a theorem rather than a measurement.

    order          permutation of stroke indices, in execution order
    reversed_flags reversed_flags[i] applies to the stroke drawn i-th (post-permutation)
    profiles       (family, strength) per drawn stroke -- the shape of the velocity ramp
    speed_mults    per-drawn-stroke multiplier on the demo's base speed
    base_speed     the demo's base speed in px/s

Why the ink is identical across all of these
--------------------------------------------
The env inks the closed arc-length interval swept by each step (`StrokeGeometry.edges_in`).
Consecutive steps share an endpoint exactly, so the per-step intervals chain contiguously and
their union is [min_t s_t, max_t s_t]. `schedule_arclengths` pins s_0 = 0 and s_T = L, so that
union is exactly [0, L] -- the whole stroke -- for any monotone profile and any step count.
Reversal traverses L -> 0, which has the same union. Permuting strokes permutes a set union.

Positional noise stays ink-neutral for the same reason, provided (a) it keeps the cursor within
the env's `snap_tol` of the polyline, and (b) the first and last action of each stroke are the
exact endpoints. `build_stroke_actions` enforces both.

Label names deliberately match `scripts/draw/strategy_variation.py`'s `strategy_*` keys so
existing analysis scripts keep working against these datasets.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from behavior_prompting.train_network.env.draw_simple.geometry import (
    StrokeGeometry,
    board_to_canvas,
)

PROFILE_FAMILIES = ('linear', 'ease_in', 'ease_out', 'ease_in_out', 'ease_mid')


# ---------------------------------------------------------------------------------------
# velocity profiles: monotone warps of [0, 1] -> [0, 1]
# ---------------------------------------------------------------------------------------

def warp(u: np.ndarray, family: str = 'linear', strength: float = 0.7) -> np.ndarray:
    """
    Warp normalised progress `u` into normalised arc length.

    Every family is strictly increasing with warp(0) = 0 and warp(1) = 1, so the schedule stays
    monotone and covers the full stroke. `strength` in [0, 1) interpolates from linear to the
    family's extreme; strength = 0 is the identity for every family.
    """
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    s = float(np.clip(strength, 0.0, 0.99))

    if family == 'linear':
        return u
    if family == 'ease_in':          # slow start, fast finish
        return u ** (1.0 + 2.0 * s)
    if family == 'ease_out':         # fast start, slow finish
        return 1.0 - (1.0 - u) ** (1.0 + 2.0 * s)
    if family == 'ease_in_out':      # slow at both ends, fast through the middle
        return (1.0 - s) * u + s * (3.0 * u ** 2 - 2.0 * u ** 3)
    if family == 'ease_mid':         # fast at both ends, slow through the middle
        inv = 0.5 - np.sin(np.arcsin(np.clip(1.0 - 2.0 * u, -1.0, 1.0)) / 3.0)
        return (1.0 - s) * u + s * inv
    raise ValueError(f'unknown profile family {family!r}; expected one of {PROFILE_FAMILIES}')


def profile_is_monotone(family: str, strength: float, n: int = 4096) -> bool:
    w = warp(np.linspace(0.0, 1.0, n), family, strength)
    return bool(np.all(np.diff(w) >= -1e-12))


def schedule_arclengths(length: float, speed: float, control_hz: int,
                        family: str = 'linear', strength: float = 0.7,
                        min_steps: int = 2,
                        max_step_len: Optional[float] = None,
                        max_samples: int = 200_000) -> np.ndarray:
    """
    Arc lengths at which to sample a stroke of length `length`, traversed at mean `speed` px/s.

    `speed` sets the STEP COUNT (how finely the stroke is sampled) and the profile sets how
    those samples are distributed along it. Endpoints are pinned exactly -- that pin is what
    guarantees full coverage, so do not remove it.

    ⚠️ `max_step_len` is not optional in practice: pass the env's `step_cap`.

    A profile does not change the mean step but it very much changes the MAXIMUM step -- an
    `ease_out` at strength 0.95 puts ~2.5x the mean into its first step, and `ease_mid` has an
    unbounded derivative at both ends. If a commanded step exceeds what the env can move in one
    control period, the env clips it, and the clipped point sits on the straight chord rather
    than on the polyline -- so it lands beyond `snap_tol`, inks nothing, and the stroke silently
    comes out partially drawn. Coverage stops being 1.0 and invariance is lost.

    So any interval exceeding `max_step_len` is SUBDIVIDED IN u until it fits. Subdivision adds
    samples that lie on the same warp curve, so the profile's shape is preserved exactly; the
    profile just saturates against the env's speed limit, which is honest. Ink is unaffected
    either way -- the union of the intervals is [0, L] regardless of how finely they are cut.
    """
    step_len = max(speed / float(control_hz), 1e-6)
    n_seg = max(int(min_steps), int(math.ceil(length / step_len)))
    u = np.linspace(0.0, 1.0, n_seg + 1)

    if max_step_len is None or length <= 0:
        s = float(length) * warp(u, family, strength)
        s[0], s[-1] = 0.0, float(length)
        return s

    cap = float(max_step_len)
    for _ in range(64):
        s = float(length) * warp(u, family, strength)
        d = np.diff(s)
        bad = d > cap
        if not bad.any():
            break
        if len(u) > max_samples:
            raise ValueError(
                f'could not make a ({family}, {strength}) schedule over {length:.1f}px fit a '
                f'{cap:.1f}px step cap within {max_samples} samples; lower --profile-strength '
                f'or raise --max-speed')
        pieces = [u[:1]]
        for i, is_bad in enumerate(bad):
            if is_bad:
                k = int(math.ceil(d[i] / cap))
                pieces.append(np.linspace(u[i], u[i + 1], k + 1)[1:])
            else:
                pieces.append(u[i + 1:i + 2])
        u = np.concatenate(pieces)
    else:
        raise ValueError(f'schedule refinement did not converge for ({family}, {strength})')

    s = float(length) * warp(u, family, strength)
    s[0], s[-1] = 0.0, float(length)
    return s


# ---------------------------------------------------------------------------------------
# the strategy
# ---------------------------------------------------------------------------------------

def count_inversions(order: Sequence[int]) -> int:
    """0 for the identity permutation, n(n-1)/2 for a full reversal."""
    o = list(order)
    return sum(1 for i in range(len(o)) for j in range(i + 1, len(o)) if o[i] > o[j])


def order_rank(order: Sequence[int]) -> int:
    """Lexicographic rank of a permutation, so an order can be logged as one integer."""
    o = list(order)
    n = len(o)
    rank = 0
    for i in range(n):
        smaller = sum(1 for j in range(i + 1, n) if o[j] < o[i])
        rank += smaller * math.factorial(n - i - 1)
    return rank


@dataclass
class Strategy:
    order: List[int]
    reversed_flags: List[bool]
    profiles: List[Tuple[str, float]]
    speed_mults: List[float]
    base_speed: float

    def __post_init__(self):
        n = len(self.order)
        assert sorted(self.order) == list(range(n)), f'order is not a permutation: {self.order}'
        assert len(self.reversed_flags) == n
        assert len(self.profiles) == n
        assert len(self.speed_mults) == n
        for family, strength in self.profiles:
            assert profile_is_monotone(family, strength), \
                f'profile ({family}, {strength}) is not monotone; coverage would not be exact'

    def summary_labels(self) -> Dict[str, float]:
        n = max(len(self.order), 1)
        return {
            'strategy_n_strokes': float(len(self.order)),
            'strategy_order_rank': float(order_rank(self.order)),
            'strategy_frac_reversed': float(np.mean(self.reversed_flags)) if self.reversed_flags else 0.0,
            'strategy_profile_id': float(PROFILE_FAMILIES.index(self.profiles[0][0])) if self.profiles else 0.0,
            'strategy_profile_strength': float(self.profiles[0][1]) if self.profiles else 0.0,
            'strategy_base_speed': float(self.base_speed),
            'strategy_order_first': float(self.order[0]) if self.order else -1.0,
            'strategy_order_last': float(self.order[-1]) if self.order else -1.0,
            'strategy_order_inversions': float(count_inversions(self.order)) / max(n * (n - 1) / 2, 1),
            'strategy_mean_speed_mult': float(np.mean(self.speed_mults)) if self.speed_mults else 1.0,
        }

    def __repr__(self) -> str:
        fams = ','.join(f for f, _ in self.profiles)
        return (f'Strategy(order={self.order}, reversed={self.reversed_flags}, '
                f'profiles=[{fams}], base_speed={self.base_speed:.1f})')


def sample_strategy(rng: np.random.Generator,
                    n_strokes: int,
                    base_speed: float,
                    vary_order: bool = True,
                    vary_direction: bool = True,
                    reverse_probability: float = 0.5,
                    profile_families: Sequence[str] = PROFILE_FAMILIES,
                    profile_strength: float = 0.7,
                    profile_per_stroke: bool = False,
                    stroke_speed_ratio: float = 1.0) -> Strategy:
    """
    Sample one demo's strategy.

    `profile_per_stroke=False` draws one family for the whole demo, which makes the profile axis
    a demo-level cadence signature rather than per-stroke jitter. `stroke_speed_ratio=1.0`
    disables per-stroke speed variation; r > 1 samples each multiplier log-uniformly in
    [1/r, r], so the demo's mean speed stays unbiased.
    """
    order = ([int(i) for i in rng.permutation(n_strokes)]
             if (vary_order and n_strokes > 1) else list(range(n_strokes)))

    reversed_flags = [bool(rng.random() < reverse_probability) if vary_direction else False
                      for _ in range(n_strokes)]

    families = list(profile_families) or ['linear']
    if profile_per_stroke:
        profiles = [(str(rng.choice(families)), float(profile_strength)) for _ in range(n_strokes)]
    else:
        one = (str(rng.choice(families)), float(profile_strength))
        profiles = [one] * n_strokes

    if stroke_speed_ratio > 1.0:
        lo, hi = math.log(1.0 / stroke_speed_ratio), math.log(stroke_speed_ratio)
        speed_mults = [float(math.exp(rng.uniform(lo, hi))) for _ in range(n_strokes)]
    else:
        speed_mults = [1.0] * n_strokes

    return Strategy(order, reversed_flags, profiles, speed_mults, float(base_speed))


# ---------------------------------------------------------------------------------------
# strategy -> actions
# ---------------------------------------------------------------------------------------

def build_stroke_actions(geom: StrokeGeometry, stroke_idx: int, reverse: bool,
                         speed: float, control_hz: int,
                         profile: Tuple[str, float],
                         angle: float,
                         rng: Optional[np.random.Generator] = None,
                         noise_std: float = 0.0,
                         noise_bounds: float = 1.5,
                         pen_value: float = 1.0,
                         max_step_len: Optional[float] = None
                         ) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build the (T, 3) pen-down action array for one stroke, in the canvas frame.

    Returns `(actions, info)`. `info` reports what actually happened rather than what was
    commanded -- `speed_capped` is set when the profile had to be refined to fit `max_step_len`,
    i.e. when this stroke's requested cadence saturated the env's speed limit.

    Pass `max_step_len=env.step_cap`; see `schedule_arclengths` for why omitting it silently
    costs coverage.

    The first and last actions are the exact stroke endpoints, unperturbed. That is not
    cosmetic: it is what pins the swept arc-length interval to exactly [0, L] and therefore what
    makes the inked edge set independent of the schedule and of the noise.
    """
    length = geom.stroke_length(stroke_idx)
    family, strength = profile
    requested_steps = max(2, int(math.ceil(length / max(speed / control_hz, 1e-6))))
    s = schedule_arclengths(length, speed, control_hz, family, strength,
                            max_step_len=max_step_len)
    if reverse:
        s = length - s          # traverse L -> 0, keeping the profile in traversal order

    pts = geom.points_at(stroke_idx, s)

    if noise_std > 0 and len(pts) > 2:
        assert rng is not None, 'pass an rng to use noise'
        noise = np.clip(rng.normal(0.0, noise_std, size=(len(pts) - 2, 2)),
                        -noise_bounds, noise_bounds)
        pts[1:-1] = pts[1:-1] + noise

    canvas = board_to_canvas(pts, angle)
    pen = np.full((len(canvas), 1), float(pen_value))
    actions = np.concatenate([canvas, pen], axis=1).astype(np.float32)

    steps = np.linalg.norm(np.diff(canvas, axis=0), axis=1) if len(canvas) > 1 else np.zeros(1)
    n_seg = max(len(actions) - 1, 1)
    info = {
        'n_steps': float(len(actions)),
        'max_step_px': float(steps.max()),
        'mean_step_px': float(steps.mean()),
        'realized_speed': float(length / n_seg * control_hz),
        'speed_capped': float(n_seg > requested_steps),
    }
    return actions, info


def realized_strokes(strategy: Strategy) -> List[Tuple[int, bool, Tuple[str, float], float]]:
    """
    Flatten a strategy into the per-execution-slot tuples the generator loops over:
    (stroke index, reverse flag, profile, speed). Reversal flags and profiles are indexed by
    execution position, not by original stroke index, so "the third stroke I draw is drawn
    backwards" is expressible independently of the permutation.
    """
    out = []
    for exec_idx, stroke_idx in enumerate(strategy.order):
        speed = strategy.base_speed * strategy.speed_mults[exec_idx]
        out.append((int(stroke_idx), bool(strategy.reversed_flags[exec_idx]),
                    strategy.profiles[exec_idx], float(speed)))
    return out
