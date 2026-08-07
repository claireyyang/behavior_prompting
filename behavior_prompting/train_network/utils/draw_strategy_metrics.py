"""
Recover the strategy a `SimpleDrawEnv` rollout actually realised, and score its distribution.

Why this exists
---------------
`scripts/draw_simple/generate_simple_drawings.py` builds datasets in which every demo of a task
renders a bit-identical goal image while the *strategy* -- stroke order and per-stroke pacing --
varies. A goal-image-conditioned policy therefore gets no strategy signal from its input, and the
question "which strategy did it pick?" is a real one: does it cover the demo distribution, or has
it collapsed onto one mode?

Answering it needs the realised strategy of a rollout. Here that is GROUND TRUTH, not an estimate:
`SimpleDrawEnv` resolves which stroke the pen is on (`_active_stroke`, with the continuity rule
that fixes attribution where strokes cross) and how far along it (`arc_s`), and records both every
step in `stroke_trace`. So none of `scripts/draw/draw_axes.py`'s dwell/pen segmentation is needed
-- that estimator exists to cope with the PD-controlled `env/draw/draw_env.py`, where stroke
boundaries have to be inferred from noisy positions.

Recovering pacing
-----------------
`strategy.schedule_arclengths` samples a stroke at arc lengths `L * warp(u, family, strength)`
where `u` is normalised progress through the stroke's steps. So the inverse is direct: take the
recorded arc lengths of one stroke, normalise them to [0, 1], and pick the family whose `warp`
best fits. No search over `strength` -- the generator uses a single fixed `--profile-strength`.

⚠️ This assumes step index is uniform in `u`, which holds only while no interval exceeded the
env's step cap. `schedule_arclengths` subdivides oversized intervals, which preserves the warp
CURVE but destroys uniform-in-`u` sampling and so biases this fit. At the generator's defaults
(200 px/s at 10 Hz = 20 px/step against a 120 px cap) no subdivision happens. Raise the speed or
the profile strength far enough and pacing recovery degrades -- order recovery is unaffected.

⚠️ Profiles are sampled PER STROKE (`--profile-per-stroke` is the generator default), so a demo
has no single profile. The scalar `strategy_profile_id` label reports only the first drawn
stroke's family and must not be used as a demo-level target; the per-step `profile_id` label is
the ground truth to compare against.
"""

import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from behavior_prompting.train_network.scripts.draw_simple.strategy import (
    count_inversions,
    order_rank,
    warp,
)

# The generator's default `--profile-families`. Deliberately not `strategy.PROFILE_FAMILIES`,
# which has five entries: fitting against families the data never sampled lets the argmin pick one
# of them and invents variation that is not there.
DEFAULT_PROFILE_FAMILIES: Tuple[str, ...] = ('linear', 'ease_in', 'ease_out')
DEFAULT_PROFILE_STRENGTH = 0.7

# A rollout whose drawing is unfinished has a truncated, meaningless stroke order. Scoring it as a
# strategy mismatch would report "poor strategy match" for what is really a task failure, so such
# rollouts are excluded and counted separately -- see `strategy_metrics`.
DEFAULT_COVERAGE_THRESHOLD = 0.9

# Below this many steps a stroke's arc-length curve has too few samples to tell the families apart.
MIN_STROKE_STEPS_FOR_PROFILE = 4


@dataclass
class RealizedStrategy:
    """What one rollout actually did. `order` is the unit of comparison; the rest is detail."""
    order: List[int] = field(default_factory=list)          # strokes, in first-acquisition order
    profiles: List[int] = field(default_factory=list)       # index into `families`, per `order`
    reversed_flags: List[bool] = field(default_factory=list)
    coverage: float = 0.0
    off_path_rate: float = 0.0
    n_steps: int = 0

    def is_complete(self, n_strokes: int, coverage_threshold: float) -> bool:
        return (self.coverage >= coverage_threshold
                and len(self.order) == n_strokes
                and sorted(self.order) == list(range(n_strokes)))


def _stroke_runs(strokes: np.ndarray) -> List[Tuple[int, int, int]]:
    """Contiguous `(stroke_idx, start, end)` runs of pen-down-on-a-stroke steps."""
    runs = []
    start = None
    for i, s in enumerate(strokes):
        if s < 0:
            if start is not None:
                runs.append((int(strokes[start]), start, i))
                start = None
        elif start is None:
            start = i
        elif strokes[i] != strokes[start]:
            runs.append((int(strokes[start]), start, i))
            start = i
    if start is not None:
        runs.append((int(strokes[start]), start, len(strokes)))
    return runs


def fit_profile(arc_s: np.ndarray,
                families: Sequence[str] = DEFAULT_PROFILE_FAMILIES,
                strength: float = DEFAULT_PROFILE_STRENGTH) -> Optional[int]:
    """
    Index of the profile family whose `warp` best explains one stroke's arc-length schedule.

    Returns None when the run is too short to discriminate, rather than guessing -- a guess would
    show up downstream as spurious pacing diversity.
    """
    n = len(arc_s)
    if n < MIN_STROKE_STEPS_FOR_PROFILE:
        return None
    span = arc_s[-1] - arc_s[0]
    if abs(span) < 1e-9:
        return None
    # Dividing by a signed span normalises reversed strokes to a rising 0 -> 1 curve too, so the
    # same family set fits both directions.
    a = (arc_s - arc_s[0]) / span
    u = np.linspace(0.0, 1.0, n)
    errs = [float(np.mean((a - warp(u, fam, strength)) ** 2)) for fam in families]
    return int(np.argmin(errs))


def recover_strategy(trace: Sequence[Tuple[int, float, bool, float]],
                     families: Sequence[str] = DEFAULT_PROFILE_FAMILIES,
                     profile_strength: float = DEFAULT_PROFILE_STRENGTH) -> RealizedStrategy:
    """
    Turn one env's `stroke_trace` into the strategy it realised.

    `trace` entries are `(active_stroke, arc_s, pen_down, coverage)`; `active_stroke` is -1 while
    the pen is up or off the canonical path.
    """
    if len(trace) == 0:
        return RealizedStrategy()

    strokes = np.array([t[0] for t in trace], dtype=np.int64)
    arc = np.array([t[1] for t in trace], dtype=np.float64)
    pen = np.array([bool(t[2]) for t in trace])

    n_pen_down = int(pen.sum())
    off_path_rate = float((pen & (strokes < 0)).sum() / n_pen_down) if n_pen_down else 0.0

    result = RealizedStrategy(coverage=float(trace[-1][3]),
                              off_path_rate=off_path_rate,
                              n_steps=len(trace))

    # Order is first ACQUISITION order: a stroke revisited later does not move in the order, and
    # its pacing is read from the first run, which is the one the generator scheduled.
    for k, lo, hi in _stroke_runs(strokes):
        if k in result.order:
            continue
        result.order.append(k)
        seg = arc[lo:hi]
        result.reversed_flags.append(bool(len(seg) > 1 and seg[-1] < seg[0]))
        fam = fit_profile(seg, families, profile_strength)
        result.profiles.append(-1 if fam is None else fam)
    return result


def _tv_to_uniform(counts: np.ndarray) -> float:
    """Total variation between an empirical distribution and the uniform one over its support."""
    total = counts.sum()
    if total == 0:
        return float('nan')
    k = len(counts)
    return float(0.5 * np.abs(counts / total - 1.0 / k).sum())


def _tv(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    ta, tb = counts_a.sum(), counts_b.sum()
    if ta == 0 or tb == 0:
        return float('nan')
    return float(0.5 * np.abs(counts_a / ta - counts_b / tb).sum())


def _normalized_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return float('nan')
    k = len(counts)
    if k <= 1:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum() / math.log(k))


def order_histogram(orders: Sequence[Sequence[int]], n_strokes: int) -> np.ndarray:
    """Counts over the n! permutations, indexed by `strategy.order_rank`."""
    counts = np.zeros(math.factorial(n_strokes), dtype=np.float64)
    for o in orders:
        counts[order_rank(list(o))] += 1
    return counts


def strategy_metrics(realized: Sequence[RealizedStrategy],
                     n_strokes: int,
                     families: Sequence[str] = DEFAULT_PROFILE_FAMILIES,
                     coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
                     reference_orders: Optional[Sequence[Sequence[int]]] = None,
                     ) -> Dict[str, float]:
    """
    Score a set of rollouts of ONE task against the generator's strategy distribution.

    The reference is analytic, not empirical: `strategy.sample_strategy` draws the stroke order
    with `rng.permutation` and each stroke's profile with `rng.choice`, both uniform. Comparing
    against the task's own demos instead would compare against an 8-sample draw from that uniform;
    pass them as `reference_orders` to get that comparison alongside, as a check that the demos
    really do look uniform.

    Every strategy metric is computed over COMPLETE rollouts only -- see
    `DEFAULT_COVERAGE_THRESHOLD`. `valid_rate` reports how many survived, and must be read first:
    strategy numbers over a handful of rollouts say little, and say nothing at all about a policy
    that mostly fails to draw.
    """
    out: Dict[str, float] = {}
    n_total = len(realized)
    if n_total == 0:
        return out

    out['coverage_mean'] = float(np.mean([r.coverage for r in realized]))
    out['off_path_rate'] = float(np.mean([r.off_path_rate for r in realized]))
    out['order_is_valid_permutation_rate'] = float(
        np.mean([r.is_complete(n_strokes, coverage_threshold) for r in realized]))

    valid = [r for r in realized if r.is_complete(n_strokes, coverage_threshold)]
    out['n_valid'] = float(len(valid))
    out['valid_rate'] = float(len(valid) / n_total)
    if not valid:
        return out

    counts = order_histogram([r.order for r in valid], n_strokes)
    out['order_tv_to_uniform'] = _tv_to_uniform(counts)
    out['order_entropy'] = _normalized_entropy(counts)
    out['order_inversions'] = float(np.mean(
        [count_inversions(r.order) / max(n_strokes * (n_strokes - 1) / 2, 1) for r in valid]))
    if reference_orders:
        out['order_tv_to_demos'] = _tv(counts, order_histogram(reference_orders, n_strokes))

    # Profiles are per stroke, so pool every stroke of every valid rollout.
    prof = np.array([p for r in valid for p in r.profiles])
    prof = prof[prof >= 0]
    if len(prof):
        prof_counts = np.bincount(prof, minlength=len(families)).astype(np.float64)
        out['profile_tv_to_uniform'] = _tv_to_uniform(prof_counts)
        out['profile_entropy'] = _normalized_entropy(prof_counts)

    # Null-axis control: the generator's `--vary-direction` is off, so the data contains no
    # reversals at all. Anything above 0 is variation the policy invented.
    out['frac_reversed'] = float(np.mean([np.mean(r.reversed_flags) if r.reversed_flags else 0.0
                                          for r in valid]))
    return out


def all_orders(n_strokes: int) -> List[Tuple[int, ...]]:
    """The permutation support, in `order_rank` order. Handy for reporting."""
    return sorted(permutations(range(n_strokes)), key=lambda o: order_rank(list(o)))
