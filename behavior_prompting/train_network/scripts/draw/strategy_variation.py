"""
Per-demo strategy variation for procedural drawing generation.

The generator in `procedural_generate_drawings.py` samples one trajectory per task and then
varies only *execution* across that task's demos (speed, noise, dwell, board rotation). The
result is a dataset where the strong strategy axes -- stroke order, traversal direction,
velocity-profile shape -- are fixed per task, so a goal-image-conditioned policy has nothing
to be multimodal about. This module supplies the missing variation.

Four axes, in decreasing order of how confound-free they are:

1. ORDER      permute the strokes. Confound-free: no stroke's own geometry or speed changes.
2. DIRECTION  reverse a stroke's traversal. Confound-free for the same reason.
3. PROFILE    reparameterize time within each part (ease-in/out, ramp, multi-peak), applied
              as one shape family regardless of part type so it does not correlate with the
              oval-vs-line geometry.
4. PART SPEED per-part speed multipliers instead of one scalar per demo.

⚠️ 3 and 4 are NOT image-invariant. The canvas records the agent's ACHIEVED position, and the
agent is a PD-tracked body (k_p=100, k_v=20). Tangential lag (k_v/k_p)*v is a pure time delay
and repaints the same pixels, but tracking a curve of radius R at speed v requires
centripetal acceleration v^2/R, which the P term only supplies by sitting off-path by
roughly v^2/(R*k_p) -- 8 px at v=200 on R=50, 30 px at v=300 on R=30, against a 12 px pen. So
changing local speed on a curved part changes the drawing. `curvature_speed_cap` bounds that;
verify empirically with `verify_strategy_invariance.py` rather than trusting the algebra.

1 and 2 are image-invariant only if each stroke begins from the same dynamic state, since
otherwise entry velocity changes the lag at the stroke's start. The generator therefore
settles the cursor to rest at each stroke's start with the pen up before drawing it.

The geometry helpers here are pure functions of the part dictionaries defined in
`procedural_generate_drawings.py`, so they are unit-testable without pygame or the env.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Matches DrawEnv: k_p=100, k_v=20, pen line width 12.
ENV_K_P = 100.0
ENV_PEN_WIDTH = 12.0

DRAWING_PART_TYPES = ('straight', 'curve', 'oval')

PROFILE_FAMILIES = ('linear', 'ease_in', 'ease_out', 'ease_in_out', 'two_peak')


# ---------------------------------------------------------------------------------------
# time reparameterization (the PROFILE axis)
# ---------------------------------------------------------------------------------------

def apply_profile(u: np.ndarray, family: str = 'linear', strength: float = 0.7) -> np.ndarray:
    """
    Map a uniform parameter u in [0, 1] through a monotone easing function s(u).

    Monotonicity is what keeps the geometry untouched: s only redistributes *when* samples
    land along the part, never reorders them, so the drawn path is identical and only the
    speed profile along it changes. s(0)=0 and s(1)=1 so part endpoints are preserved
    exactly -- important because the generator asserts each part ends where it should and
    because a stroke's endpoints are where the cursor settles.

    Deliberately applied identically to straights, curves and ovals. Making the family
    independent of part type is what decouples this axis from geometry; if instead ovals got
    one profile and lines another, the axis would just be a proxy for part type (which is a
    task property, and was already showing up as an ICC of ~0.5 in the measured data).

    strength scales the departure from linear, in [0, 1]. Two-peak stays monotone for
    strength <= 1 because d/du [u - strength*sin(4*pi*u)/(4*pi)] = 1 - strength*cos(4*pi*u).
    """
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    a = float(np.clip(strength, 0.0, 1.0))

    if family == 'linear' or a == 0.0:
        s = u
    elif family == 'ease_in':
        s = (1 - a) * u + a * u ** 2
    elif family == 'ease_out':
        s = (1 - a) * u + a * (1 - (1 - u) ** 2)
    elif family == 'ease_in_out':
        s = (1 - a) * u + a * (3 * u ** 2 - 2 * u ** 3)
    elif family == 'two_peak':
        s = u - a * np.sin(4 * math.pi * u) / (4 * math.pi)
    else:
        raise ValueError(f"unknown profile family '{family}'; expected one of {PROFILE_FAMILIES}")

    # guard the invariants the callers rely on
    s = np.clip(s, 0.0, 1.0)
    if s.size:
        s[0], s[-1] = 0.0, 1.0
    return s


def profile_is_monotone(family: str, strength: float, n: int = 512) -> bool:
    """Check s(u) is non-decreasing -- a regression guard on new families."""
    s = apply_profile(np.linspace(0.0, 1.0, n), family, strength)
    return bool(np.all(np.diff(s) >= -1e-12))


# ---------------------------------------------------------------------------------------
# direction reversal (the DIRECTION axis)
# ---------------------------------------------------------------------------------------

def reverse_part(part: Dict) -> Dict:
    """
    Return a part that traces exactly the same geometry in the opposite direction.

    straight  swap the endpoints.
    curve     swap the endpoints and swap the two Bezier control points. A cubic Bezier
              reversed is B(1-t), which is the same curve with (P0,P1,P2,P3) reversed.
    oval      start at the old end angle and sweep back: start_angle += angular_range and
              angular_range negates. The ellipse centre is reconstructed at action-generation
              time from (start_pos, start_angle), and it is invariant under this change:
                  centre' = P_end - R*e(theta_s + delta) = centre,
              since P_end = centre + R*e(theta_s + delta). So the traced ellipse is identical.

    In every case the reversed part's end_pos is the original's start_pos.
    """
    p = dict(part)
    kind = part['part_type']

    if kind in ('straight', 'movement'):
        p['start_pos'] = np.array(part['end_pos'], dtype=np.float64)
        p['end_pos'] = np.array(part['start_pos'], dtype=np.float64)
    elif kind == 'curve':
        p['start_pos'] = np.array(part['end_pos'], dtype=np.float64)
        p['end_pos'] = np.array(part['start_pos'], dtype=np.float64)
        p['control1'] = np.array(part['control2'], dtype=np.float64)
        p['control2'] = np.array(part['control1'], dtype=np.float64)
    elif kind == 'oval':
        delta = float(part['angular_range'])
        p['start_pos'] = np.array(part['end_pos'], dtype=np.float64)
        p['end_pos'] = np.array(part['start_pos'], dtype=np.float64)
        p['start_angle'] = float(part['start_angle']) + delta
        p['angular_range'] = -delta
    else:
        raise ValueError(f"cannot reverse part of type '{kind}'")

    return p


# ---------------------------------------------------------------------------------------
# strokes: the unit of ordering
# ---------------------------------------------------------------------------------------

def group_parts_into_strokes(parts: Sequence[Dict], tol: float = 1e-6) -> List[List[Dict]]:
    """
    Group a chained part list into strokes, where a stroke is a maximal run of *drawing*
    parts joined end-to-start.

    A stroke, not a part, is the unit of ordering. Parts inside a chain share endpoints --
    they are one continuous pen-down run, like the sides of a polygon -- so permuting them
    individually would either break the chain or require pen lifts that change the drawing's
    pen-up structure. Explicit `movement` parts are pen-up bridges and are dropped: bridges
    are regenerated per demo to connect whatever order was sampled.

    Note this is exactly the distinction that made the earlier stroke-order measurement come
    out undefined on the shipped dataset -- 320 of 500 demos were a single stroke because the
    parts within them were all connected.
    """
    strokes: List[List[Dict]] = []
    current: List[Dict] = []

    for part in parts:
        if part['part_type'] == 'movement':
            if current:
                strokes.append(current)
                current = []
            continue
        if current:
            prev_end = np.asarray(current[-1]['end_pos'], dtype=np.float64)
            this_start = np.asarray(part['start_pos'], dtype=np.float64)
            if np.linalg.norm(prev_end - this_start) > tol:
                strokes.append(current)
                current = []
        current.append(part)

    if current:
        strokes.append(current)
    return strokes


def reverse_stroke(stroke: Sequence[Dict]) -> List[Dict]:
    """Reverse a stroke: reverse the part order and reverse each part."""
    return [reverse_part(p) for p in reversed(list(stroke))]


def stroke_start(stroke: Sequence[Dict]) -> np.ndarray:
    return np.asarray(stroke[0]['start_pos'], dtype=np.float64)


def stroke_end(stroke: Sequence[Dict]) -> np.ndarray:
    return np.asarray(stroke[-1]['end_pos'], dtype=np.float64)


# ---------------------------------------------------------------------------------------
# curvature-aware speed cap (bounds the PROFILE / SPEED image leak)
# ---------------------------------------------------------------------------------------

def part_min_radius(part: Dict) -> float:
    """
    Smallest radius of curvature on a part, or inf for a straight line.

    For an ellipse with semi-axes a >= b the tightest curvature is at the end of the major
    axis, where R = b^2 / a. Cubic Beziers are bounded crudely by the chord length, which is
    enough for a speed cap.
    """
    kind = part['part_type']
    if kind == 'straight':
        return float('inf')
    if kind == 'oval':
        a = max(float(part['major_axis']), float(part['minor_axis']))
        b = min(float(part['major_axis']), float(part['minor_axis']))
        return (b ** 2) / a if a > 1e-9 else float('inf')
    if kind == 'curve':
        # Numerically: R = 1 / max|kappa|, kappa = |x'y'' - y'x''| / |r'|^3.
        #
        # A chord-length bound is wrong here, badly. The generator's connection_probability lets a
        # curve end back at its own start, giving a chord of ~0 -- and chord/2 then reports a
        # sub-pixel radius for what is actually a large loop. On real data that produced
        # max_radial_error_px readings of 1731 px on a 512 px board, and would make the speed cap
        # throttle those parts to almost zero.
        pts = sample_part_points(part, 96)
        d1 = np.gradient(pts, axis=0)
        d2 = np.gradient(d1, axis=0)
        speed = np.linalg.norm(d1, axis=1)
        cross = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        ok = speed > 1e-9
        if not np.any(ok):
            return float('inf')
        kappa = cross[ok] / speed[ok] ** 3
        kappa_max = float(np.max(kappa))
        return (1.0 / kappa_max) if kappa_max > 1e-12 else float('inf')
    return float('inf')


def curvature_speed_cap(part: Dict, tol_px: float, k_p: float = ENV_K_P) -> float:
    """
    Largest speed at which a part can be drawn while the achieved path stays within tol_px
    of the commanded path.

    Tracking a curve of radius R at speed v needs centripetal acceleration v^2/R, which the
    environment's P term supplies only by sitting off-path by e = v^2/(R*k_p). Inverting for
    e = tol_px gives v = sqrt(tol_px * R * k_p). Straight parts are uncapped.

    This is what keeps per-part speed and profile variation from rewriting the goal image.
    Concretely at tol_px=3, k_p=100: an R=30 arc caps at ~95 px/s and an R=50 arc at ~122
    px/s -- both below the generator's default 100-300 px/s range, so enabling those axes
    without a cap WILL change the rendered drawing.
    """
    R = part_min_radius(part)
    if not np.isfinite(R):
        return float('inf')
    return math.sqrt(max(tol_px, 0.0) * R * k_p)


# ---------------------------------------------------------------------------------------
# per-demo strategy sampling
# ---------------------------------------------------------------------------------------

class DemoStrategy:
    """
    The 'how' of one demo, sampled independently of the task's geometry.

    order          permutation of stroke indices, in execution order
    reversed_flags reversed_flags[i] applies to the stroke drawn i-th (i.e. post-permutation)
    profiles       (family, strength) per drawn stroke
    speed_mults    per-part multiplier on the demo's base speed, per drawn stroke
    base_speed     the demo's base speed in px/s
    """

    def __init__(self, order, reversed_flags, profiles, speed_mults, base_speed):
        self.order = list(order)
        self.reversed_flags = list(reversed_flags)
        self.profiles = list(profiles)
        self.speed_mults = [list(m) for m in speed_mults]
        self.base_speed = float(base_speed)

    def __repr__(self):
        fams = ','.join(f for f, _ in self.profiles)
        return (f'DemoStrategy(order={self.order}, reversed={self.reversed_flags}, '
                f'profiles=[{fams}], base_speed={self.base_speed:.1f})')

    def summary_labels(self) -> Dict[str, float]:
        """Per-demo scalars suitable for broadcasting into the replay buffer labels."""
        n = max(len(self.order), 1)
        return {
            'strategy_n_strokes': float(len(self.order)),
            'strategy_order_rank': float(order_rank(self.order)),
            'strategy_frac_reversed': float(np.mean(self.reversed_flags)) if self.reversed_flags else 0.0,
            'strategy_profile_id': float(PROFILE_FAMILIES.index(self.profiles[0][0])) if self.profiles else 0.0,
            'strategy_base_speed': self.base_speed,
            'strategy_order_first': float(self.order[0]) if self.order else -1.0,
            'strategy_order_last': float(self.order[-1]) if self.order else -1.0,
            'strategy_order_inversions': float(count_inversions(self.order)) / max(n * (n - 1) / 2, 1),
        }


def count_inversions(order: Sequence[int]) -> int:
    """Number of out-of-order pairs; 0 for the identity, n(n-1)/2 for a full reversal."""
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


def sample_demo_strategy(strokes: Sequence[Sequence[Dict]],
                         base_speed: float,
                         vary_order: bool = True,
                         vary_direction: bool = True,
                         profile_families: Sequence[str] = ('linear',),
                         profile_strength: float = 0.7,
                         profile_per_stroke: bool = False,
                         part_speed_ratio: float = 1.0,
                         reverse_probability: float = 0.5) -> DemoStrategy:
    """
    Sample one demo's strategy. Uses the global numpy RNG, matching the generator's
    existing per-demo `np.random.seed(...)` scheme so results stay reproducible.

    profile_per_stroke=False draws one profile family for the whole demo, which makes the
    axis a demo-level 'cadence signature' rather than per-stroke jitter.
    part_speed_ratio=1.0 disables per-part speed variation; r>1 samples each part's
    multiplier log-uniformly in [1/r, r], so the demo's mean speed is unbiased.
    """
    n = len(strokes)
    order = list(np.random.permutation(n)) if (vary_order and n > 1) else list(range(n))
    order = [int(i) for i in order]

    reversed_flags = [bool(np.random.random() < reverse_probability) if vary_direction else False
                      for _ in range(n)]

    families = list(profile_families) if profile_families else ['linear']
    if profile_per_stroke:
        profiles = [(str(np.random.choice(families)), profile_strength) for _ in range(n)]
    else:
        one = (str(np.random.choice(families)), profile_strength)
        profiles = [one for _ in range(n)]

    speed_mults = []
    for stroke_idx in order:
        n_parts = len(strokes[stroke_idx])
        if part_speed_ratio > 1.0:
            lo, hi = math.log(1.0 / part_speed_ratio), math.log(part_speed_ratio)
            mults = [float(math.exp(np.random.uniform(lo, hi))) for _ in range(n_parts)]
        else:
            mults = [1.0] * n_parts
        speed_mults.append(mults)

    return DemoStrategy(order, reversed_flags, profiles, speed_mults, base_speed)


def apply_strategy(strokes: Sequence[Sequence[Dict]], strategy: DemoStrategy) -> List[List[Dict]]:
    """
    Realize a strategy as the concrete list of strokes to draw, in execution order.

    Reversal flags are indexed by execution position, not by original stroke index, so that
    'the third stroke I draw is drawn backwards' is expressible independently of the
    permutation.
    """
    out = []
    for exec_idx, stroke_idx in enumerate(strategy.order):
        stroke = list(strokes[stroke_idx])
        if exec_idx < len(strategy.reversed_flags) and strategy.reversed_flags[exec_idx]:
            stroke = reverse_stroke(stroke)
        out.append(stroke)
    return out


# ---------------------------------------------------------------------------------------
# geometry sanity checks (used by the generator and by the verifier)
# ---------------------------------------------------------------------------------------

def stroke_geometry_signature(stroke: Sequence[Dict], n_per_part: int = 64) -> np.ndarray:
    """
    A direction- and order-independent signature of the pixels a stroke will draw: the
    densely sampled point set, lexicographically sorted.

    Two strokes with the same signature command the same ink. Used to assert that reversing
    a stroke or permuting strokes did not alter the geometry, independently of the renderer.
    """
    pts = []
    for part in stroke:
        pts.append(sample_part_points(part, n_per_part))
    if not pts:
        return np.zeros((0, 2))
    allp = np.concatenate(pts, axis=0)
    order = np.lexsort((allp[:, 1], allp[:, 0]))
    return allp[order]


def sample_part_points(part: Dict, n: int = 64) -> np.ndarray:
    """Sample a part's commanded path at n points, uniform in its natural parameter."""
    return eval_part(part, np.linspace(0.0, 1.0, n))


def eval_part(part: Dict, u: np.ndarray) -> np.ndarray:
    """
    Evaluate a part's commanded path at arbitrary parameter values u in [0, 1].

    Separated from `sample_part_points` so the profile axis can pass a non-uniform u without
    duplicating the geometry: the path is a function of u alone, and a monotone
    reparameterization of u changes only the timing.
    """
    kind = part['part_type']
    u = np.asarray(u, dtype=np.float64)
    start = np.asarray(part['start_pos'], dtype=np.float64)
    end = np.asarray(part['end_pos'], dtype=np.float64)

    if kind in ('straight', 'movement'):
        return start[None, :] + u[:, None] * (end - start)[None, :]

    if kind == 'curve':
        c1 = np.asarray(part['control1'], dtype=np.float64)
        c2 = np.asarray(part['control2'], dtype=np.float64)
        ui = (1 - u)[:, None]
        return (ui ** 3 * start + 3 * ui ** 2 * u[:, None] * c1
                + 3 * ui * u[:, None] ** 2 * c2 + u[:, None] ** 3 * end)

    if kind == 'oval':
        a = float(part['major_axis'])
        b = float(part['minor_axis'])
        ang0 = float(part['start_angle'])
        delta = float(part['angular_range'])
        rot = float(part['major_angle'])
        cos_r, sin_r = math.cos(rot), math.sin(rot)

        def on_ellipse(theta):
            x, y = a * np.cos(theta), b * np.sin(theta)
            return np.stack([x * cos_r - y * sin_r, x * sin_r + y * cos_r], axis=-1)

        centre = start - on_ellipse(np.array(ang0)).reshape(2)
        return centre[None, :] + on_ellipse(ang0 + u * delta)

    raise ValueError(f"unknown part type '{kind}'")


# ---------------------------------------------------------------------------------------
# turning a stroke into actions
# ---------------------------------------------------------------------------------------

# Minimum sample intervals per part type, matching the old sample-count floors of 5/8/16.
MIN_INTERVALS = {'straight': 4, 'curve': 7, 'oval': 15, 'movement': 4}

# Steps of pen-down dwell needed at an interior part boundary for the env's PD tracker to
# converge, so that reversing a multi-part stroke does not change the rendered corners. The
# tracker is critically damped at omega=10 rad/s against a 10 Hz control rate, so the error
# decays by roughly e^-1 per control step; 8 steps leaves ~e^-8 of it.
MIN_CORNER_DWELL_STEPS = 8


def part_path_length(part: Dict) -> float:
    """Approximate arc length of a part, used to convert speed into a duration."""
    kind = part['part_type']
    start = np.asarray(part['start_pos'], dtype=np.float64)
    end = np.asarray(part['end_pos'], dtype=np.float64)

    if kind in ('straight', 'movement'):
        return float(np.linalg.norm(end - start))
    if kind == 'curve':
        return 1.5 * float(np.linalg.norm(end - start))  # matches the generator's estimate
    if kind == 'oval':
        a, b = float(part['major_axis']), float(part['minor_axis'])
        h = ((a - b) / (a + b)) ** 2
        perim = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
        return perim * abs(float(part['angular_range'])) / (2 * math.pi)
    raise ValueError(f"unknown part type '{kind}'")


def build_stroke_actions(stroke: Sequence[Dict],
                         control_hz: float,
                         base_speed: float,
                         speed_mults: Optional[Sequence[float]] = None,
                         profile: Tuple[str, float] = ('linear', 0.0),
                         noise_std: float = 0.0,
                         noise_bounds: Optional[float] = 20.0,
                         part_delay_min: int = 0,
                         part_delay_max: int = 0,
                         curvature_tol_px: Optional[float] = None,
                         end_hold_steps: int = 10) -> Tuple[np.ndarray, List[Dict]]:
    """
    Convert one stroke into an (T, 3) array of (x, y, pen_down) actions.

    Two deliberate departures from the original `convert_control_points_to_actions`:

    * Step counts use `n_intervals = round(duration * hz)` and interpolate over
      `i / n_intervals`, so the REALIZED speed is `length / (n_intervals / hz)` -- within
      rounding of the commanded speed. The original used `steps = int(duration * hz)` and
      then interpolated over `steps - 1`, which inflated realized speed by
      `steps / (steps - 1)` on top of a truncation: measured +4% to +28%, median +12%, and
      worst on short parts. Since part lengths are a task property, that artifact leaked
      geometry into speed -- a between-task ICC of 0.37 on an axis that is sampled per demo.

    * The realized speed of each part is returned, so the dataset can label what actually
      happened instead of what was commanded.

    `curvature_tol_px` caps each part's speed at `sqrt(tol * R * k_p)` so the achieved path
    stays within tol of the commanded path; without it, per-part speed and profile variation
    rewrite the rendered drawing. See `curvature_speed_cap`.

    `end_hold_steps` repeats the stroke's final position with the pen STILL DOWN, and must be
    > 0 for EVERY stroke, not just the last one. The trail lags the command tangentially by
    about (k_v/k_p)*v -- 40 px at 200 px/s -- so lifting the pen the instant the command
    reaches the stroke's end leaves that last 40 px unpainted. Measured effect of getting this
    wrong: reversing a stroke swaps which end is truncated and drops render IoU to 0.67-0.87,
    which destroys the order/direction invariance the whole design depends on. Ten steps is
    ample: the tracker is critically damped at omega=10 rad/s against a 10 Hz control rate, so
    the residual decays by ~e^-10 over the hold.
    """
    family, strength = profile
    mults = list(speed_mults) if speed_mults is not None else [1.0] * len(stroke)
    assert len(mults) == len(stroke), f'{len(mults)} speed multipliers for {len(stroke)} parts'

    actions: List[np.ndarray] = []
    part_final_indices: List[int] = []
    dwell_indices: List[int] = []
    part_info: List[Dict] = []

    for part_idx, part in enumerate(stroke):
        kind = part['part_type']
        length = part_path_length(part)
        commanded = max(base_speed * mults[part_idx], 1e-6)

        cap = curvature_speed_cap(part, curvature_tol_px) if curvature_tol_px is not None else float('inf')
        speed = min(commanded, cap)

        min_intervals = MIN_INTERVALS.get(kind, 4)
        n_intervals = max(int(round((length / speed) * control_hz)), min_intervals)
        # The floor is the irreducible remnant of the speed/geometry coupling: a short part at
        # a high speed wants fewer samples than the floor allows, so its realized speed
        # saturates at length/(min_intervals/hz). Lowering the floor is not an option -- the
        # polyline would visibly coarsen and the rendered image would then change with speed.
        # So record it, and label realized rather than commanded speed downstream.
        max_achievable = length / (min_intervals / control_hz)
        was_floored = n_intervals == min_intervals and speed > max_achievable * 1.001
        u = np.arange(n_intervals + 1, dtype=np.float64) / n_intervals
        s = apply_profile(u, family, strength)
        pts = eval_part(part, s)

        pen = float(part['pen_down'])
        for p in pts:
            actions.append(np.array([p[0], p[1], pen], dtype=np.float32))
        part_final_indices.append(len(actions) - 1)

        realized = length / (n_intervals / control_hz)
        part_info.append({
            'part_type': kind,
            'length': length,
            'commanded_speed': commanded,
            'capped_speed': speed,
            'realized_speed': realized,
            'was_capped': bool(cap < commanded),
            'was_floored': bool(was_floored),
            'max_achievable_speed': max_achievable,
            'min_radius': part_min_radius(part),
            'radial_error_px': (speed ** 2) / (part_min_radius(part) * ENV_K_P)
                               if np.isfinite(part_min_radius(part)) else 0.0,
            'n_intervals': n_intervals,
        })

        # Pen-down dwell at each interior part boundary -- a CORNER DWELL, not just padding.
        #
        # Without it the tracker carries velocity through the corner between two parts, and
        # because it is a causal filter its corner rounding is direction-dependent: the response
        # to a path traversed forwards is not the mirror of the response traversed backwards. So
        # reversing a stroke changes the rendered pixels, and measurably more for strokes with
        # more interior corners (measured: reversing a 1-part stroke gave render IoU 0.997,
        # reversing a 3-part stroke 0.919). Dwelling long enough for the tracker to converge
        # makes each part a rest-to-rest segment, which restores direction symmetry.
        #
        # MEASURED: this turned out NOT to be the cause of the direction leak. Adding an 8-step
        # dwell moved render IoU only 0.9535 -> 0.9543, and the leak decomposes exactly additively
        # per reversed stroke including single-part strokes that have no interior corner at all.
        # The real driver is radial tracking error (v^2/(R*k_p)), whose acceleration transients at
        # a stroke's two ends are not symmetric under reversal. Bound that with
        # curvature_tol_px instead. The dwell is kept because it is harmless and physically
        # sensible, but part_delay_min defaults to 0; once the radial error is bounded it is worth
        # retesting whether corners contribute a second-order effect.
        if part_idx < len(stroke) - 1 and part_delay_max > 0:
            delay = int(np.random.randint(part_delay_min, part_delay_max + 1))
            for _ in range(delay):
                dwell_indices.append(len(actions))
                actions.append(np.array([pts[-1][0], pts[-1][1], pen], dtype=np.float32))

    hold_indices: List[int] = []
    if end_hold_steps > 0 and actions:
        last = actions[-1].copy()
        for _ in range(end_hold_steps):
            hold_indices.append(len(actions))
            actions.append(last.copy())

    arr = np.asarray(actions, dtype=np.float32)

    if noise_std > 0 and arr.shape[0] > 0:
        mask = np.ones(arr.shape[0], dtype=bool)
        mask[np.asarray(part_final_indices, dtype=int)] = False  # keep part endpoints exact
        # The end hold must be noise-free too. Its whole job is to let the achieved position
        # converge to the commanded stroke end; dithering the target by ~sigma each step leaves
        # the last painted point jittering per demo, which is the same per-demo render
        # difference the hold was added to remove.
        if hold_indices:
            mask[np.asarray(hold_indices, dtype=int)] = False
        # Corner dwells are noise-free for the same reason: they exist so the tracker settles
        # exactly on the part boundary, and dithering the target defeats that.
        if dwell_indices:
            mask[np.asarray(dwell_indices, dtype=int)] = False
        n_noisy = int(mask.sum())
        if n_noisy:
            noise = np.random.normal(0, noise_std, size=(n_noisy, 2))
            if noise_bounds is not None:
                noise = np.clip(noise, -noise_bounds, noise_bounds)
            arr[mask, :2] += noise.astype(np.float32)

    return arr, part_info


def assert_geometry_preserved(original: Sequence[Sequence[Dict]],
                             realized: Sequence[Sequence[Dict]],
                             tol: float = 1e-6) -> None:
    """
    Assert that a permuted/reversed stroke set commands exactly the same ink as the original.

    Compares the multiset of sampled points across all strokes, so it is blind to both the
    order strokes are drawn in and the direction each is traced -- which is the whole point:
    those are the axes we are varying, and they must not change the drawing.
    """
    def signature(strokes):
        pts = [stroke_geometry_signature(s) for s in strokes]
        allp = np.concatenate([p for p in pts if p.size], axis=0) if pts else np.zeros((0, 2))
        order = np.lexsort((allp[:, 1], allp[:, 0]))
        return allp[order]

    a, b = signature(original), signature(realized)
    assert a.shape == b.shape, f'geometry changed: {a.shape} vs {b.shape} sampled points'
    max_dev = float(np.max(np.abs(a - b))) if a.size else 0.0
    assert max_dev <= tol, f'geometry changed: max point deviation {max_dev:.3e} px > {tol:.0e}'
