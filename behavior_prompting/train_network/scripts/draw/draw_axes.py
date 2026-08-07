"""
Extract per-episode "strategy axes" from a procedural drawing dataset and decompose
their variation into within-task vs between-task components.

The axes, and where their variation is expected to live given
`procedural_generate_drawings.py` (one trajectory per task, resampled execution per demo):

    axis                  varies within a task?   extracted as
    --------------------  ---------------------   -----------------------------------------
    rotation              yes (per demo)          boundary_angle label
    speed                 yes (per demo)          noise-debiased median step displacement
    pause structure       yes (per demo)          absolute dwell steps (speed-decoupled)
    noise / smoothness    yes (per demo)          sigma from 2nd differences of the path
    approach              yes (per demo)          pen-up prefix length / steps / straightness
    velocity profile      mostly no               within-stroke normalized speed profile,
                                                  plus across-part speed dispersion
    stroke order          no (fixed per task)     Hungarian match to the task's first demo,
                                                  then Kendall tau of the temporal order
    direction             no (fixed per task)     upright heading, signed turning, signed area

Everything directional is computed in the *upright* frame: each episode's actions are
un-rotated by its own `boundary_angle` label about the board centre. This is exact, not
estimated. Without it, the +-pi/4 per-demo board rotation lands on every directional
feature and makes direction look like a live within-task axis when it is only board pose.
`heading_raw` is emitted alongside `heading_up` so the size of that inflation is visible.

Usage:
    python draw_axes.py --dataset ../../datasets/draw/procedural_2000_10.zarr \
                        --out /tmp/draw_axes_report --max-tasks 50

`--dataset` takes a merged zarr store (what the training datasets ship as) or a directory
of per-task `*.zarr` stores (raw `procedural_generate_drawings.py` output). Start with
`--max-tasks`: a full 2000-task store is far more than the decomposition needs.

Outputs written to --out:
    episodes.csv    one row per episode, episode-level scalars
    strokes.csv     one row per (episode, stroke), with the matched stroke_slot
    profiles.npy    (n_strokes, PROFILE_LEN) resampled within-stroke speed profiles
    profiles.csv    index for profiles.npy (task, episode, stroke idx)
    axes_report.csv the within/between decomposition (also printed to stdout)
"""

import csv
import math
import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

BOARD_CENTER = 256.0
PROFILE_LEN = 24  # resampled length of a within-stroke speed profile
PATH_LEN = 64  # resampled length of the arc-length drawn-path signature
STRAIGHT_MIN_FOR_HEADING = 0.6  # below this, net-displacement heading is degenerate
PEN_THRESHOLD = 0.5
TURN_SMOOTH_WINDOW = 5  # moving-average window before measuring turning
MIN_STROKE_STEPS = 4  # strokes shorter than this are recorded but not featurized in detail


# --------------------------------------------------------------------------------------
# small geometry / circular-statistics helpers
# --------------------------------------------------------------------------------------

def wrap_to_pi(a):
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def canonicalize_xy(xy: np.ndarray, boundary_angle: float, center: float = BOARD_CENTER) -> np.ndarray:
    """
    Un-rotate a trajectory into the upright board frame.

    The generator rotates the upright trajectory by +boundary_angle about (center, center),
    so applying -boundary_angle recovers the upright trajectory exactly (up to the
    per-step noise, which is added before rotation).
    """
    xy = np.asarray(xy, dtype=np.float64)
    c, s = math.cos(-boundary_angle), math.sin(-boundary_angle)
    rel = xy - center
    out = np.empty_like(rel)
    out[:, 0] = rel[:, 0] * c - rel[:, 1] * s
    out[:, 1] = rel[:, 0] * s + rel[:, 1] * c
    return out + center


def circ_mean(angles: Sequence[float]) -> float:
    angles = np.asarray(angles, dtype=np.float64)
    if angles.size == 0:
        return float('nan')
    return float(math.atan2(np.sin(angles).mean(), np.cos(angles).mean()))


def circ_std(angles: Sequence[float]) -> float:
    """
    Circular standard deviation in radians, sqrt(-2 ln R). Comparable to a linear std
    for concentrated samples and saturating near 2.4 rad for uniform ones.
    """
    angles = np.asarray(angles, dtype=np.float64)
    if angles.size < 2:
        return float('nan')
    r = math.hypot(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    r = min(max(r, 1e-12), 1.0)
    return float(math.sqrt(-2.0 * math.log(r)))


def robust_std(x: np.ndarray) -> float:
    """MAD-based std estimate; resistant to the part boundaries and dwell edges."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float('nan')
    mad = float(np.median(np.abs(x - np.median(x))))
    return 1.4826 * mad


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average along axis 0, edges shortened (mode='valid')."""
    if w <= 1 or x.shape[0] < w:
        return x
    kernel = np.ones(w) / w
    return np.stack([np.convolve(x[:, i], kernel, mode='valid') for i in range(x.shape[1])], axis=1)


def kendall_tau(order: Sequence[int]) -> float:
    """
    Kendall tau between `order` and the identity ordering. +1 = same order as reference,
    -1 = exactly reversed. O(n^2), which is irrelevant at n ~ 5 strokes.
    """
    o = np.asarray(order, dtype=np.int64)
    n = o.size
    if n < 2:
        return float('nan')
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if o[j] == o[i]:
                continue
            if o[j] > o[i]:
                conc += 1
            else:
                disc += 1
    total = conc + disc
    if total == 0:
        return float('nan')
    return (conc - disc) / total


# --------------------------------------------------------------------------------------
# noise estimation and segmentation
# --------------------------------------------------------------------------------------

def estimate_noise_sigma(xy: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Estimate the per-axis, per-step position noise sigma from FOURTH differences.

    The obvious estimator uses second differences -- Var(n[t+1] - 2n[t] + n[t-1]) = 6 sigma^2
    -- on the argument that a smooth clean path contributes negligibly. That argument fails
    badly on this dataset: a second difference of a curve carries curvature * step^2, which
    at a 20 px step on a 50 px-radius arc is ~8 px, dwarfing the ~2.4 sigma noise term. Used
    on the real data it reported sigma ~3.6 where the generator's default is 1.0, and since
    sigma sets the dwell threshold and the speed debias, that error propagated into three
    other axes.

    The fourth-difference operator [1, -4, 6, -4, 1] annihilates any cubic exactly, so both
    curvature and jerk drop out, leaving only O(step^4 * 4th derivative). Its noise gain is
    Var = (1 + 16 + 36 + 16 + 1) sigma^2 = 70 sigma^2. On a 50 px arc at a 20 px step the
    residual path term is ~1.3 px against a noise term of sqrt(70) sigma ~ 8.4 sigma -- a
    ~15% contamination rather than a 3x overestimate.

    Still underestimates if the generator's noise is clipped hard by `noise_bounds`.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 5:
        return float('nan')
    d4 = xy[4:] - 4 * xy[3:-1] + 6 * xy[2:-2] - 4 * xy[1:-3] + xy[:-4]
    if mask is not None:
        # a fourth difference at t spans t-2..t+2, so require all five samples valid
        m = np.asarray(mask, dtype=bool)
        valid = m[4:] & m[3:-1] & m[2:-2] & m[1:-3] & m[:-4]
        d4 = d4[valid]
    if d4.shape[0] < 4:
        return float('nan')
    per_axis = [robust_std(d4[:, i]) / math.sqrt(70.0) for i in range(d4.shape[1])]
    return float(np.mean(per_axis))


def runs_of(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Maximal [start, stop) runs where mask is True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    stops = list(np.where(edges == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        stops = stops + [mask.size]
    return list(zip(starts, stops))


class Segmentation:
    """
    Segmentation of one episode.

    approach     [start, stop) pen-up prefix before the first pen-down sample
    strokes      list of [start, stop) pen-down runs, in execution order
    transits     list of [start, stop) pen-up runs after the trajectory begins
    dwell        bool mask over steps (length T-1, indexed by step displacement)
    sigma        estimated per-axis noise
    """

    def __init__(self, approach, strokes, transits, dwell, sigma, dwell_eps, dwell_eps_capped):
        self.approach = approach
        self.strokes = strokes
        self.transits = transits
        self.dwell = dwell
        self.sigma = sigma
        self.dwell_eps = dwell_eps
        self.dwell_eps_capped = dwell_eps_capped


def segment_episode(xy: np.ndarray, pen: np.ndarray, dwell_eps_sigma: float = 4.0,
                    dwell_eps_floor: float = 1.5) -> Segmentation:
    """
    Split an episode into the pen-up approach, pen-down strokes, and pen-up transits,
    and flag dwell steps.

    A "dwell" is a step whose displacement is within noise of zero. The generator's
    inter-part delays repeat the part's end position, and those repeated actions still
    receive per-step noise, so the threshold has to scale with the estimated sigma
    rather than being a fixed epsilon.

    Caveat: a stroke here is a maximal pen-down run, which is *not* the same as a
    generator "part" -- consecutive drawing parts with zero inter-part delay merge into
    one stroke. `split_stroke_into_parts` recovers part boundaries only where a nonzero
    delay produced a detectable dwell.
    """
    xy = np.asarray(xy, dtype=np.float64)
    pen = np.asarray(pen, dtype=np.float64) > PEN_THRESHOLD
    T = xy.shape[0]

    pen_idx = np.where(pen)[0]
    if pen_idx.size == 0:
        approach = (0, T)
        traj_start = T
    else:
        traj_start = int(pen_idx[0])
        approach = (0, traj_start)

    strokes = runs_of(pen)
    transits = [(a, b) for (a, b) in runs_of(~pen) if a >= traj_start]

    sigma = estimate_noise_sigma(xy, mask=pen)
    if not np.isfinite(sigma):
        sigma = estimate_noise_sigma(xy)
    if not np.isfinite(sigma):
        sigma = 0.0

    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    # a stationary target perturbed by iid noise gives E||delta|| ~ sigma * sqrt(pi)
    dwell_eps = max(dwell_eps_sigma * sigma, dwell_eps_floor)

    # Cap the threshold against the episode's own moving step size. At low sampled speed
    # and high noise, dwell_eps_sigma * sigma can exceed the real per-step displacement,
    # and an uncapped threshold then classifies genuine motion as dwell, keeping only the
    # noise-inflated large steps and biasing the measured speed sharply upward.
    #
    # The reference must be an UPPER quantile, not the median: the fixed 10-step final hold
    # plus inter-part delays can be half of a short episode's pen-down steps, which drags
    # the median down into the dwell population and collapses the cap.
    #
    # `dwell_eps_capped` marks episodes where the cap bound, i.e. where noise is comparable
    # to per-step motion and the speed / noise / pause axes stop being separately
    # identifiable. Treat those episodes' speed as low-confidence.
    pen_steps = step[pen[:step.size]] if step.size else np.zeros(0)
    dwell_eps_capped = False
    if pen_steps.size >= 5:
        cap = 0.35 * float(np.percentile(pen_steps, 75))
        if dwell_eps_floor < cap < dwell_eps:
            dwell_eps = cap
            dwell_eps_capped = True

    dwell = step < dwell_eps

    return Segmentation(approach, strokes, transits, dwell, float(sigma), float(dwell_eps),
                        dwell_eps_capped)


def split_stroke_into_parts(sl: Tuple[int, int], dwell: np.ndarray,
                            min_dwell_len: int = 2) -> List[Tuple[int, int]]:
    """
    Split a stroke at internal dwell runs, which mark generator inter-part delays.
    Returns the moving sub-segments. Delays of 0 steps are undetectable, so this
    under-counts parts; it is a lower bound, reported as `n_parts_est`.
    """
    a, b = sl
    if b - a < 2:
        return [sl]
    local = dwell[a:max(b - 1, a)]
    parts: List[Tuple[int, int]] = []
    cursor = a
    for (ra, rb) in runs_of(local):
        if rb - ra < min_dwell_len:
            continue
        gap_start, gap_stop = a + ra, a + rb
        if gap_start - cursor >= 2:
            parts.append((cursor, gap_start + 1))
        cursor = gap_stop
    if b - cursor >= 2:
        parts.append((cursor, b))
    return parts if parts else [sl]


# --------------------------------------------------------------------------------------
# per-stroke features
# --------------------------------------------------------------------------------------

def debiased_speed(step: np.ndarray, sigma: float, control_hz: float) -> float:
    """
    Speed in px/s from a stroke's step displacements, with the noise contribution removed.

    Uses the MEDIAN step, not the root-mean-square. A measured step is
    ||clean + (n[t+1] - n[t])|| with the noise difference distributed N(0, 2 sigma^2) per
    axis. Decomposing into along- and perpendicular-to-motion components,

        step ~= (c + a) + p^2 / (2c),   a, p ~ N(0, 2 sigma^2)

    so median(step)^2 ~= c^2 + 2 sigma^2, giving c = sqrt(median^2 - 2 sigma^2).

    The RMS form (mean of squares, minus 4 sigma^2) is algebraically cleaner but is not
    robust, and that matters here: dwell steps that squeak past the dwell threshold leak a
    handful of near-zero displacements into an otherwise uniform stroke. On a short stroke
    those can be 20% of the samples, and they drag the RMS down by ~10% while barely moving
    the median. The generator's fixed 10-step final hold makes that leakage routine rather
    than rare, so robustness is worth the small approximation.

    Measured on synthetic strokes, this is unbiased to <0.4% across step sizes 10-36 px and
    sigma 1-3. Its scatter is set by the sample count: ~3.8% on a 9-step stroke, ~2% at 20
    steps, <1% at 100+. Read per-stroke speeds on short strokes accordingly; episode-level
    speed pools all strokes and is correspondingly tighter.
    """
    if step.size == 0:
        return float('nan')
    med = float(np.median(step))
    clean_sq = max(med ** 2 - 2.0 * (sigma ** 2), 0.0)
    return math.sqrt(clean_sq) * control_hz


def signed_turning(xy: np.ndarray, window: int = TURN_SMOOTH_WINDOW) -> float:
    """
    Total signed turning in radians along a path. Smoothed first, because per-step
    noise dominates raw turning: at sigma=1 px and 20 px steps the noise contributes
    ~0.07 rad of spurious turning per step, which accumulates as a random walk.

    Treat this as a *relative* descriptor, comparable across demos of the same stroke, not
    as an absolute turn count: the moving-average trims (window-1) samples, so a closed
    loop reads meaningfully under 2*pi (~4.5 rad at 30-40 samples). The trim is a fixed
    fraction of the sample count, so it varies with speed -- which is why the sign, and
    comparisons at matched stroke_slot, are what to rely on.
    """
    sm = moving_average(np.asarray(xy, dtype=np.float64), window)
    if sm.shape[0] < 3:
        return float('nan')
    d = np.diff(sm, axis=0)
    n = np.linalg.norm(d, axis=1)
    keep = n > 1e-9
    d = d[keep]
    if d.shape[0] < 2:
        return float('nan')
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.sum(wrap_to_pi(np.diff(ang))))


def signed_area_norm(xy: np.ndarray) -> float:
    """
    Signed shoelace area normalized isoperimetrically: 4*pi*A / P^2.

    +1 for a counter-clockwise circle, -1 for clockwise, ~0 for a straight line. This is
    the traversal-direction descriptor that still works for closed strokes (ovals), where
    net displacement -- and therefore heading -- is degenerate.

    Sign is in image coordinates (y down), so the visual handedness is mirrored. Only
    consistency across episodes matters here.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 3:
        return float('nan')
    closed = np.vstack([xy, xy[:1]])  # close the polygon so the closing edge counts
    x2, y2 = closed[:, 0], closed[:, 1]
    area = 0.5 * float(np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1]))
    perim = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
    if perim <= 1e-9:
        return float('nan')
    return 4.0 * math.pi * area / (perim ** 2)


def resample_by_arclength(xy: np.ndarray, n: int = PATH_LEN) -> Optional[np.ndarray]:
    """
    Resample a path to n points equally spaced along its arc length.

    Arc-length (rather than time) parameterization strips speed, dwell and noise-timing out
    of the geometry, leaving a shape-and-order signature that two demos of the same task can
    be compared pointwise. Returns None for a degenerate (zero-length) path.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 2:
        return None
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-9:
        return None
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], axis=1)


def resample_profile(v: np.ndarray, length: int = PROFILE_LEN) -> np.ndarray:
    """Resample a 1-D sequence to fixed length on a normalized time axis."""
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return np.full(length, np.nan)
    if v.size == 1:
        return np.full(length, float(v[0]))
    src = np.linspace(0.0, 1.0, v.size)
    dst = np.linspace(0.0, 1.0, length)
    return np.interp(dst, src, v)


def moving_polyline(xy: np.ndarray, sl: Tuple[int, int], dwell: np.ndarray) -> np.ndarray:
    """
    The stroke as a polyline with stationary samples dropped.

    This matters more than it looks: the generator's inter-part delays and its fixed
    10-step final hold repeat a position, and those repeats still carry per-step noise.
    Left in, they inject spurious tangent directions (wrecking `signed_turning`), pad the
    perimeter, and pull the centroid toward whichever endpoint was held -- which would
    then corrupt the cross-demo stroke matching that the order axis depends on.
    """
    a, b = sl
    seg = xy[a:b]
    if seg.shape[0] < 2:
        return seg
    n_steps = seg.shape[0] - 1
    dwell_local = dwell[a:a + n_steps]
    keep = np.ones(seg.shape[0], dtype=bool)
    keep[1:] = ~dwell_local  # drop the destination sample of every stationary step
    return seg[keep]


def stroke_features(xy_up: np.ndarray, sl: Tuple[int, int], dwell: np.ndarray,
                    sigma: float, control_hz: float) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Features for one stroke, computed in the upright frame. Returns (features, profile).
    """
    a, b = sl
    seg = moving_polyline(xy_up, sl, dwell)
    n_steps = max(b - a - 1, 0)
    step_all = np.linalg.norm(np.diff(xy_up[a:b], axis=0), axis=1) if b - a >= 2 else np.zeros(0)
    dwell_local = dwell[a:a + step_all.size] if step_all.size else np.zeros(0, dtype=bool)
    moving = step_all[~dwell_local] if step_all.size else np.zeros(0)

    feats: Dict[str, float] = {
        'n_steps': float(n_steps),
        'n_moving_steps': float(moving.size),
        'dwell_steps': float(int(dwell_local.sum())) if step_all.size else 0.0,
        'path_len': float(np.sum(moving)) if moving.size else 0.0,
        'centroid_x': float(np.mean(seg[:, 0])) if seg.size else float('nan'),
        'centroid_y': float(np.mean(seg[:, 1])) if seg.size else float('nan'),
        # Endpoints in the upright frame. Emitted so a downstream consumer can reconstruct the
        # pen-up routing geometry -- the distance the pen must travel between strokes for a
        # given order -- which is what the transit residual is measured against. These are
        # execution-order endpoints (start = where the pen went down), so under a reversed
        # stroke they swap; with direction fixed per task they are pure task geometry.
        'start_x': float(seg[0, 0]) if seg.size else float('nan'),
        'start_y': float(seg[0, 1]) if seg.size else float('nan'),
        'end_x': float(seg[-1, 0]) if seg.size else float('nan'),
        'end_y': float(seg[-1, 1]) if seg.size else float('nan'),
    }

    if seg.shape[0] < MIN_STROKE_STEPS or moving.size == 0:
        for k in ('speed_px_s', 'net_disp', 'heading_up', 'straightness', 'signed_area_norm',
                  'total_turning', 'vprof_cv', 'vprof_peak', 'vprof_slow_frac',
                  'n_parts_est', 'part_speed_iqr_rel'):
            feats[k] = float('nan')
        return feats, np.full(PROFILE_LEN, np.nan)

    net = seg[-1] - seg[0]
    net_disp = float(np.linalg.norm(net))
    feats['speed_px_s'] = debiased_speed(moving, sigma, control_hz)
    feats['net_disp'] = net_disp
    feats['heading_up'] = float(math.atan2(net[1], net[0]))
    feats['straightness'] = net_disp / feats['path_len'] if feats['path_len'] > 1e-9 else float('nan')
    feats['signed_area_norm'] = signed_area_norm(seg)
    feats['total_turning'] = signed_turning(seg)

    # within-stroke velocity profile, normalized so speed itself is factored out
    med = float(np.median(moving))
    if med > 1e-9:
        v = moving / med
        q75, q25 = np.percentile(v, [75, 25])
        feats['vprof_cv'] = float((q75 - q25) / 1.349)  # robust CV, median-normalized
        feats['vprof_peak'] = float(np.percentile(v, 95))
        feats['vprof_slow_frac'] = float(np.mean(v < 0.5))
        profile = resample_profile(v)
    else:
        feats['vprof_cv'] = feats['vprof_peak'] = feats['vprof_slow_frac'] = float('nan')
        profile = np.full(PROFILE_LEN, np.nan)

    # across-part speed dispersion: the generator's per-part step-count floors
    # (max(steps,5) straight, 8 curve, 16 oval) saturate for short parts at high sampled
    # speed, so a fast demo is NOT a uniform time-rescale of a slow one.
    parts = split_stroke_into_parts(sl, dwell)
    part_speeds = []
    for (pa, pb) in parts:
        ps = np.linalg.norm(np.diff(xy_up[pa:pb], axis=0), axis=1)
        pd = dwell[pa:pa + ps.size] if ps.size else np.zeros(0, dtype=bool)
        pm = ps[~pd] if ps.size else np.zeros(0)
        if pm.size >= 2:
            part_speeds.append(debiased_speed(pm, sigma, control_hz))
    feats['n_parts_est'] = float(len(parts))
    if len(part_speeds) >= 2:
        q75, q25 = np.percentile(part_speeds, [75, 25])
        m = float(np.median(part_speeds))
        feats['part_speed_iqr_rel'] = float((q75 - q25) / m) if m > 1e-9 else float('nan')
    else:
        feats['part_speed_iqr_rel'] = float('nan')

    return feats, profile


# --------------------------------------------------------------------------------------
# per-episode features
# --------------------------------------------------------------------------------------

def episode_features(action: np.ndarray, boundary_angle: float, control_hz: float = 10.0,
                     canonicalize: bool = True
                     ) -> Tuple[Dict[str, float], List[Dict[str, float]], np.ndarray, Optional[np.ndarray]]:
    """
    Extract axes from one episode.

    action:         (T, 3) array of (x, y, pen_down) -- the commanded trajectory
    boundary_angle: the episode's board rotation in radians (from the labels)

    Returns (episode_features, per_stroke_features, profiles[(n_strokes, PROFILE_LEN)],
    drawn_path[(PATH_LEN, 2)] or None) -- the last being the arc-length path signature used
    for the order/direction axis.
    """
    action = np.asarray(action, dtype=np.float64)
    assert action.ndim == 2 and action.shape[1] == 3, f'expected (T, 3) action, got {action.shape}'
    xy_raw = action[:, :2]
    pen = action[:, 2]
    xy_up = canonicalize_xy(xy_raw, boundary_angle) if canonicalize else xy_raw

    seg = segment_episode(xy_up, pen)
    T = action.shape[0]
    step_up = np.linalg.norm(np.diff(xy_up, axis=0), axis=1)

    ep: Dict[str, float] = {
        'boundary_angle': float(boundary_angle),
        'total_steps': float(T),
        'noise_sigma': seg.sigma,
        'dwell_eps': seg.dwell_eps,
        'dwell_eps_capped': float(seg.dwell_eps_capped),
        'n_strokes': float(len(seg.strokes)),
        'n_transits': float(len(seg.transits)),
        # the pen's pose at t=0, upright frame -- the origin the approach is measured from
        'start_x': float(xy_up[0, 0]),
        'start_y': float(xy_up[0, 1]),
    }

    # --- approach (pen-up prefix) ---
    a0, a1 = seg.approach
    ap = xy_up[a0:a1]
    if ap.shape[0] >= 2:
        ap_step = np.linalg.norm(np.diff(ap, axis=0), axis=1)
        ep['approach_steps'] = float(a1 - a0)
        ep['approach_path_len'] = float(np.sum(ap_step))
        net = float(np.linalg.norm(ap[-1] - ap[0]))
        ep['approach_straightness'] = net / ep['approach_path_len'] if ep['approach_path_len'] > 1e-9 else float('nan')
    else:
        ep['approach_steps'] = float(a1 - a0)
        ep['approach_path_len'] = float('nan')
        ep['approach_straightness'] = float('nan')

    traj_start = a1
    ep['traj_steps'] = float(T - traj_start)

    # --- per-stroke ---
    stroke_rows: List[Dict[str, float]] = []
    profiles: List[np.ndarray] = []
    drawn_pieces: List[np.ndarray] = []
    for i, sl in enumerate(seg.strokes):
        piece = moving_polyline(xy_up, sl, seg.dwell)
        if piece.shape[0] >= 2:
            drawn_pieces.append(piece)
        f, prof = stroke_features(xy_up, sl, seg.dwell, seg.sigma, control_hz)
        f['stroke_idx'] = float(i)
        f['start'] = float(sl[0])
        f['stop'] = float(sl[1])
        # raw-frame heading is the upright heading plus the board rotation, exactly;
        # emitted so the rotation-induced inflation of a directional axis is visible.
        f['heading_raw'] = float(wrap_to_pi(f['heading_up'] + boundary_angle)) \
            if np.isfinite(f.get('heading_up', float('nan'))) else float('nan')
        stroke_rows.append(f)
        profiles.append(prof)

    def _agg(key, weights_key='n_moving_steps', how='wmean'):
        vals = np.array([r.get(key, np.nan) for r in stroke_rows], dtype=np.float64)
        w = np.array([r.get(weights_key, 0.0) for r in stroke_rows], dtype=np.float64)
        ok = np.isfinite(vals) & (w > 0)
        if not np.any(ok):
            return float('nan')
        if how == 'wmean':
            return float(np.sum(vals[ok] * w[ok]) / np.sum(w[ok]))
        return float(np.nanmedian(vals[ok]))

    ep['speed_px_s'] = _agg('speed_px_s')
    ep['vprof_cv'] = _agg('vprof_cv')
    ep['vprof_peak'] = _agg('vprof_peak')
    ep['vprof_slow_frac'] = _agg('vprof_slow_frac')
    ep['part_speed_iqr_rel'] = _agg('part_speed_iqr_rel')
    ep['n_parts_est'] = float(np.nansum([r.get('n_parts_est', np.nan) for r in stroke_rows]))
    ep['drawn_path_len'] = float(np.nansum([r.get('path_len', np.nan) for r in stroke_rows]))

    # --- pause structure ---
    # absolute dwell steps is the speed-decoupled measure: the generator samples
    # inter-part delays in *steps* and holds a fixed 10 steps at the end, while stroke
    # duration scales as 1/speed. So dwell_frac is a deterministic function of sampled
    # speed and must not be read as an independent axis; report both.
    traj_dwell = seg.dwell[traj_start:] if traj_start < seg.dwell.size else np.zeros(0, dtype=bool)
    dwell_runs = [(b - a) for (a, b) in runs_of(traj_dwell)]
    ep['dwell_steps'] = float(int(traj_dwell.sum()))
    ep['n_dwells'] = float(len(dwell_runs))
    ep['mean_dwell_len'] = float(np.mean(dwell_runs)) if dwell_runs else 0.0
    ep['max_dwell_len'] = float(np.max(dwell_runs)) if dwell_runs else 0.0
    ep['dwell_frac'] = float(traj_dwell.mean()) if traj_dwell.size else float('nan')

    # --- transits (pen-up moves between strokes) ---
    tr_lens, tr_straight = [], []
    for (ta, tb) in seg.transits:
        t = xy_up[ta:tb]
        if t.shape[0] < 2:
            continue
        L = float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)))
        tr_lens.append(L)
        if L > 1e-9:
            tr_straight.append(float(np.linalg.norm(t[-1] - t[0]) / L))
    ep['transit_path_len'] = float(np.sum(tr_lens)) if tr_lens else 0.0
    ep['transit_straightness'] = float(np.mean(tr_straight)) if tr_straight else float('nan')

    prof_arr = np.stack(profiles) if profiles else np.zeros((0, PROFILE_LEN))
    drawn_path = resample_by_arclength(np.concatenate(drawn_pieces, axis=0)) if drawn_pieces else None
    return ep, stroke_rows, prof_arr, drawn_path


# --------------------------------------------------------------------------------------
# stroke matching and order features (within a task)
# --------------------------------------------------------------------------------------

def match_to_reference(ref_centroids: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each stroke to a reference stroke by upright centroid, minimizing total
    distance (Hungarian). Returns slot[i] = reference index for stroke i, or -1.

    Centroid matching is adequate because within a task every demo draws the *same*
    upright geometry -- only the execution differs. It is not a shape matcher and should
    not be used across tasks.
    """
    n, m = centroids.shape[0], ref_centroids.shape[0]
    slots = -np.ones(n, dtype=np.int64)
    if n == 0 or m == 0:
        return slots
    cost = np.linalg.norm(centroids[:, None, :] - ref_centroids[None, :, :], axis=2)
    cost = np.where(np.isfinite(cost), cost, 1e9)
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        slots[r] = c
    return slots


def add_path_order_features(task_episodes: List[Tuple], ) -> None:
    """
    Segmentation-free order/direction agreement, written onto each episode row.

    This exists because the stroke-level order axis is close to unmeasurable on this data:
    a stroke is a pen-down run, but the generator separates consecutive *drawing* parts with
    dwells while the pen stays down, so 320 of 500 real demos are a single stroke and
    `order_tau` is simply undefined for them.

    Instead compare each demo's arc-length path signature against the task's first demo,
    pointwise, forward and reversed:

        order_agreement = (dev_reversed - dev_forward) / (dev_reversed + dev_forward)

    +1 means the demo traverses the same geometry in the same order and direction as the
    reference, -1 means it is traversed backwards, and ~0 means the comparison cannot tell
    (a shape symmetric under reversal, e.g. a single closed oval). `path_dev_fwd_norm` is
    the forward deviation in units of the reference path's RMS radius, so a large value
    means the demo is not drawing the same geometry at all rather than merely reordering it.
    """
    entries = [(ep, path) for (_, _, ep, path) in task_episodes]
    ref_path = entries[0][1] if entries else None
    if ref_path is None:
        for ep, _ in entries:
            ep['order_agreement'] = float('nan')
            ep['path_dev_fwd_norm'] = float('nan')
        return

    scale = float(np.sqrt(np.mean(np.sum((ref_path - ref_path.mean(axis=0)) ** 2, axis=1))))
    scale = scale if scale > 1e-9 else float('nan')
    rev = ref_path[::-1]

    for ep, path in entries:
        if path is None or path.shape != ref_path.shape:
            ep['order_agreement'] = float('nan')
            ep['path_dev_fwd_norm'] = float('nan')
            continue
        dev_f = float(np.mean(np.linalg.norm(path - ref_path, axis=1)))
        dev_r = float(np.mean(np.linalg.norm(path - rev, axis=1)))
        total = dev_f + dev_r
        ep['order_agreement'] = ((dev_r - dev_f) / total) if total > 1e-9 else float('nan')
        ep['path_dev_fwd_norm'] = dev_f / scale


def add_order_features(task_episodes: List[Tuple[str, List[Dict[str, float]], Dict[str, float]]]) -> None:
    """
    For one task: match every episode's strokes to the first episode's strokes, then
    write `stroke_slot` onto each stroke row and order agreement onto each episode row.

    Mutates the passed rows in place. `task_episodes` is a list of
    (episode_name, stroke_rows, episode_row, drawn_path) in dataset order; the first entry
    is the reference.

    NOTE: `order_tau` here is stroke-level and is undefined for single-stroke episodes,
    which are the majority on this dataset -- see `add_path_order_features` for the measure
    that actually works.
    """
    if not task_episodes:
        return

    def centroids_of(rows):
        return np.array([[r['centroid_x'], r['centroid_y']] for r in rows], dtype=np.float64) \
            if rows else np.zeros((0, 2))

    ref_rows = task_episodes[0][1]
    ref_c = centroids_of(ref_rows)
    for i, r in enumerate(ref_rows):
        r['stroke_slot'] = float(i)
    task_episodes[0][2]['order_tau'] = 1.0 if len(ref_rows) > 1 else float('nan')
    task_episodes[0][2]['order_exact'] = 1.0
    task_episodes[0][2]['stroke_count_mismatch'] = 0.0
    task_episodes[0][2]['is_order_reference'] = 1.0

    for _, rows, ep, _path in task_episodes[1:]:
        slots = match_to_reference(ref_c, centroids_of(rows))
        for r, s in zip(rows, slots):
            r['stroke_slot'] = float(s)
        seq = [int(s) for s in slots if s >= 0]
        ep['order_tau'] = kendall_tau(seq)
        ep['order_exact'] = float(seq == sorted(seq)) if seq else float('nan')
        ep['stroke_count_mismatch'] = float(len(rows) != len(ref_rows))
        ep['is_order_reference'] = 0.0


# --------------------------------------------------------------------------------------
# within-task vs between-task decomposition
# --------------------------------------------------------------------------------------

def straight_enough_for_heading(row: Dict) -> bool:
    """
    Keep strokes whose net displacement defines a direction.

    Net-displacement heading is degenerate on a curved or closed stroke: as straightness
    goes to 0 the endpoints converge and the heading becomes noise. On this dataset the mean
    stroke straightness is 0.39 and half the strokes are under 0.3, so leaving them in
    reported a within-task heading spread of ~0.55 rad that was pure degeneracy -- and it
    masked the rotation confound, since `heading_raw` and `heading_up` came out nearly
    identical once both were swamped by it.
    """
    s = row.get('straightness', float('nan'))
    return bool(np.isfinite(s) and s >= STRAIGHT_MIN_FOR_HEADING)


def curved_only(row: Dict) -> bool:
    """
    Keep strokes that actually curve. Turning and signed area are meaningless on a
    straight stroke: with no real curvature their value is a noise-driven random walk,
    whose per-demo resampling would otherwise register as live within-task variation on
    what is really a fixed axis.
    """
    s = row.get('straightness', float('nan'))
    return bool(np.isfinite(s) and s < 0.9)


def decompose(rows: List[Dict], feature: str, group_keys: Sequence[str] = ('task',),
              circular: bool = False, min_group: int = 2, row_filter=None) -> Dict[str, float]:
    """
    Split the variation of `feature` into within-group and between-group parts.

    within_std   pooled std within a group (across demos of the same task)
    between_std  std of the group means
    icc          between_var / (between_var + within_var); 1.0 = pure task property
    ratio        within_std / between_std; ~0 = no live within-task variation

    For stroke-level features pass group_keys=('task', 'stroke_slot') so strokes are
    compared slot-to-slot rather than pooled across different strokes of a drawing.
    """
    groups: Dict[Tuple, List[float]] = {}
    for r in rows:
        if row_filter is not None and not row_filter(r):
            continue
        v = r.get(feature, float('nan'))
        if v is None or not np.isfinite(v):
            continue
        key = tuple(r.get(k) for k in group_keys)
        if any(k is None for k in key):
            continue
        groups.setdefault(key, []).append(float(v))

    usable = {k: np.asarray(v) for k, v in groups.items() if len(v) >= min_group}
    if len(usable) < 2:
        return {'feature': feature, 'n_groups': len(usable), 'n_obs': sum(len(v) for v in groups.values()),
                'mean': float('nan'), 'within_std': float('nan'), 'between_std': float('nan'),
                'icc': float('nan'), 'ratio': float('nan'), 'circular': float(circular)}

    if circular:
        within = float(np.mean([circ_std(v) ** 2 for v in usable.values()]))
        between = circ_std([circ_mean(v) for v in usable.values()]) ** 2
    else:
        within = float(np.mean([np.var(v, ddof=1) for v in usable.values()]))
        between = float(np.var([np.mean(v) for v in usable.values()], ddof=1))

    all_vals = np.concatenate([v for v in usable.values()])
    grand_mean = circ_mean(all_vals) if circular else float(np.mean(all_vals))

    total = within + between
    return {
        'feature': feature,
        'n_groups': len(usable),
        'n_obs': int(sum(len(v) for v in usable.values())),
        'mean': grand_mean,
        'within_std': math.sqrt(within) if np.isfinite(within) else float('nan'),
        'between_std': math.sqrt(between) if np.isfinite(between) else float('nan'),
        'icc': (between / total) if total > 0 else float('nan'),
        'ratio': (math.sqrt(within / between) if between > 0 and np.isfinite(within) else float('inf')),
        'circular': float(circular),
    }


# episode-level axes: (feature, circular, label), grouped by task
EPISODE_AXES: List[Tuple[str, bool, str]] = [
    ('boundary_angle', True, 'rotation'),
    ('speed_px_s', False, 'speed'),
    ('dwell_steps', False, 'pause (absolute)'),
    ('dwell_frac', False, 'pause (frac, speed-coupled)'),
    ('n_dwells', False, 'pause count'),
    ('noise_sigma', False, 'noise'),
    ('approach_steps', False, 'approach length'),
    ('approach_path_len', False, 'approach path'),
    ('approach_straightness', False, 'approach shape'),
    ('transit_path_len', False, 'transit path'),
    ('vprof_cv', False, 'velocity profile (CV)'),
    ('vprof_peak', False, 'velocity profile (peak)'),
    ('vprof_slow_frac', False, 'velocity profile (slow frac)'),
    ('part_speed_iqr_rel', False, 'across-part speed dispersion'),
    ('order_agreement', False, 'order+direction (path vs demo 0)'),
    ('path_dev_fwd_norm', False, 'geometry deviation vs demo 0'),
    ('order_tau', False, 'stroke order (tau, multi-stroke only)'),
    ('n_strokes', False, 'stroke count'),
    ('drawn_path_len', False, 'drawn path length'),
    ('total_steps', False, 'episode length'),
]

# stroke-level axes: (feature, circular, label, row_filter), grouped by (task, stroke_slot)
STROKE_AXES: List[Tuple[str, bool, str, object]] = [
    ('heading_up', True, 'direction (upright, straight only)', straight_enough_for_heading),
    ('heading_raw', True, 'direction (raw frame, straight only -- rotation confound)',
     straight_enough_for_heading),
    ('total_turning', False, 'direction (signed turning, curved only)', curved_only),
    ('signed_area_norm', False, 'direction (signed area, curved only)', curved_only),
    ('speed_px_s', False, 'speed (per stroke)', None),
    ('vprof_cv', False, 'velocity profile (per stroke)', None),
    ('straightness', False, 'stroke straightness', None),
    ('path_len', False, 'stroke path length', None),
]


# --------------------------------------------------------------------------------------
# dataset loading
# --------------------------------------------------------------------------------------

def load_episodes(dataset_path: str, max_tasks: Optional[int] = None,
                  only_task_names: Optional[Sequence[str]] = None
                  ) -> Iterator[Tuple[str, str, np.ndarray, float]]:
    """
    Yield (task_name, episode_name, action, boundary_angle) for every demo.

    Accepts a single merged zarr store (e.g. datasets/draw/procedural_2000_10.zarr, which
    holds every task in one store) -- that is the layout the training datasets ship in,
    after `group_demos.py` merges the per-task stores that
    `procedural_generate_drawings.py` writes. A directory of per-task `*.zarr` stores, the
    raw generator output, also works.

    Iterates over *tasks* and slices the underlying arrays directly rather than calling
    `get_episode`, which rescans all task boundaries per episode -- O(n_tasks) per call,
    so ~40M iterations on a 2000-task store. Only `action` and the `boundary_angle` label
    are touched; `image` and `drawing_image` are never read, and they are essentially all
    of the ~40 GB on disk.

    The replay buffer is imported lazily so the feature functions above stay usable
    without the project environment.
    """
    # The store's `image` array is JPEG-XL compressed, and ReplayBuffer.__init__ walks every
    # array's metadata -- so the codec must be registered even though we only ever read
    # `action`. Normally `behavior_prompting.train_network.__init__` does this, but running
    # this file as a script never executes that package __init__.
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
    from behavior_prompting.common.replay_buffer import ReplayBuffer
    register_codecs(verbose=False)

    store_paths: List[str]
    if os.path.isdir(os.path.join(dataset_path, 'meta')):
        store_paths = [dataset_path]  # a single merged store
    else:
        import glob
        store_paths = sorted(glob.glob(os.path.join(dataset_path, '*.zarr')))
        if not store_paths:
            raise FileNotFoundError(
                f'{dataset_path} is neither a zarr store (no meta/ subdirectory) nor a '
                f'directory containing *.zarr stores')

    wanted = set(only_task_names) if only_task_names else None

    for store_path in store_paths:
        if not os.path.isdir(os.path.join(store_path, 'meta')):
            print(f'skipping {os.path.basename(store_path)}: no meta/ group '
                  f'(not a task-annotated replay buffer)')
            continue
        rb = ReplayBuffer.create_from_path(store_path, mode='r')
        if 'boundary_angle' not in rb.labels:
            print(f'skipping {os.path.basename(store_path)}: no boundary_angle label')
            continue

        task_names = np.asarray(rb.task_names[:])
        task_lengths = np.asarray(rb.task_lengths[:])
        task_data_ends = np.asarray(rb.task_data_ends[:])
        task_labels_ends = np.asarray(rb.task_labels_ends[:])
        task_to_episode = rb.get_task_to_episode_idxs()
        episode_names = np.asarray(rb.episode_names[:])
        actions = rb.data['action']
        angle_label = rb.labels['boundary_angle']

        # select tasks in first-appearance order, so --max-tasks N means "the first N tasks"
        selected: Dict[str, List[int]] = {}
        for t_idx, name in enumerate(task_names):
            name = str(name)
            if wanted is not None and name not in wanted:
                continue
            if name not in selected:
                if max_tasks is not None and len(selected) >= max_tasks:
                    continue
                selected[name] = []
            selected[name].append(t_idx)

        for name, task_idxs in selected.items():
            for t_idx in task_idxs:
                d_end = int(task_data_ends[t_idx])
                d_start = d_end - int(task_lengths[t_idx])
                l_end = int(task_labels_ends[t_idx])
                l_start = l_end - int(task_lengths[t_idx])

                action = np.asarray(actions[d_start:d_end], dtype=np.float64)
                angles = np.asarray(angle_label[l_start:l_end], dtype=np.float64).reshape(-1)
                if angles.size == 0:
                    raise KeyError(f"task segment {t_idx} ('{name}') has no boundary_angle label")
                spread = float(np.ptp(angles))
                assert spread < 1e-4, (f'boundary_angle varies within task segment {t_idx} '
                                       f"('{name}'): ptp={spread}")

                ep_idx = int(task_to_episode[t_idx])
                ep_name = str(episode_names[ep_idx]) if ep_idx < episode_names.size else f'seg{t_idx}'
                yield name, ep_name, action, float(angles[0])


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _write_csv(path: str, rows: List[Dict], preferred: Sequence[str] = ()) -> None:
    if not rows:
        return
    keys = [k for k in preferred if any(k in r for r in rows)]
    keys += sorted({k for r in rows for k in r} - set(keys))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _verdict(d: Dict[str, float]) -> str:
    r = d.get('ratio', float('nan'))
    if not np.isfinite(r):
        return 'n/a'
    if r < 0.05:
        return 'FIXED per task'
    if r < 0.25:
        return 'mostly task property'
    if r < 1.0:
        return 'live within task'
    return 'dominated by within-task'


def build_report(episode_rows: List[Dict], stroke_rows: List[Dict]) -> List[Dict]:
    report = []
    for feat, circular, label in EPISODE_AXES:
        d = decompose(episode_rows, feat, ('task',), circular=circular)
        d.update({'label': label, 'level': 'episode', 'verdict': _verdict(d)})
        report.append(d)
    for feat, circular, label, row_filter in STROKE_AXES:
        d = decompose(stroke_rows, feat, ('task', 'stroke_slot'), circular=circular,
                      row_filter=row_filter)
        d.update({'label': label, 'level': 'stroke', 'verdict': _verdict(d)})
        report.append(d)
    return report


def print_report(report: List[Dict]) -> None:
    hdr = (f"{'axis':<52}{'level':<9}{'mean':>10}{'within':>10}{'between':>10}"
           f"{'ratio':>9}{'ICC':>7}  verdict")
    print('\n' + hdr)
    print('-' * len(hdr))
    for d in report:
        def fmt(x, w, p=3):
            return f'{x:>{w}.{p}f}' if np.isfinite(x) else f'{"-":>{w}}'
        print(f"{d['label']:<52}{d['level']:<9}{fmt(d.get('mean', float('nan')), 10)}"
              f"{fmt(d['within_std'], 10)}{fmt(d['between_std'], 10)}"
              f"{fmt(d['ratio'], 9)}{fmt(d['icc'], 7, 2)}  {d['verdict']}")
    print()
    print('ratio = within-task std / between-task std.  ICC = between / (between + within).')
    print('Angles (rotation, direction) use circular statistics; std saturates near 2.4 rad.')
    print()
    print('READ `mean` ALONGSIDE `ratio`. For a bounded agreement score sitting near its')
    print('ceiling -- order+direction (mean ~0.8 of a max of 1) and geometry deviation -- both')
    print('variances are measurement noise, and a ratio near 1 means "nothing varies", NOT')
    print('that strategy varies within task. The verdict column reads the ratio alone and')
    print('mislabels those rows; check the mean before believing it.')
    print()


def analyze(dataset_path: str, out_dir: str, control_hz: float = 10.0, max_tasks: Optional[int] = None,
            canonicalize: bool = True, only_task_names: Optional[Sequence[str]] = None) -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)
    if not canonicalize:
        print('WARNING: --no-canonicalize is a diagnostic mode. Directional axes will carry '
              'each demo\'s board rotation and cannot be read as strategy variation.')

    episode_rows: List[Dict] = []
    stroke_rows: List[Dict] = []
    profile_rows: List[Dict] = []
    profiles: List[np.ndarray] = []
    per_task: Dict[str, List[Tuple[str, List[Dict], Dict]]] = {}

    for n_seen, (task_name, ep_name, action, angle) in enumerate(
            load_episodes(dataset_path, max_tasks, only_task_names), start=1):
        if n_seen % 200 == 0:
            print(f'  ...{n_seen} demos across {len(per_task)} tasks')
        ep, strokes, prof, path = episode_features(action, angle, control_hz, canonicalize)
        ep.update({'task': task_name, 'episode': ep_name})
        for r in strokes:
            r.update({'task': task_name, 'episode': ep_name})
        for i in range(prof.shape[0]):
            profile_rows.append({'task': task_name, 'episode': ep_name, 'stroke_idx': i})
            profiles.append(prof[i])
        per_task.setdefault(task_name, []).append((ep_name, strokes, ep, path))
        episode_rows.append(ep)
        stroke_rows.extend(strokes)

    for task_name, entries in per_task.items():
        add_order_features(entries)
        add_path_order_features(entries)

    _write_csv(os.path.join(out_dir, 'episodes.csv'), episode_rows,
               preferred=['task', 'episode', 'boundary_angle', 'speed_px_s', 'noise_sigma',
                          'dwell_steps', 'n_strokes', 'order_tau'])
    _write_csv(os.path.join(out_dir, 'strokes.csv'), stroke_rows,
               preferred=['task', 'episode', 'stroke_idx', 'stroke_slot', 'heading_up',
                          'heading_raw', 'total_turning', 'signed_area_norm', 'speed_px_s'])
    _write_csv(os.path.join(out_dir, 'profiles.csv'), profile_rows,
               preferred=['task', 'episode', 'stroke_idx'])
    np.save(os.path.join(out_dir, 'profiles.npy'),
            np.stack(profiles) if profiles else np.zeros((0, PROFILE_LEN)))

    report = build_report(episode_rows, stroke_rows)
    _write_csv(os.path.join(out_dir, 'axes_report.csv'), report,
               preferred=['label', 'level', 'feature', 'mean', 'within_std', 'between_std',
                          'ratio', 'icc', 'n_groups', 'n_obs', 'verdict'])

    print(f'{len(episode_rows)} episodes across {len(per_task)} tasks; {len(stroke_rows)} strokes')
    print_report(report)
    print(f'wrote episodes.csv, strokes.csv, profiles.{{csv,npy}}, axes_report.csv to {out_dir}')
    return report


def main():
    import click

    @click.command()
    @click.option('--dataset', required=True,
                  help='merged .zarr store, or a directory of per-task .zarr stores')
    @click.option('--out', required=True, help='output directory for the tables and report')
    @click.option('--control-hz', default=10.0, show_default=True)
    @click.option('--max-tasks', default=None, type=int, help='use only the first N tasks')
    @click.option('--task-name', 'task_names', multiple=True, help='restrict to these task names')
    @click.option('--no-canonicalize', is_flag=True,
                  help='skip un-rotating by boundary_angle (diagnostic: shows the rotation confound)')
    def cli(dataset, out, control_hz, max_tasks, task_names, no_canonicalize):
        analyze(dataset, out, control_hz=control_hz, max_tasks=max_tasks,
                canonicalize=not no_canonicalize,
                only_task_names=list(task_names) if task_names else None)

    cli()


if __name__ == '__main__':
    main()
