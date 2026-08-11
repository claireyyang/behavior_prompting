"""
The four manners of touching the dots.

The task is fixed by the dot set; these are four legitimate ways to accomplish it. Two pairs carry
the separability structure the whole experiment rests on:

    CONNECT vs TOUCH  -- bitwise identical xy, different contact  -> separable ONLY by contact
    CONNECT vs CURVE  -- identical contact, different xy          -> separable by KINEMATICS alone

Both properties hold *by construction*, for any dot layout, so dots can be placed uniformly at
random and no instance ever needs rejecting:

  * `CURVE` bows each segment perpendicular by at least `MIN_CURVE_BOW` (see `layout._bowed_arc`),
    rather than splining through the dots. A spline's deviation depends on how the dots happen to be
    arranged and vanishes as they approach collinear; a floored bow does not.
  * `CONNECT` and `TOUCH` traverse **the same waypoint sequence, including the same dwell steps at
    each dot**. CONNECT holds the pen down throughout; TOUCH lowers it only during the dwells. Since
    z never influences xy, the two xy streams come out bitwise equal. Without the matching dwells
    TOUCH would need extra steps at the dots and the sequences would merely be geometrically
    similar, not identical.
  * `PARALLEL`'s dashes are longer than a dot diameter, so it can never collapse into `TOUCH`.

⚠️ The bitwise-identity claim holds for NOISELESS generation. Dataset repeats add independent action
noise per demo, which perturbs xy; the separability assertion is run at `noise_std=0`.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from behavior_prompting.train_network.env.draw_dot.layout import (
    MANNER_TO_ID,
    MANNERS,
    DotLayout,
    connect_path,
    curve_path,
    dash_segments,
    resample_by_arclength,
)

__all__ = ['MANNERS', 'MANNER_TO_ID', 'Demo', 'make_demo', 'make_all',
           'make_connect', 'make_touch', 'make_curve', 'make_parallel', 'path_to_demo']

Z_DOWN = 0.0
Z_UP = 1.0

DEFAULT_SPEED = 0.035      # scene units per control step; must stay <= env.max_delta
DWELL_STEPS = 2            # steps held at each dot -- see the CONNECT/TOUCH note above
FINAL_LIFT_STEPS = 2       # trailing pen-up steps so the last stroke is closed before the episode ends


@dataclass
class Demo:
    """One generated demonstration: the delta actions plus the labels the dataset stores."""
    actions: np.ndarray                  # (T, 3) float32, [dx, dy, dz]
    pen_start: np.ndarray                # (3,) float64
    manner: str
    target_dot_index: np.ndarray         # (T,) int64, original dot index this step belongs to
    xy_path: np.ndarray = field(repr=False, default=None)   # (T, 2) absolute waypoints, for tests
    z_path: np.ndarray = field(repr=False, default=None)    # (T,)  absolute z, for tests

    @property
    def n_steps(self) -> int:
        return len(self.actions)


class _Emitter:
    """Accumulates absolute waypoints and emits the deltas between them."""

    def __init__(self, pen_start: np.ndarray, speed: float):
        self.pos = np.asarray(pen_start, dtype=np.float64)[:2].copy()
        self.z = float(pen_start[2])
        self.speed = float(speed)
        self.actions: List[np.ndarray] = []
        self.targets: List[int] = []
        self.xy: List[np.ndarray] = []
        self.zs: List[float] = []

    def _emit(self, target_xy: np.ndarray, z: float, target_idx: int) -> None:
        target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
        self.actions.append(np.array([target_xy[0] - self.pos[0],
                                      target_xy[1] - self.pos[1],
                                      z - self.z], dtype=np.float64))
        self.pos = target_xy.copy()
        self.z = float(z)
        self.targets.append(int(target_idx))
        self.xy.append(self.pos.copy())
        self.zs.append(self.z)

    def goto(self, target_xy: np.ndarray, z: float, target_idx: int) -> None:
        """Step toward `target_xy` in `speed`-sized increments until it is reached."""
        target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
        while True:
            d = target_xy - self.pos
            dist = float(np.linalg.norm(d))
            if dist <= self.speed:
                self._emit(target_xy, z, target_idx)
                return
            self._emit(self.pos + d / dist * self.speed, z, target_idx)

    def follow(self, path: np.ndarray, z: float, target_idx_per_point: List[int]) -> None:
        """Traverse a dense path, resampled so every hop is one `speed`-sized step."""
        pts = resample_by_arclength(path, self.speed)
        # `target_idx_per_point` is given per ORIGINAL path vertex; map by fractional position so it
        # survives the resample without the caller needing to know the resampled length.
        n = len(pts)
        for i, p in enumerate(pts):
            frac = i / max(n - 1, 1)
            idx = target_idx_per_point[min(int(frac * len(target_idx_per_point)),
                                           len(target_idx_per_point) - 1)]
            self._emit(p, z, idx)

    def dwell(self, n: int, z: float, target_idx: int) -> None:
        """Hold position for `n` steps. Pen-down dwells are how TOUCH inks."""
        for _ in range(n):
            self._emit(self.pos, z, target_idx)

    def finish(self, manner: str, pen_start: np.ndarray) -> Demo:
        return Demo(actions=np.stack(self.actions).astype(np.float32),
                    pen_start=np.asarray(pen_start, dtype=np.float64).copy(),
                    manner=manner,
                    target_dot_index=np.asarray(self.targets, dtype=np.int64),
                    xy_path=np.stack(self.xy),
                    z_path=np.asarray(self.zs, dtype=np.float64))


def _segment_targets(layout: DotLayout, path_len_hint: int) -> List[int]:
    """Per-point dot index along the connect/curve route: point i belongs to the dot it approaches."""
    order = layout.order
    per_seg = max(1, path_len_hint // max(len(order) - 1, 1))
    out: List[int] = []
    for k in range(1, len(order)):
        out.extend([int(order[k])] * per_seg)
    return out or [int(order[-1])]


def _connect_like(layout: DotLayout, pen_start: np.ndarray, speed: float,
                  pen_down_everywhere: bool) -> _Emitter:
    """
    Shared body of CONNECT and TOUCH.

    Both walk exactly the same waypoints -- approach, then per-dot [traverse, dwell] -- so their xy
    streams are bitwise identical. The only difference is `pen_down_everywhere`, which decides
    whether z is held down through the traverses or only during the dwells.
    """
    e = _Emitter(pen_start, speed)
    dots = layout.ordered_dots()
    order = layout.order

    # approach the first dot with the pen up
    e.goto(dots[0], Z_UP, int(order[0]))
    # ink the first dot (TOUCH) or start the single stroke (CONNECT)
    e.dwell(DWELL_STEPS, Z_DOWN, int(order[0]))

    for k in range(1, len(dots)):
        traverse_z = Z_DOWN if pen_down_everywhere else Z_UP
        e.goto(dots[k], traverse_z, int(order[k]))
        e.dwell(DWELL_STEPS, Z_DOWN, int(order[k]))

    e.dwell(FINAL_LIFT_STEPS, Z_UP, int(order[-1]))
    return e


def make_connect(layout: DotLayout, pen_start: np.ndarray, speed: float = DEFAULT_SPEED) -> Demo:
    """One continuous stroke: straight lines from dot to dot, pen down throughout."""
    return _connect_like(layout, pen_start, speed, pen_down_everywhere=True).finish(
        'CONNECT', pen_start)


def make_touch(layout: DotLayout, pen_start: np.ndarray, speed: float = DEFAULT_SPEED) -> Demo:
    """Same route as CONNECT, but the pen only comes down on the dots themselves."""
    return _connect_like(layout, pen_start, speed, pen_down_everywhere=False).finish(
        'TOUCH', pen_start)


def make_curve(layout: DotLayout, pen_start: np.ndarray, speed: float = DEFAULT_SPEED) -> Demo:
    """One continuous stroke like CONNECT, but each segment bows into an arc."""
    e = _Emitter(pen_start, speed)
    order = layout.order
    dots = layout.ordered_dots()
    path = curve_path(layout)

    e.goto(dots[0], Z_UP, int(order[0]))
    e.dwell(DWELL_STEPS, Z_DOWN, int(order[0]))
    e.follow(path, Z_DOWN, _segment_targets(layout, len(path)))
    e.dwell(DWELL_STEPS, Z_DOWN, int(order[-1]))
    e.dwell(FINAL_LIFT_STEPS, Z_UP, int(order[-1]))
    return e.finish('CURVE', pen_start)


def make_parallel(layout: DotLayout, pen_start: np.ndarray, speed: float = DEFAULT_SPEED) -> Demo:
    """A short dash through each dot, all at the same angle, lifting between them."""
    e = _Emitter(pen_start, speed)
    order = layout.order
    for k, dash in enumerate(dash_segments(layout)):
        idx = int(order[k])
        e.goto(dash[0], Z_UP, idx)                       # approach the dash start, pen up
        e.follow(dash, Z_DOWN, [idx] * len(dash))        # draw the dash
        e.dwell(1, Z_UP, idx)                            # lift before moving on
    e.dwell(FINAL_LIFT_STEPS, Z_UP, int(order[-1]))
    return e.finish('PARALLEL', pen_start)


_BUILDERS = {
    'CONNECT': make_connect,
    'TOUCH': make_touch,
    'CURVE': make_curve,
    'PARALLEL': make_parallel,
}


def make_demo(manner: str, layout: DotLayout, pen_start: np.ndarray,
              speed: float = DEFAULT_SPEED,
              rng: Optional[np.random.Generator] = None,
              noise_std: float = 0.0) -> Demo:
    """
    Build one demonstration. `noise_std > 0` jitters the xy deltas so repeats of the same manner are
    not byte-identical; it is left at 0 for the separability assertions.
    """
    assert manner in _BUILDERS, f'unknown manner {manner!r}; expected one of {MANNERS}'
    demo = _BUILDERS[manner](layout, pen_start, speed)
    if noise_std > 0:
        assert rng is not None, 'noise_std > 0 requires an rng'
        # Jitter the absolute WAYPOINTS, then recompute the deltas -- do not jitter the deltas
        # directly. Because actions are relative, per-step delta noise integrates into a random walk
        # on position: over a ~60-step episode with std s the pen drifts by ~s*sqrt(60), which walks
        # it off the dots and silently costs coverage. Perturbing waypoints keeps the trajectory
        # anchored to the intended path, so noise stays local and bounded.
        jitter = rng.normal(0.0, noise_std, size=(len(demo.xy_path), 2))
        xy = np.clip(demo.xy_path + jitter, 0.0, 1.0)
        prev = np.concatenate([np.asarray(pen_start, dtype=np.float64)[None, :2], xy[:-1]], axis=0)
        demo.actions[:, :2] = (xy - prev).astype(np.float32)
        demo.xy_path = xy
    return demo


def make_all(layout: DotLayout, pen_start: np.ndarray,
             speed: float = DEFAULT_SPEED) -> Dict[str, Demo]:
    """All four manners for one instance, noiseless. Used by the separability check."""
    return {m: make_demo(m, layout, pen_start, speed) for m in MANNERS}


def path_to_demo(xy_path: np.ndarray, z_path: np.ndarray, pen_start: np.ndarray,
                 target_dot_index: Optional[np.ndarray] = None, manner: str = 'HUMAN',
                 speed: float = DEFAULT_SPEED, min_step_frac: float = 0.25) -> Demo:
    """
    Turn an arbitrary absolute `(xy, z)` path into a `Demo`, for paths this module did not author.

    This is the entry point for a **human** demonstration (see `scripts/draw_dot/collect_demo.py`).
    It runs through the same `_Emitter` the four scripted manners use, which is the point: a captured
    path comes out as an action stream structurally indistinguishable from a generated one -- same
    speed limit, same delta convention, same dwell semantics -- so nothing downstream needs to know
    which kind it is looking at.

    Two things happen on the way in.

    **Decimation.** A mouse is sampled tens of times a second and mostly does not move. A sample is
    dropped unless it either moved at least `min_step_frac * speed` or changed the pen state; without
    this, a few seconds of capture becomes thousands of near-zero actions. Contact transitions are
    never dropped, because the pen state is the entire content of two of the four manners.

    **Re-timing, not re-shaping.** `_Emitter.goto` walks toward each surviving sample in `speed`-sized
    hops, so hand speed is discarded and the geometry is kept exactly. That is the right trade here:
    all four scripted manners are generated at one constant `DEFAULT_SPEED`, so pacing carries no
    manner information in this dataset, while shape and contact carry all of it. A caller that wants
    the human's pacing should keep the raw path -- `collect_demo` saves it alongside.
    """
    xy = np.asarray(xy_path, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(z_path, dtype=np.float64).reshape(-1)
    assert len(xy) == len(z) and len(xy) > 0, 'xy_path and z_path must be non-empty and same length'

    min_step = float(min_step_frac) * float(speed)
    keep = [0]
    for i in range(1, len(xy)):
        moved = float(np.linalg.norm(xy[i] - xy[keep[-1]]))
        if z[i] != z[keep[-1]] or moved >= min_step:
            keep.append(i)
    xy, z = xy[keep], z[keep]

    if target_dot_index is None:
        targets = np.zeros(len(xy), dtype=np.int64)
    else:
        targets = np.asarray(target_dot_index, dtype=np.int64).reshape(-1)[keep]

    e = _Emitter(pen_start, speed)
    for p, zi, idx in zip(xy, z, targets):
        e.goto(p, float(zi), int(idx))
    return e.finish(manner, pen_start)
