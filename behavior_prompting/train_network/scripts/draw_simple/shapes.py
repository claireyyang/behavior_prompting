"""
Procedural shape sampling for the simplified draw environment.

Self-contained on purpose: it shares no code with `scripts/draw/procedural_generate_drawings.py`.
That generator's shape sampler is entangled with action generation (speeds, per-part sample
counts, delay steps, noise), because in the old env geometry and timing genuinely interact. Here
they do not, so this module produces *only* geometry and knows nothing about speed.

A task is a list of STROKES. A stroke is one continuous pen-down run, built as a chain of
primitives (line / arc / cubic Bezier) that share endpoints, then discretised once at fixed
arc-length spacing `ds` by `geometry.resample_by_arclength`. After that point there are no
primitives, only vertices -- which is what lets the env treat ink as an edge set.
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from behavior_prompting.train_network.env.draw_simple.geometry import (
    BOARD_CENTER,
    StrokeGeometry,
    chain_polylines,
    resample_by_arclength,
)

PRIMITIVES = ('line', 'arc', 'bezier')
_DENSE = 2048  # samples used before arc-length resampling; only affects resample accuracy


# ---------------------------------------------------------------------------------------
# primitives -> dense point sets
# ---------------------------------------------------------------------------------------

def line_points(p0, p1) -> np.ndarray:
    return np.stack([np.asarray(p0, float), np.asarray(p1, float)], axis=0)


def bezier_points(p0, c1, c2, p3, n: int = _DENSE) -> np.ndarray:
    p0, c1, c2, p3 = (np.asarray(v, float) for v in (p0, c1, c2, p3))
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1
            + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p3)


def arc_points(center, rx: float, ry: float, rot: float,
               a0: float, a1: float, n: int = _DENSE) -> np.ndarray:
    """Points on an axis-rotated ellipse, angle sweeping a0 -> a1 (signed)."""
    center = np.asarray(center, float)
    a = np.linspace(a0, a1, n)
    x, y = rx * np.cos(a), ry * np.sin(a)
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([center[0] + x * c - y * s, center[1] + x * s + y * c], axis=1)


def arc_from_start(start, rx: float, ry: float, rot: float,
                   a0: float, a1: float, n: int = _DENSE) -> np.ndarray:
    """Same as `arc_points`, but with the centre chosen so the arc begins at `start`."""
    start = np.asarray(start, float)
    c, s = math.cos(rot), math.sin(rot)
    x0, y0 = rx * math.cos(a0), ry * math.sin(a0)
    offset = np.array([x0 * c - y0 * s, x0 * s + y0 * c])
    return arc_points(start - offset, rx, ry, rot, a0, a1, n)


# ---------------------------------------------------------------------------------------
# sampling helpers
# ---------------------------------------------------------------------------------------

class BoardBox:
    """The axis-aligned region, in the upright board frame, that geometry must stay inside."""

    def __init__(self, board_length: float, margin: float):
        half = board_length / 2 - margin
        self.lo = BOARD_CENTER - half
        self.hi = BOARD_CENTER + half

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lo, self.hi, size=2)

    def sample_far_from(self, rng: np.random.Generator, p: np.ndarray,
                        min_dist: float, max_dist: Optional[float] = None,
                        attempts: int = 200) -> Optional[np.ndarray]:
        for _ in range(attempts):
            q = self.sample(rng)
            d = float(np.linalg.norm(q - p))
            if d >= min_dist and (max_dist is None or d <= max_dist):
                return q
        return None

    def contains(self, pts: np.ndarray) -> bool:
        pts = np.asarray(pts, float).reshape(-1, 2)
        return bool(np.all(pts >= self.lo) and np.all(pts <= self.hi))


def _sample_primitive(rng: np.random.Generator, kind: str, start: np.ndarray, box: BoardBox,
                      min_len: float, max_len: float,
                      arc_radius_range: Tuple[float, float],
                      attempts: int = 60) -> Optional[np.ndarray]:
    """
    Sample one primitive starting at `start` and lying entirely inside `box`.

    Returns dense points (start included) or None if no in-box sample was found. Callers fall
    back to a shorter primitive rather than shrinking the box, so shapes stay well spread.
    """
    for _ in range(attempts):
        if kind == 'line':
            end = box.sample_far_from(rng, start, min_len, max_len)
            if end is None:
                continue
            pts = line_points(start, end)

        elif kind == 'bezier':
            end = box.sample_far_from(rng, start, min_len, max_len)
            if end is None:
                continue
            d = end - start
            dist = float(np.linalg.norm(d))
            perp = np.array([-d[1], d[0]]) / dist
            bow = min(0.35 * dist, 90.0)
            c1 = start + 0.33 * d + perp * rng.uniform(-bow, bow)
            c2 = start + 0.67 * d + perp * rng.uniform(-bow, bow)
            pts = bezier_points(start, c1, c2, end)

        elif kind == 'arc':
            rlo, rhi = arc_radius_range
            rx = float(rng.uniform(rlo, rhi))
            ry = rx * float(rng.uniform(0.35, 2.2))
            rot = float(rng.uniform(0.0, 2 * np.pi))
            a0 = float(rng.uniform(0.0, 2 * np.pi))
            span = float(rng.uniform(np.pi / 3, 2 * np.pi)) * (1.0 if rng.random() < 0.5 else -1.0)
            pts = arc_from_start(start, rx, ry, rot, a0, a0 + span)

        else:
            raise ValueError(f'unknown primitive {kind!r}')

        if not box.contains(pts):
            continue
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if length < min_len:
            continue
        return pts

    return None


# ---------------------------------------------------------------------------------------
# stroke / task sampling
# ---------------------------------------------------------------------------------------

def sample_stroke(rng: np.random.Generator, box: BoardBox, ds: float,
                  n_parts: int, min_part_len: float, max_part_len: float,
                  arc_radius_range: Tuple[float, float],
                  allowed: Sequence[str]) -> np.ndarray:
    """
    Sample one stroke as a chain of `n_parts` primitives, returned as a resampled polyline.

    Each primitive starts where the previous one ended, so the stroke is a single continuous
    pen-down run -- the unit the strategy layer permutes and reverses.
    """
    start = box.sample(rng)
    parts: List[np.ndarray] = []
    cursor = start
    for _ in range(n_parts):
        kind = str(rng.choice(list(allowed)))
        pts = _sample_primitive(rng, kind, cursor, box, min_part_len, max_part_len,
                                arc_radius_range)
        if pts is None and kind != 'line':
            pts = _sample_primitive(rng, 'line', cursor, box, min_part_len, max_part_len,
                                    arc_radius_range)
        if pts is None:
            break
        parts.append(resample_by_arclength(pts, ds))
        cursor = parts[-1][-1]

    if not parts:
        raise RuntimeError('failed to sample any primitive for a stroke; '
                           'check --margin / --min-part-len against --board-length')
    return chain_polylines(parts)


def sample_task_geometry(rng: np.random.Generator,
                         n_strokes: int,
                         parts_per_stroke: Tuple[int, int],
                         board_length: float = 350.0,
                         margin: float = 30.0,
                         ds: float = 1.0,
                         min_part_len: float = 60.0,
                         max_part_len: float = 220.0,
                         arc_radius_range: Tuple[float, float] = (30.0, 80.0),
                         allowed: Sequence[str] = PRIMITIVES,
                         min_stroke_separation: float = 0.0,
                         attempts: int = 40) -> StrokeGeometry:
    """
    Sample a whole task's geometry: `n_strokes` independent strokes inside the board.

    `min_stroke_separation` > 0 rejects strokes that come closer than that to an already-placed
    one. It defaults to 0 -- crossing strokes are allowed and make for more interesting shapes --
    because the env resolves crossings by stroke continuity (`SimpleDrawEnv._ink`) rather than by
    re-deciding attribution each step. Raising it also rules out the one remaining ambiguous
    case, where two strokes pass within `snap_tol` of the point at which the pen goes down.
    """
    box = BoardBox(board_length, margin)
    lo, hi = parts_per_stroke

    # Accept strokes ONE AT A TIME, retrying only the stroke that clashes. Rejecting the whole
    # task on any clash fails exponentially in n_strokes -- 4 strokes at 20px separation on a
    # 350px board was already unsatisfiable within 40 whole-task attempts.
    accepted: List[np.ndarray] = []
    for stroke_i in range(n_strokes):
        for _ in range(attempts * 8):
            cand = sample_stroke(rng, box, ds, int(rng.integers(lo, hi + 1)),
                                 min_part_len, max_part_len, arc_radius_range, allowed)
            if min_stroke_separation <= 0 or all(
                    _separated(cand, other, min_stroke_separation, ds) for other in accepted):
                accepted.append(cand)
                break
        else:
            raise RuntimeError(
                f'could not place stroke {stroke_i + 1}/{n_strokes} at least '
                f'{min_stroke_separation}px from the {len(accepted)} already placed. Lower '
                f'--min-stroke-separation / --n-strokes / --min-part-len, or raise '
                f'--board-length.')

    return StrokeGeometry(accepted, ds=ds)


def _separated(a: np.ndarray, b: np.ndarray, min_sep: float, ds: float, stride: int = 8) -> bool:
    """
    True if two strokes stay at least `min_sep` apart.

    Cheap conservative test first: vertices sit ~`ds` apart, so a strided subsample's minimum
    distance underestimates the true minimum by at most `stride * ds`. Clearing
    `min_sep + stride * ds` on the subsample therefore proves separation without the full
    pairwise distance matrix; only borderline pairs pay for the exact check.
    """
    coarse = np.linalg.norm(a[::stride, None, :] - b[None, ::stride, :], axis=2).min()
    if coarse >= min_sep + stride * ds:
        return True
    return bool(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min() >= min_sep)
