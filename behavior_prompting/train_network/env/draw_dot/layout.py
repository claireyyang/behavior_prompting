"""
Geometry for `DrawingDotEnv`: dot layouts, the paths each manner traces, and rasterization.

This module owns *pure geometry* and nothing else. `scripts/draw_dot/strategies.py` turns these
paths into timed action sequences, and `utils/dot_strategy_metrics.py` rasterizes them into ink
templates for scoring. Both read the same builders here, so a rollout is always scored against the
same curve the demonstrator was asked to trace -- there is no second, subtly-different copy of the
geometry to drift out of sync.

Frames
------
Everything is in the **scene frame**, the unit square [0, 1]^2, with y pointing DOWN so that scene
coordinates map to image coordinates by a single multiply. Canvas resolution is therefore a pure
rendering choice: nothing about the task, the actions, or the metrics changes if it moves.

Why the manners are distinct *by construction*
----------------------------------------------
`CURVE` is not a spline interpolating the dots. A spline's deviation from the straight polyline is a
function of how the dots happen to be arranged, and it collapses toward `CONNECT` as they approach
collinear -- which would make manner distinctness contingent on layout, and force us to reject
layouts to protect it. Rejecting layouts would bias the very placement distribution that provides
across-instance generalization.

Instead each segment is replaced by an arc bowed perpendicular to it by
`max(CURVE_BOW * segment_length, MIN_CURVE_BOW)`. That guarantees a minimum deviation for *any*
layout, collinear included, so dots can be placed uniformly at random and every manner stays valid
everywhere.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# -- scene ------------------------------------------------------------------------------

SCENE_MIN = 0.0
SCENE_MAX = 1.0

# The four valid manners. Defined here, alongside the geometry that distinguishes them, so that
# `utils/dot_strategy_metrics.py` can classify a rollout without importing from `scripts/`.
MANNERS = ('CONNECT', 'TOUCH', 'CURVE', 'PARALLEL')
MANNER_TO_ID = {m: i for i, m in enumerate(MANNERS)}

# Manners drawn as a single continuous stroke, vs. those that lift between dots. This is the
# `contact axis` -- the sub-axis reported separately when diagnosing which mode structure collapsed.
CONTINUOUS_MANNERS = ('CONNECT', 'CURVE')
LIFTING_MANNERS = ('TOUCH', 'PARALLEL')

N_DOTS = 4
DOT_RADIUS = 0.04          # scene units; ~4 px at 96, so a dot is several pixels across
DOT_MIN_SEP = 0.15         # far enough apart to stay resolvable as distinct blobs at 96 px
DOT_MARGIN = 0.12          # keep dots off the scene border so every manner's ink stays in frame

# CURVE: perpendicular bow as a fraction of segment length, with an absolute floor so that short
# segments still bow visibly rather than degenerating toward a straight line, and a ceiling so that
# long segments cannot bow the arc out of the scene.
#
# ⚠️ MAX_CURVE_BOW < DOT_MARGIN is load-bearing, not cosmetic. A quadratic Bezier's perpendicular
# deviation from its chord peaks at exactly `bow`, so the whole arc lies inside a capsule of radius
# `bow` around the chord. With both endpoints inside the margin box, capping the bow below the
# margin proves the arc never leaves [0, 1]^2. Without it, the env clips the pen at the boundary
# (sliding it along the edge) while the template rasterizer clips the drawn *line* instead -- the
# two disagree, and a perfect CURVE demo scores ink_precision < 1.
CURVE_BOW = 0.30
MIN_CURVE_BOW = 0.06
MAX_CURVE_BOW = 0.10

# PARALLEL: dash length must exceed a dot diameter (2 * DOT_RADIUS = 0.08) or the dashes would be
# indistinguishable from TOUCH's blobs. This is what makes that pair non-collapsible for any layout.
DASH_LENGTH = 0.20
DASH_ANGLE = np.pi / 2     # vertical dashes; fixed so PARALLEL has one canonical form

PEN_RADIUS_PX = 2          # -> a ~4 px stroke at 96
PATH_DS = 0.004            # arc-length resampling step for dense paths (~0.4 px at 96)

# -- colours for the human render -------------------------------------------------------

BACKGROUND_COLOR = (255, 255, 255)
INK_COLOR = (0, 0, 255)
DOT_COLOR = (40, 40, 40)
WALL_COLOR = (200, 60, 60)
PEN_DOWN_COLOR = (48, 156, 54)
PEN_UP_COLOR = (150, 200, 150)


@dataclass
class DotLayout:
    """One instance of the task: where the dots are, and (Phase 2) where the walls are."""
    dots: np.ndarray                       # (N_DOTS, 2) scene frame
    walls: Optional[np.ndarray] = None     # (M, 2, 2) segments; None in v1

    @property
    def n_dots(self) -> int:
        return len(self.dots)

    @property
    def order(self) -> np.ndarray:
        """
        Visit order, fixed left-to-right for every manner.

        Order is deliberately NOT one of the manner axes: randomizing it would inject uncontrolled
        multimodality on top of the manner variation and blur the mode structure being measured.
        """
        return np.argsort(self.dots[:, 0], kind='stable')

    def ordered_dots(self) -> np.ndarray:
        return self.dots[self.order]


# -- sampling ---------------------------------------------------------------------------

def sample_layout(rng: np.random.Generator,
                  n_dots: int = N_DOTS,
                  min_sep: float = DOT_MIN_SEP,
                  margin: float = DOT_MARGIN,
                  max_tries: int = 10_000) -> DotLayout:
    """
    Place `n_dots` uniformly at random, subject only to a minimum separation.

    The separation is a RENDERING requirement -- two dots closer than this stop being resolvable as
    distinct blobs at 96 px and their ink templates start to overlap. It is not a
    strategy-distinctness filter: no layout is ever rejected for its geometry (collinear included),
    because every manner is valid everywhere by construction.
    """
    lo, hi = SCENE_MIN + margin, SCENE_MAX - margin
    dots: List[np.ndarray] = []
    tries = 0
    while len(dots) < n_dots:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                f'could not place {n_dots} dots with min_sep={min_sep} inside [{lo}, {hi}]^2; '
                f'lower min_sep or margin')
        p = rng.uniform(lo, hi, size=2)
        if all(np.linalg.norm(p - q) >= min_sep for q in dots):
            dots.append(p)
    return DotLayout(dots=np.stack(dots, axis=0))


def sample_pen_start(rng: np.random.Generator, margin: float = DOT_MARGIN) -> np.ndarray:
    """Initial `[x, y, z]`. z starts up, so an episode never begins mid-stroke."""
    lo, hi = SCENE_MIN + margin, SCENE_MAX - margin
    return np.array([*rng.uniform(lo, hi, size=2), 1.0], dtype=np.float64)


# -- path builders ----------------------------------------------------------------------

def resample_by_arclength(pts: np.ndarray, ds: float = PATH_DS) -> np.ndarray:
    """Resample a polyline to roughly uniform `ds` spacing. Endpoints are pinned exactly."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-12:
        return pts[:1].copy()
    n = max(2, int(np.ceil(total / ds)) + 1)
    want = np.linspace(0.0, total, n)
    out = np.stack([np.interp(want, cum, pts[:, 0]), np.interp(want, cum, pts[:, 1])], axis=1)
    out[0], out[-1] = pts[0], pts[-1]
    return out


def _bowed_arc(a: np.ndarray, b: np.ndarray, bow_frac: float = CURVE_BOW,
               min_bow: float = MIN_CURVE_BOW, max_bow: float = MAX_CURVE_BOW,
               n: int = 64) -> np.ndarray:
    """
    A quadratic Bezier from `a` to `b`, pushed perpendicular to `ab` at its midpoint.

    The deviation is `clip(bow_frac * |ab|, min_bow, max_bow)`. The floor is what lets CURVE stay
    distinct from CONNECT no matter how the dots are arranged; the ceiling is what keeps the arc
    inside the scene (see the MAX_CURVE_BOW note above).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    length = float(np.linalg.norm(d))
    if length <= 1e-12:
        return np.stack([a, b], axis=0)
    # left-hand normal, in a y-down frame
    normal = np.array([-d[1], d[0]]) / length
    bow = float(np.clip(bow_frac * length, min_bow, max_bow))
    # A quadratic Bezier reaches half its control-point offset at t=0.5, so double it to make the
    # midpoint deviation equal `bow`.
    control = (a + b) / 2.0 + normal * (2.0 * bow)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * control + t ** 2 * b


def connect_path(layout: DotLayout) -> np.ndarray:
    """CONNECT / TOUCH route: straight lines between consecutive dots, densely resampled."""
    return resample_by_arclength(layout.ordered_dots())


def curve_path(layout: DotLayout, bow_frac: float = CURVE_BOW,
               min_bow: float = MIN_CURVE_BOW, max_bow: float = MAX_CURVE_BOW) -> np.ndarray:
    """CURVE route: same dots in the same order, each segment replaced by a bowed arc."""
    assert max_bow < DOT_MARGIN, (
        f'max_bow ({max_bow}) must stay under DOT_MARGIN ({DOT_MARGIN}) so the arc provably stays '
        f'inside the scene; otherwise env pen-clipping and template rasterization disagree.')
    pts = layout.ordered_dots()
    pieces = [_bowed_arc(pts[i], pts[i + 1], bow_frac, min_bow, max_bow)
              for i in range(len(pts) - 1)]
    # drop each piece's duplicated first point except the very first
    chained = np.concatenate([pieces[0]] + [p[1:] for p in pieces[1:]], axis=0)
    return resample_by_arclength(chained)


def dash_segments(layout: DotLayout, length: float = DASH_LENGTH,
                  angle: float = DASH_ANGLE) -> List[np.ndarray]:
    """PARALLEL: one dash centred on each dot, all at the same angle."""
    direction = np.array([np.cos(angle), np.sin(angle)])
    half = direction * (length / 2.0)
    return [resample_by_arclength(np.stack([d - half, d + half], axis=0))
            for d in layout.ordered_dots()]


def dot_waypoints(layout: DotLayout) -> np.ndarray:
    """The dots themselves, in visit order. TOUCH inks only here."""
    return layout.ordered_dots()


# -- rasterization ----------------------------------------------------------------------

def to_px(pts: np.ndarray, size: int) -> np.ndarray:
    """Scene [0,1]^2 -> integer pixel coordinates. y is already down in both frames."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return np.rint(pts * (size - 1)).astype(np.int32)


def draw_polyline(mask: np.ndarray, pts: np.ndarray, size: int,
                  pen_radius_px: int = PEN_RADIUS_PX, value: int = 1) -> None:
    """Paint a polyline into a uint8 mask in place, with round caps so joins are not notched."""
    px = to_px(pts, size)
    thickness = max(1, 2 * int(pen_radius_px))
    if len(px) == 1:
        cv2.circle(mask, tuple(px[0]), int(pen_radius_px), value, -1, lineType=cv2.LINE_8)
        return
    for i in range(len(px) - 1):
        cv2.line(mask, tuple(px[i]), tuple(px[i + 1]), value, thickness, lineType=cv2.LINE_8)
    for p in (px[0], px[-1]):
        cv2.circle(mask, tuple(p), int(pen_radius_px), value, -1, lineType=cv2.LINE_8)


def rasterize_paths(paths: Sequence[np.ndarray], size: int,
                    pen_radius_px: int = PEN_RADIUS_PX) -> np.ndarray:
    """Union of several polylines as a binary uint8 mask."""
    mask = np.zeros((size, size), dtype=np.uint8)
    for p in paths:
        if len(p):
            draw_polyline(mask, p, size, pen_radius_px, 1)
    return mask


def dot_mask(layout: DotLayout, size: int, radius: float = DOT_RADIUS) -> np.ndarray:
    """Filled discs at each dot -- observation channel 1, and the TOUCH ink template."""
    mask = np.zeros((size, size), dtype=np.uint8)
    r = max(1, int(round(radius * (size - 1))))
    for p in to_px(layout.dots, size):
        cv2.circle(mask, tuple(p), r, 1, -1, lineType=cv2.LINE_8)
    return mask


def wall_mask(layout: DotLayout, size: int) -> np.ndarray:
    """Observation channel 2. All-zero in v1; Phase 2 fills it from `layout.walls`."""
    mask = np.zeros((size, size), dtype=np.uint8)
    if layout.walls is None:
        return mask
    for seg in layout.walls:
        px = to_px(seg, size)
        cv2.line(mask, tuple(px[0]), tuple(px[1]), 1, 2, lineType=cv2.LINE_8)
    return mask


def compose_canvas(ink: np.ndarray, layout: DotLayout, size: int) -> np.ndarray:
    """
    The `(3, size, size)` float32 observation: ch0 ink, ch1 dots, ch2 walls.

    Three meaningful signals in the three channels a pretrained RGB encoder expects, rather than one
    replicated three times -- and Phase 2's walls slot into ch2 with no shape change.
    """
    return np.stack([ink.astype(np.float32),
                     dot_mask(layout, size).astype(np.float32),
                     wall_mask(layout, size).astype(np.float32)], axis=0)


# -- ink templates, and the allowed-ink region ------------------------------------------

def ink_templates(layout: DotLayout, size: int,
                  pen_radius_px: int = PEN_RADIUS_PX) -> dict:
    """
    The ink each manner leaves behind, as binary masks. Used to classify a rollout's manner and to
    build the allowed-ink region.
    """
    # TOUCH's template is pen-radius blobs at the dots, NOT the DOT_RADIUS discs of `dot_mask`:
    # TOUCH inks by dwelling motionless at each dot, so what it actually draws is a disc the width
    # of the pen. `dot_mask` is the observation channel, which is a different thing.
    touch_blobs = [layout.ordered_dots()[i:i + 1] for i in range(layout.n_dots)]
    return {
        'CONNECT': rasterize_paths([connect_path(layout)], size, pen_radius_px),
        'TOUCH': rasterize_paths(touch_blobs, size, pen_radius_px),
        'CURVE': rasterize_paths([curve_path(layout)], size, pen_radius_px),
        'PARALLEL': rasterize_paths(dash_segments(layout), size, pen_radius_px),
    }


def allowed_ink_mask(layout: DotLayout, size: int, tol_px: int = 3,
                     pen_radius_px: int = PEN_RADIUS_PX) -> np.ndarray:
    """
    Where ink is legitimate: the union of **all four manners' templates**, dilated by `tol_px`.

    It cannot be "dot discs plus straight pairwise segments" -- PARALLEL's dashes and CURVE's bowed
    arcs lie off those segments, so two perfectly valid manners would be scored as stray ink. Under
    the union definition every valid manner scores 1.0 and genuine scribbling still scores low.
    """
    union = np.zeros((size, size), dtype=np.uint8)
    for m in ink_templates(layout, size, pen_radius_px).values():
        union |= m
    if tol_px > 0:
        k = 2 * int(tol_px) + 1
        union = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return union


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return float((a & b).sum() / union)
