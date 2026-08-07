"""
Canonical polyline geometry for the simplified draw environment.

This module exists to establish one invariant, which everything downstream leans on:

    the ink laid down by an execution is a function of WHICH CANONICAL EDGES were traversed,
    and of nothing else.

A task's geometry is compiled ONCE into a fixed vertex array plus a partition of it into
strokes. Curves are discretised here, at task-definition time, at a fixed arc-length spacing
`ds` -- so the polyline *is* the ground truth rather than an approximation of a Bezier that a
faster execution would happen to approximate more coarsely. Ink is then recorded as a boolean
mask over edges, and the image is a pure function of that mask. Because a mask is a set:

  * stroke order cannot matter -- set union is commutative;
  * traversal direction cannot matter -- `edges_in` sorts its arc-length interval;
  * the velocity profile cannot matter -- for any monotone schedule 0 = s_0 < ... < s_T = L,
    the union of the per-step intervals [s_{t-1}, s_t] is exactly [0, L].

That is why this env can hit *bit-identical* goal images across permuted strategies, where a
PD-tracked env that rasterises the achieved trajectory can only approach it. There is no
velocity state anywhere in this stack, so nothing carries across a stroke boundary either.

Frames
------
Two frames, and mixing them up is the one real footgun here:

  board frame   upright, y DOWN, same units as the image. All geometry, all arc lengths, and
                the inked edge set live here -- so ink is independent of the board rotation.
  canvas frame  the board frame rotated by `boundary_angle` about the canvas centre. Actions,
                observations and rendered pixels live here.

`board_to_canvas` / `canvas_to_board` are the only conversions. Note y is DOWN in both, i.e.
image convention throughout -- unlike `env/draw/draw_env.py`, which inherits pymunk's y-up
convention and flips at render time.
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

WINDOW_SIZE = 512
BOARD_CENTER = WINDOW_SIZE / 2.0

INK_COLOR = (0, 0, 255)          # pure blue, matching draw_env's pen
BACKGROUND_COLOR = (255, 255, 255)
GOAL_COLOR = (255, 170, 170)     # faint red, drawn under the ink as a reference overlay


# ---------------------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------------------

def board_to_canvas(pts: np.ndarray, angle: float) -> np.ndarray:
    """Rotate board-frame points by `angle` about the canvas centre."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if angle == 0.0:
        return pts.copy()
    rel = pts - BOARD_CENTER
    c, s = np.cos(angle), np.sin(angle)
    out = np.empty_like(rel)
    out[:, 0] = rel[:, 0] * c - rel[:, 1] * s
    out[:, 1] = rel[:, 0] * s + rel[:, 1] * c
    return out + BOARD_CENTER


def canvas_to_board(pts: np.ndarray, angle: float) -> np.ndarray:
    """Inverse of `board_to_canvas`."""
    return board_to_canvas(pts, -angle)


# ---------------------------------------------------------------------------------------
# arc-length resampling -- the single place discretisation happens
# ---------------------------------------------------------------------------------------

def resample_by_arclength(dense: np.ndarray, ds: float) -> np.ndarray:
    """
    Resample a densely-sampled curve to near-uniform arc-length spacing `ds`.

    Every primitive (line, arc, Bezier) goes through here, so all of them end up with the
    same vertex density and no primitive-specific parametrisation leaks into the geometry.
    Returns a (n, 2) array including both endpoints exactly, with n >= 2 and no repeated
    vertices (zero-length segments would make arc-length projection ill-defined).
    """
    dense = np.asarray(dense, dtype=np.float64).reshape(-1, 2)
    if len(dense) < 2:
        raise ValueError('need at least 2 points to resample')

    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-12])
    dense = dense[keep]
    if len(dense) < 2:
        raise ValueError('curve collapsed to a single point')

    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-9:
        raise ValueError('curve has zero length')

    n_seg = max(1, int(np.ceil(total / float(ds))))
    s = np.linspace(0.0, total, n_seg + 1)
    out = np.stack([np.interp(s, cum, dense[:, 0]),
                    np.interp(s, cum, dense[:, 1])], axis=1)
    # pin the endpoints so joins between primitives are exact, not interp-approximate
    out[0] = dense[0]
    out[-1] = dense[-1]
    return out


def chain_polylines(polylines: Sequence[np.ndarray]) -> np.ndarray:
    """
    Concatenate polylines that share endpoints into one vertex array, dropping the duplicated
    joint vertex. Used to turn a chain of primitives into a single stroke.
    """
    parts = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polylines]
    out = [parts[0]]
    for prev, cur in zip(parts[:-1], parts[1:]):
        if np.linalg.norm(cur[0] - prev[-1]) > 1e-6:
            raise ValueError(f'polylines do not chain: {prev[-1]} -> {cur[0]}')
        out.append(cur[1:])
    verts = np.concatenate(out, axis=0)
    seg = np.linalg.norm(np.diff(verts, axis=0), axis=1)
    return verts[np.concatenate([[True], seg > 1e-9])]


# ---------------------------------------------------------------------------------------
# the geometry object
# ---------------------------------------------------------------------------------------

class StrokeGeometry:
    """
    A task's fixed geometry: a list of strokes, each a polyline in the board frame.

    A *stroke* is one continuous pen-down run and is the unit of ordering and reversal.
    Edges are numbered globally so an inked set is a single boolean array of length
    `n_edges`, which makes union / intersection / IoU one-liners.
    """

    def __init__(self, strokes: Sequence[np.ndarray], ds: float = 1.0):
        self.ds = float(ds)
        self._strokes: List[np.ndarray] = []
        self._cums: List[np.ndarray] = []
        self._v_offset: List[int] = []
        self._e_offset: List[int] = []

        verts, pairs, owner = [], [], []
        v_off = e_off = 0
        for k, raw in enumerate(strokes):
            sv = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
            if len(sv) < 2:
                raise ValueError(f'stroke {k} has fewer than 2 vertices')
            seg = np.linalg.norm(np.diff(sv, axis=0), axis=1)
            if not np.all(seg > 1e-9):
                raise ValueError(f'stroke {k} contains a zero-length segment')

            n = len(sv)
            self._strokes.append(sv)
            self._cums.append(np.concatenate([[0.0], np.cumsum(seg)]))
            self._v_offset.append(v_off)
            self._e_offset.append(e_off)
            verts.append(sv)
            pairs.append(np.stack([np.arange(n - 1), np.arange(1, n)], axis=1) + v_off)
            owner.append(np.full(n - 1, k, dtype=np.int64))
            v_off += n
            e_off += n - 1

        if not self._strokes:
            raise ValueError('geometry needs at least one stroke')

        self.vertices = np.concatenate(verts, axis=0)
        self.edge_pairs = np.concatenate(pairs, axis=0)
        self.edge_stroke = np.concatenate(owner, axis=0)
        self._canvas_cache: dict = {}

    # -- basic properties ---------------------------------------------------------------

    @property
    def n_strokes(self) -> int:
        return len(self._strokes)

    @property
    def n_edges(self) -> int:
        return len(self.edge_pairs)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    def stroke_vertices(self, k: int) -> np.ndarray:
        return self._strokes[k]

    def stroke_length(self, k: int) -> float:
        return float(self._cums[k][-1])

    def total_length(self) -> float:
        return float(sum(self.stroke_length(k) for k in range(self.n_strokes)))

    def all_edges_mask(self) -> np.ndarray:
        return np.ones(self.n_edges, dtype=bool)

    def empty_mask(self) -> np.ndarray:
        return np.zeros(self.n_edges, dtype=bool)

    def stroke_edge_range(self, k: int) -> Tuple[int, int]:
        start = self._e_offset[k]
        return start, start + len(self._strokes[k]) - 1

    # -- arc-length queries -------------------------------------------------------------

    def point_at(self, k: int, s: float) -> np.ndarray:
        """Board-frame point at arc length `s` along stroke `k` (clamped to the stroke)."""
        sv, cum = self._strokes[k], self._cums[k]
        s = float(np.clip(s, 0.0, cum[-1]))
        i = int(np.clip(np.searchsorted(cum, s, side='right') - 1, 0, len(sv) - 2))
        span = cum[i + 1] - cum[i]
        t = 0.0 if span <= 0 else (s - cum[i]) / span
        return sv[i] + t * (sv[i + 1] - sv[i])

    def points_at(self, k: int, s: np.ndarray) -> np.ndarray:
        return np.stack([self.point_at(k, float(v)) for v in np.atleast_1d(s)], axis=0)

    def project(self, p: np.ndarray, k: int) -> Tuple[float, float]:
        """Project a board-frame point onto stroke `k`. Returns (arc length, distance)."""
        sv, cum = self._strokes[k], self._cums[k]
        p = np.asarray(p, dtype=np.float64).reshape(2)
        a, b = sv[:-1], sv[1:]
        ab = b - a
        l2 = np.einsum('ij,ij->i', ab, ab)
        t = np.clip(np.einsum('ij,ij->i', p - a, ab) / l2, 0.0, 1.0)
        d = np.linalg.norm(a + t[:, None] * ab - p, axis=1)
        i = int(np.argmin(d))
        return float(cum[i] + t[i] * np.sqrt(l2[i])), float(d[i])

    def project_pair(self, p_prev: np.ndarray, p_cur: np.ndarray) -> Tuple[int, float, float, float]:
        """
        Assign a swept segment to a single stroke, and return (k, s_prev, s_cur, worst_dist).

        Both endpoints are scored against the same stroke and the stroke minimising the WORSE
        of the two distances wins. Scoring the pair jointly rather than each endpoint alone
        already resolves most crossings, but not all of them -- so the env calls this only to
        ACQUIRE a stroke at pen-down and then follows that stroke for the rest of the pen-down
        run. See `SimpleDrawEnv._ink`; re-deciding per step measurably loses coverage on tasks
        whose strokes cross.
        """
        best = None
        for k in range(self.n_strokes):
            s0, d0 = self.project(p_prev, k)
            s1, d1 = self.project(p_cur, k)
            score = max(d0, d1)
            if best is None or score < best[3]:
                best = (k, s0, s1, score)
        return best  # type: ignore[return-value]

    def edges_in(self, k: int, s_lo: float, s_hi: float) -> np.ndarray:
        """
        Global indices of the edges of stroke `k` overlapped by the arc-length interval
        between `s_lo` and `s_hi`, in either order.

        Sorting the interval is what makes traversal direction irrelevant. A zero-length
        interval (a pen-down dwell) inks the single edge it sits on -- a subset of what the
        surrounding traversal inks anyway, so it cannot break invariance, and it matches what
        a real pen resting on paper does.
        """
        cum = self._cums[k]
        lo, hi = (s_lo, s_hi) if s_lo <= s_hi else (s_hi, s_lo)
        local = np.nonzero((cum[1:] > lo) & (cum[:-1] < hi))[0]
        if local.size == 0:
            j = int(np.clip(np.searchsorted(cum, lo, side='right') - 1, 0, len(cum) - 2))
            local = np.array([j], dtype=np.int64)
        return local + self._e_offset[k]

    # -- rasterisation ------------------------------------------------------------------

    def canvas_points(self, angle: float) -> List[Tuple[int, int]]:
        """Integer canvas-frame pixel coordinates of every vertex, cached per angle."""
        key = round(float(angle), 12)
        cached = self._canvas_cache.get(key)
        if cached is None:
            pts = board_to_canvas(self.vertices, key)
            cached = [(int(x), int(y)) for x, y in np.rint(pts).astype(np.int64)]
            self._canvas_cache[key] = cached
        return cached

    def paint_edges(self, img: np.ndarray, edges, angle: float, pen_radius: int, color) -> None:
        """
        Paint the given edges into `img` (modified in place).

        Each edge is a capsule: a thick line plus a disk at each endpoint. Painting is
        idempotent and writes a pixel set determined solely by (v0, v1, pen_radius), so
        painting a set of edges in any order -- or incrementally, a few at a time, as the env
        does -- produces exactly the same image as painting them all at once. That equality is
        why the env can render incrementally without giving up bit-exactness.
        """
        pts = self.canvas_points(angle)
        thickness = max(1, int(round(2 * pen_radius)))
        r = int(pen_radius)
        for e in np.atleast_1d(np.asarray(edges)).ravel():
            i0, i1 = self.edge_pairs[int(e)]
            p0, p1 = pts[int(i0)], pts[int(i1)]
            cv2.line(img, p0, p1, color, thickness=thickness, lineType=cv2.LINE_8)
            cv2.circle(img, p0, r, color, -1, lineType=cv2.LINE_8)
            cv2.circle(img, p1, r, color, -1, lineType=cv2.LINE_8)

    def rasterize(self, edge_mask: np.ndarray, angle: float = 0.0, pen_radius: int = 6,
                  ink_color=INK_COLOR, background=BACKGROUND_COLOR,
                  size: Optional[int] = None) -> np.ndarray:
        """Render an inked edge set as a (size, size, 3) uint8 RGB image."""
        img = np.empty((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
        img[:] = np.asarray(background, dtype=np.uint8)
        edges = np.nonzero(np.asarray(edge_mask, dtype=bool))[0]
        if edges.size:
            self.paint_edges(img, edges, angle, pen_radius, tuple(int(c) for c in ink_color))
        if size is not None and size != WINDOW_SIZE:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return img

    # -- (de)serialisation --------------------------------------------------------------

    def to_dict(self) -> dict:
        """Flat arrays suitable for `np.savez_compressed`."""
        lengths = np.array([len(s) for s in self._strokes], dtype=np.int64)
        return {'vertices': self.vertices.astype(np.float64),
                'stroke_lengths': lengths,
                'ds': np.array(self.ds, dtype=np.float64)}

    @classmethod
    def from_dict(cls, d) -> 'StrokeGeometry':
        verts = np.asarray(d['vertices'], dtype=np.float64)
        lengths = np.asarray(d['stroke_lengths'], dtype=np.int64)
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        strokes = [verts[bounds[i]:bounds[i + 1]] for i in range(len(lengths))]
        return cls(strokes, ds=float(np.asarray(d['ds'])))

    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.to_dict())

    @classmethod
    def load(cls, path: str) -> 'StrokeGeometry':
        with np.load(path) as d:
            return cls.from_dict(d)

    def __repr__(self) -> str:
        return (f'StrokeGeometry(n_strokes={self.n_strokes}, n_edges={self.n_edges}, '
                f'total_length={self.total_length():.1f}px, ds={self.ds})')
