"""
A simplified drawing environment whose rendered ink is provably invariant to *how* a shape is
drawn -- stroke order, traversal direction, and velocity profile.

Why not `env/draw/draw_env.py`
------------------------------
There, the agent is a PD-tracked pymunk body (k_p=100, k_v=20) and the canvas records the
ACHIEVED position each control step. Both facts leak the "how" into the pixels:

  * the achieved path lags the commanded path by roughly 0.2*v px, so a faster traversal of the
    same curve paints a visibly different (corner-cut) trail;
  * velocity is state that survives across parts, so a stroke entered at speed renders
    differently from the same stroke entered from rest -- which means permuting stroke order
    changes pixels.

Everything built to fight that (settle holds, curvature-derived speed caps, per-part minimum
sample counts, end holds) reduces the leak but cannot remove it; measured render IoU across
permuted strategies lands around 0.95-0.997.

This env removes the leak instead of bounding it:

  1. Kinematic agent. `cursor <- action[:2]`, with the per-step displacement capped at
     `max_speed / control_hz`. No velocity state at all, so nothing carries across a stroke
     boundary and stroke order cannot couple to anything.
  2. Ink is an edge set, not a rasterised trajectory. The env snaps the swept segment onto the
     task's canonical polyline and records which EDGES were traversed (see `geometry.py`). The
     image is a pure function of that set, and set union is commutative, so order / direction /
     speed are invariant by construction rather than by measurement.

Consequence worth being explicit about: the env inks the traversed ARC, not the straight chord
between two commanded points. That is deliberate -- it is exactly what makes a coarse (fast)
execution paint the same curve as a fine (slow) one. The `max_speed` cap bounds how much arc a
single action can claim.

Positional noise is free here. Any perturbation that stays within `snap_tol` of the polyline
changes no ink whatsoever, so you can add action/observation noise for robustness without it
showing up in the goal image -- the opposite of the old env, where `noise_std` went straight
into pixels.

Action space
------------
`Box(low=[0, 0, 0], high=[512, 512, 1])`, i.e. `[x, y, pen]` in absolute canvas pixels, with
`pen = action[2] > 0.5`. The pen dim is a continuous relaxation of a binary choice, exactly as
a parallel-jaw gripper dim usually is -- it keeps the whole action vector continuous for
diffusion / flow-matching policies. The pen state of step t applies to the entire segment swept
during step t, which makes ink well defined without any sub-step interpolation.
"""

from typing import Optional, Tuple

import cv2
import gym
import numpy as np
from gym import spaces

from behavior_prompting.train_network.env.draw_simple.geometry import (
    BACKGROUND_COLOR,
    BOARD_CENTER,
    GOAL_COLOR,
    INK_COLOR,
    WINDOW_SIZE,
    StrokeGeometry,
    board_to_canvas,
    canvas_to_board,
)

BOUNDARY_ANGLE_LOW = -np.pi / 4
BOUNDARY_ANGLE_HIGH = np.pi / 4

WALL_COLOR = (0, 0, 0)
WALL_HIGHLIGHT_COLOR = (255, 0, 0)   # the "top" wall, so board orientation is readable
HELPER_COLOR = (211, 211, 211)
AGENT_COLOR = (48, 156, 54)


class SimpleDrawEnv(gym.Env):
    """
    Kinematic, edge-set drawing environment.

    Args:
        geometry: the task's `StrokeGeometry`. May be set later via `set_geometry`.
        boundary_angle: board rotation in radians. None randomises it on every reset.
        render_size: side length of the observation image.
        render_cache_size: side length of the human-facing render (defaults to render_size).
        control_hz: control frequency; only used to turn `max_speed` into a per-step cap.
        max_speed: maximum cursor speed in px/s. Caps the displacement of one step.
        pen_radius: half the pen width in px (6 -> a 12 px stroke, matching the old env).
        snap_tol: how far off the canonical polyline the cursor may be, pen down, and still
            ink. Keep it well under `pen_radius` so "on path" is visually exact.
        board_length: side length of the square board.
        off_path_penalty: reward penalty per step spent pen-down beyond `snap_tol`.
    """

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}
    reward_range = (-np.inf, 1.0)

    def __init__(self,
                 geometry: Optional[StrokeGeometry] = None,
                 boundary_angle: Optional[float] = 0.0,
                 render_size: int = 224,
                 render_cache_size: Optional[int] = None,
                 control_hz: int = 10,
                 max_speed: float = 1200.0,
                 pen_radius: int = 6,
                 snap_tol: float = 3.0,
                 board_length: float = 350.0,
                 render_mode: str = 'rgb_array',
                 overlay_goal: bool = True,
                 overlay_agent: bool = True,
                 off_path_penalty: float = 0.01):
        self.window_size = WINDOW_SIZE
        self.render_size = render_size
        self.render_cache_size = render_cache_size if render_cache_size is not None else render_size
        assert self.render_cache_size <= self.window_size
        self.control_hz = int(control_hz)
        self.max_speed = float(max_speed)
        self.pen_radius = int(pen_radius)
        self.snap_tol = float(snap_tol)
        self.board_length = float(board_length)
        self.mode = render_mode
        self.overlay_goal = overlay_goal
        self.overlay_agent = overlay_agent
        self.off_path_penalty = float(off_path_penalty)

        assert self.snap_tol < self.pen_radius, (
            f'snap_tol ({self.snap_tol}) should be well under pen_radius ({self.pen_radius}); '
            f'otherwise "on path" is visibly off path.')

        self.randomize_boundary_angle = boundary_angle is None
        self.boundary_angle = 0.0 if boundary_angle is None else float(boundary_angle)

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=(3, render_size, render_size),
                                dtype=np.float32),
            'agent_pos': spaces.Box(low=0, high=self.window_size, shape=(2,), dtype=np.float32),
            'pen_down': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        # [x, y, pen]. Continuous box with a thresholded pen dim -- see the module docstring.
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]),
                                       high=np.array([self.window_size, self.window_size, 1.0]),
                                       shape=(3,), dtype=np.float64)

        self._seed = None
        self.seed()

        self.geometry: Optional[StrokeGeometry] = None
        self.goal_edges: Optional[np.ndarray] = None
        self.inked: Optional[np.ndarray] = None
        self.cursor = np.array([BOARD_CENTER, BOARD_CENTER], dtype=np.float64)
        self.pen_down = False
        self._active_stroke = None
        self.latest_action = None
        self.render_cache = None
        self.n_off_path_steps = 0
        # Per-step record of WHICH stroke was being drawn and HOW far along it -- the ground truth
        # from which a rollout's realised strategy (order, direction, pacing) is recovered. Kept on
        # the env rather than passed out through `info` because MultiStepWrapper truncates info to
        # the last `n_obs_steps` entries of each action chunk; a runner reads this attribute
        # directly instead. The name must not collide with anything on the wrappers, since
        # `get_attr` resolves through gym's __getattr__ forwarding.
        self.stroke_trace: list = []

        self._background = None
        self._ink_mask = None
        self._goal_mask = None

        if geometry is not None:
            self.set_geometry(geometry)

    # -- setup --------------------------------------------------------------------------

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(np.random.randint(0, 2 ** 31 - 1))
        self._seed = int(seed)
        self.np_random = np.random.default_rng(self._seed)
        return [self._seed]

    def set_geometry(self, geometry: StrokeGeometry, goal_edges: Optional[np.ndarray] = None):
        """
        Install a task's geometry. `goal_edges` defaults to every edge, i.e. "draw the whole
        shape"; pass a subset if you ever want a partial-shape goal.
        """
        self.geometry = geometry
        self.goal_edges = (geometry.all_edges_mask() if goal_edges is None
                           else np.asarray(goal_edges, dtype=bool).copy())
        assert self.goal_edges.shape == (geometry.n_edges,)

    @property
    def step_cap(self) -> float:
        """Maximum displacement of a single control step, in px."""
        return self.max_speed / self.control_hz

    def reset(self, boundary_angle: Optional[float] = None,
              start_pos: Optional[np.ndarray] = None):
        assert self.geometry is not None, 'call set_geometry() before reset()'

        if boundary_angle is not None:
            self.boundary_angle = float(boundary_angle)
        elif self.randomize_boundary_angle:
            self.boundary_angle = float(self.np_random.uniform(BOUNDARY_ANGLE_LOW,
                                                              BOUNDARY_ANGLE_HIGH))

        self.inked = self.geometry.empty_mask()
        self.n_off_path_steps = 0
        self.pen_down = False
        self._active_stroke = None
        self.latest_action = None
        self.render_cache = None
        self.stroke_trace = []

        if start_pos is None:
            half = self.board_length / 2 - 40.0
            off = self.np_random.uniform(-half, half, size=2)
            start_pos = board_to_canvas(BOARD_CENTER + off, self.boundary_angle)[0]
        self.cursor = np.clip(np.asarray(start_pos, dtype=np.float64).reshape(2),
                              0.0, self.window_size)

        self._background = self._build_background()
        self._ink_mask = np.zeros((self.window_size, self.window_size), dtype=np.uint8)
        self._goal_mask = self._build_goal_mask()
        return self._get_obs()

    # -- dynamics -----------------------------------------------------------------------

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """
        Move the cursor toward `action[:2]`, capped at `step_cap`, and ink the arc swept.

        Clipping the step is safe for invariance: it changes the arc-length SCHEDULE (and hence
        the episode length) but not the union of the intervals, so a strategy that commands
        overlong hops still paints exactly the same edges.
        """
        assert self.geometry is not None and self.inked is not None, 'call reset() first'
        action = np.asarray(action, dtype=np.float64).reshape(3)
        self.latest_action = action.copy()

        target = np.clip(action[:2], 0.0, self.window_size)
        delta = target - self.cursor
        dist = float(np.linalg.norm(delta))
        if dist > self.step_cap:
            target = self.cursor + delta * (self.step_cap / dist)

        prev = self.cursor
        self.cursor = target
        self.pen_down = bool(action[2] > 0.5)

        off_path = False
        stroke_idx, arc_s = -1, float('nan')
        if self.pen_down:
            off_path, stroke_idx, arc_s = self._ink(prev, self.cursor)
        else:
            # lifting the pen ends the current stroke, so the next pen-down re-acquires
            self._active_stroke = None
        if off_path:
            self.n_off_path_steps += 1

        obs = self._get_obs()
        coverage = self.coverage()
        self.stroke_trace.append((stroke_idx, arc_s, self.pen_down, coverage))
        reward = self.compute_reward() - (self.off_path_penalty if off_path else 0.0)
        info = {'off_path': off_path,
                'n_off_path_steps': self.n_off_path_steps,
                'coverage': coverage,
                'edge_iou': self.edge_iou(),
                # Also in `info` for debugging a single step. NOT the source for strategy metrics
                # -- see the note on `self.stroke_trace`.
                'active_stroke': stroke_idx,
                'arc_s': arc_s}
        return obs, reward, False, info

    def _ink(self, prev_canvas: np.ndarray, cur_canvas: np.ndarray) -> Tuple[bool, int, float]:
        """
        Record the edges traversed while moving `prev -> cur` with the pen down.

        Returns `(off_path, stroke_idx, arc_s)`. `off_path` is True if the move was off the
        canonical path, in which case nothing is inked and the other two are `(-1, nan)`.
        `arc_s` is the arc length reached along stroke `stroke_idx`, which is what makes
        traversal direction and pacing recoverable from a rollout.

        Stroke attribution is CONTINUOUS: once a pen-down run has acquired a stroke it keeps
        following that stroke until the cursor leaves `snap_tol` of it. Re-deciding per step from
        geometry alone is wrong where two strokes cross -- both are within tolerance there, so a
        step could be credited to the wrong stroke, inking edges that were already inked and
        leaving the intended ones blank. That silently cost ~1.5% of coverage on tasks with
        crossing strokes and broke bit-exactness for the affected strategies. Continuity is also
        just what a pen does: a stroke does not teleport onto a different curve mid-contact.
        """
        prev_b = canvas_to_board(prev_canvas, self.boundary_angle)[0]
        cur_b = canvas_to_board(cur_canvas, self.boundary_angle)[0]

        k = self._active_stroke
        if k is not None:
            s_prev, d_prev = self.geometry.project(prev_b, k)
            s_cur, d_cur = self.geometry.project(cur_b, k)
            if max(d_prev, d_cur) > self.snap_tol:
                k = None  # left the stroke we were following -- fall through and re-acquire
        if k is None:
            k, s_prev, s_cur, worst = self.geometry.project_pair(prev_b, cur_b)
            if worst > self.snap_tol:
                self._active_stroke = None
                return True, -1, float('nan')
        self._active_stroke = k

        new_edges = self.geometry.edges_in(k, s_prev, s_cur)
        fresh = new_edges[~self.inked[new_edges]]
        if fresh.size:
            self.inked[new_edges] = True
            # Incremental painting is exactly equal to rasterising the whole set -- painting is
            # idempotent and writes a set of pixels determined only by the edge. See
            # `StrokeGeometry.paint_edges`.
            self.geometry.paint_edges(self._ink_mask, fresh, self.boundary_angle,
                                     self.pen_radius, 1)
        return False, int(k), float(s_cur)

    # -- reward / metrics ---------------------------------------------------------------

    def coverage(self) -> float:
        """Fraction of the goal edge set that has been inked."""
        if self.goal_edges is None or not self.goal_edges.any():
            return 1.0
        return float((self.inked & self.goal_edges).sum() / self.goal_edges.sum())

    def edge_iou(self) -> float:
        """IoU over edge sets -- exact, and cheap enough to call every step."""
        if self.goal_edges is None:
            return 0.0
        union = int((self.inked | self.goal_edges).sum())
        if union == 0:
            return 1.0
        return float((self.inked & self.goal_edges).sum() / union)

    def pixel_iou(self) -> float:
        """IoU of the rendered ink masks, for comparison with the old env's metric."""
        a = self._ink_mask.astype(bool)
        b = self._goal_mask.astype(bool)
        union = int((a | b).sum())
        if union == 0:
            return 1.0
        return float((a & b).sum() / union)

    def compute_reward(self) -> float:
        return self.edge_iou()

    # -- rendering ----------------------------------------------------------------------

    def _board_corners(self) -> np.ndarray:
        half = self.board_length / 2
        corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        return board_to_canvas(BOARD_CENTER + corners, self.boundary_angle)

    def _build_background(self) -> np.ndarray:
        img = np.empty((self.window_size, self.window_size, 3), dtype=np.uint8)
        img[:] = np.asarray(BACKGROUND_COLOR, dtype=np.uint8)
        pts = [tuple(int(v) for v in p) for p in np.rint(self._board_corners())]
        tl, tr, br, bl = pts
        # helper lines first so the walls draw over them
        for a, b in ((tl, br), (tr, bl),
                     (self._mid(tl, tr), self._mid(bl, br)),
                     (self._mid(tl, bl), self._mid(tr, br))):
            cv2.line(img, a, b, HELPER_COLOR, 4, cv2.LINE_8)
        for i in range(4):
            color = WALL_HIGHLIGHT_COLOR if i == 0 else WALL_COLOR
            cv2.line(img, pts[i], pts[(i + 1) % 4], color, 4, cv2.LINE_8)
        return img

    @staticmethod
    def _mid(a, b):
        return (int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2))

    def _build_goal_mask(self) -> np.ndarray:
        mask = np.zeros((self.window_size, self.window_size), dtype=np.uint8)
        edges = np.nonzero(self.goal_edges)[0]
        if edges.size:
            self.geometry.paint_edges(mask, edges, self.boundary_angle, self.pen_radius, 1)
        return mask

    def _compose(self, with_goal: bool) -> np.ndarray:
        frame = self._background.copy()
        if with_goal and self.overlay_goal:
            frame[self._goal_mask.astype(bool)] = GOAL_COLOR
        frame[self._ink_mask.astype(bool)] = INK_COLOR
        if self.overlay_agent:
            c = tuple(int(v) for v in np.rint(self.cursor))
            cv2.circle(frame, c, 15, AGENT_COLOR, -1, cv2.LINE_8)
            cv2.circle(frame, c, 15, BACKGROUND_COLOR, 2, cv2.LINE_8)
        return frame

    def _get_obs(self) -> dict:
        obs_frame = self._compose(with_goal=False)
        img = cv2.resize(obs_frame, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_AREA)

        cache = self._compose(with_goal=True)
        self._draw_hud(cache)
        self.render_cache = cv2.resize(cache, (self.render_cache_size, self.render_cache_size),
                                       interpolation=cv2.INTER_AREA)

        return {'image': np.moveaxis(img.astype(np.float32) / 255.0, -1, 0),
                'agent_pos': self.cursor.astype(np.float32),
                'pen_down': np.array([float(self.pen_down)], dtype=np.float32)}

    def _draw_hud(self, img: np.ndarray) -> None:
        if self.latest_action is not None:
            x, y = int(round(self.latest_action[0])), int(round(self.latest_action[1]))
            cv2.drawMarker(img, (x, y), (0, 0, 0), cv2.MARKER_CROSS, 18, 1, cv2.LINE_8)
        cv2.putText(img, f'IoU {self.edge_iou():.3f}', (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        if self.render_cache is None:
            self._get_obs()
        return self.render_cache

    def get_drawing_image(self) -> np.ndarray:
        """(512, 512, 3) uint8: white background, blue ink, nothing else."""
        img = np.empty((self.window_size, self.window_size, 3), dtype=np.uint8)
        img[:] = np.asarray(BACKGROUND_COLOR, dtype=np.uint8)
        img[self._ink_mask.astype(bool)] = INK_COLOR
        return img

    def get_goal_image(self) -> np.ndarray:
        """
        (512, 512, 3) uint8 rendering of the goal edge set.

        This is the strategy-free target: it is rasterised from the task's canonical geometry,
        never from a rollout. A completed demo's `get_drawing_image()` equals this array
        exactly, for every strategy -- which is the property this whole env exists to provide.
        """
        img = np.empty((self.window_size, self.window_size, 3), dtype=np.uint8)
        img[:] = np.asarray(BACKGROUND_COLOR, dtype=np.uint8)
        img[self._goal_mask.astype(bool)] = INK_COLOR
        return img

    def close(self):
        pass
