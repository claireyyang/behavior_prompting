"""
`DrawingDotEnv` -- one task ("touch all the dots"), many valid manners.

Why this exists next to `env/draw_simple`
-----------------------------------------
`SimpleDrawEnv` optimises for TASK diversity: trace an arbitrary unseen shape. Two of its properties
make it unusable for studying *manner*:

  1. Its ink is an edge set snapped to the target polyline, so inked edges are always a subset of
     goal edges and `edge_iou == coverage` -- a pure recall metric with no precision term. A policy
     that scribbles across the canvas scores exactly as well as one that does nothing.
  2. One shape has one correct answer, so there is no residual freedom for a manner to live in.

This env inverts both. The task is fully specified by the dot set, leaving *how* you touch them
free; and ink is an honest rasterization of wherever the pen actually went, so drawing in the wrong
place leaves a mark and is penalized.

Action space
------------
`[Δx, Δy, Δz]`, continuous, applied as offsets to the current pose -- observations are scene-frame
(absolute) while actions are relative, the usual pairing. Deltas are stored natively in the dataset,
so there is no absolute<->relative conversion layer anywhere: the policy predicts deltas and the env
integrates them.

`z ∈ [0, 1]` with contact at `z <= 0.5`. The pen state AFTER a step governs the whole segment swept
during that step, which makes ink well defined without sub-step interpolation. This is the same
convention `SimpleDrawEnv` documents, kept so the semantics stay familiar.
"""

from collections import deque
from typing import Optional, Tuple

import cv2
import gym
import numpy as np
from gym import spaces

from behavior_prompting.train_network.env.draw_dot.layout import (
    BACKGROUND_COLOR,
    DOT_COLOR,
    DOT_RADIUS,
    INK_COLOR,
    N_DOTS,
    PEN_DOWN_COLOR,
    PEN_RADIUS_PX,
    PEN_UP_COLOR,
    SCENE_MAX,
    SCENE_MIN,
    WALL_COLOR,
    DotLayout,
    allowed_ink_mask,
    compose_canvas,
    draw_polyline,
    sample_layout,
    sample_pen_start,
    to_px,
)

PEN_DOWN_Z = 0.5
ACTION_BUFFER_LEN = 8      # matches exec_action_horizon: the actions actually executed per chunk


class DrawingDotEnv(gym.Env):
    """
    Args:
        layout: the instance to draw. May be set later via `set_layout`.
        canvas_size: side length of the observation canvas.
        max_delta: per-step cap on |Δ| in xy (scene units). A speed limit.
        dot_radius: contact radius; the pen must be within this of a dot, pen down, to touch it.
        ink_tol_px: dilation of the allowed-ink region when scoring precision.
        render_size: side length of the human/video render.
    """

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}
    reward_range = (0.0, 1.0)

    def __init__(self,
                 layout: Optional[DotLayout] = None,
                 canvas_size: int = 96,
                 max_delta: float = 0.08,
                 max_delta_z: float = 1.0,
                 dot_radius: float = DOT_RADIUS,
                 pen_radius_px: int = PEN_RADIUS_PX,
                 ink_tol_px: int = 3,
                 render_size: int = 256,
                 render_mode: str = 'rgb_array'):
        self.canvas_size = int(canvas_size)
        self.max_delta = float(max_delta)
        self.max_delta_z = float(max_delta_z)
        self.dot_radius = float(dot_radius)
        self.pen_radius_px = int(pen_radius_px)
        self.ink_tol_px = int(ink_tol_px)
        self.render_size = int(render_size)
        self.mode = render_mode

        self.observation_space = spaces.Dict({
            'canvas': spaces.Box(low=0.0, high=1.0,
                                 shape=(3, self.canvas_size, self.canvas_size), dtype=np.float32),
            'dots': spaces.Box(low=SCENE_MIN, high=SCENE_MAX, shape=(N_DOTS, 2), dtype=np.float32),
            'pen_pose': spaces.Box(low=SCENE_MIN, high=SCENE_MAX, shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([-self.max_delta, -self.max_delta, -self.max_delta_z]),
            high=np.array([self.max_delta, self.max_delta, self.max_delta_z]),
            shape=(3,), dtype=np.float64)

        self._seed = None
        self.seed()

        self.layout: Optional[DotLayout] = None
        self.allowed: Optional[np.ndarray] = None
        # `MultiStepWrapper.reset()` takes no arguments, so a runner that needs a specific starting
        # pose has to stash it here first (see `draw_dot_runner.DotEnvSetup`). None means "sample".
        self.pending_pen_start: Optional[np.ndarray] = None
        self.ink: Optional[np.ndarray] = None
        self.pen = np.array([0.5, 0.5, 1.0], dtype=np.float64)
        self.dots_visited: Optional[np.ndarray] = None
        self.latest_action = None
        self.render_cache = None
        self._window = None

        # Rolling window of the actions that were actually EXECUTED (8 of the 16 predicted). Not an
        # observation: handing the policy its own action history would let it infer the manner from
        # its own past rather than from the steering signal, which is the opposite of what we want to
        # measure. It exists because motion axes are functions of actions -- the pacing analysis and
        # the Phase-2 steering controller both read it. Named to collide with nothing on
        # MultiStepWrapper / VideoRecordingWrapper, since `get_attr` forwards through gym wrappers.
        self.executed_action_buffer = deque(maxlen=ACTION_BUFFER_LEN)

        # Per-step trace, for manner recovery. Same reasoning as the buffer: read off the env rather
        # than out of `info`, because MultiStepWrapper truncates info to the last n_obs_steps entries
        # of each action chunk.
        self.dot_trace = []

        if layout is not None:
            self.set_layout(layout)

    # -- setup ---------------------------------------------------------------------------

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(np.random.randint(0, 2 ** 31 - 1))
        self._seed = int(seed)
        self.np_random = np.random.default_rng(self._seed)
        return [self._seed]

    def set_layout(self, layout: DotLayout):
        self.layout = layout
        self.allowed = allowed_ink_mask(layout, self.canvas_size, self.ink_tol_px,
                                        self.pen_radius_px)

    def reset(self, layout: Optional[DotLayout] = None, pen_start: Optional[np.ndarray] = None):
        if layout is not None:
            self.set_layout(layout)
        if self.layout is None:
            self.set_layout(sample_layout(self.np_random))

        if pen_start is None:
            pen_start = (self.pending_pen_start if self.pending_pen_start is not None
                         else sample_pen_start(self.np_random))
        self.pen = np.asarray(pen_start, dtype=np.float64).reshape(3).copy()
        self.pen[:2] = np.clip(self.pen[:2], SCENE_MIN, SCENE_MAX)
        self.pen[2] = np.clip(self.pen[2], 0.0, 1.0)

        self.ink = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        self.dots_visited = np.zeros(self.layout.n_dots, dtype=bool)
        self.latest_action = None
        self.render_cache = None
        self.executed_action_buffer = deque(maxlen=ACTION_BUFFER_LEN)
        self.dot_trace = []
        return self._get_obs()

    # -- dynamics ------------------------------------------------------------------------

    @property
    def pen_down(self) -> bool:
        return bool(self.pen[2] <= PEN_DOWN_Z)

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        assert self.layout is not None and self.ink is not None, 'call reset() first'
        action = np.asarray(action, dtype=np.float64).reshape(3)
        self.latest_action = action.copy()

        delta_xy = np.clip(action[:2], -self.max_delta, self.max_delta)
        delta_z = np.clip(action[2], -self.max_delta_z, self.max_delta_z)

        prev_xy = self.pen[:2].copy()
        self.pen[:2] = np.clip(prev_xy + delta_xy, SCENE_MIN, SCENE_MAX)
        self.pen[2] = np.clip(self.pen[2] + delta_z, 0.0, 1.0)

        # The pen state AFTER the step governs the whole swept segment.
        if self.pen_down:
            draw_polyline(self.ink, np.stack([prev_xy, self.pen[:2]], axis=0),
                          self.canvas_size, self.pen_radius_px, 1)
            touched = self._mark_touched()
        else:
            touched = -1

        # Record what was actually executed -- the env sees exactly the executed actions, so
        # "8 executed of 16 predicted" falls out with no runner bookkeeping.
        self.executed_action_buffer.append(np.array([delta_xy[0], delta_xy[1], delta_z]))
        self.dot_trace.append((touched, self.pen_down, float(self.pen[2])))

        obs = self._get_obs()
        info = {'dot_coverage': self.dot_coverage(),
                'ink_precision': self.ink_precision(),
                'pen_down': self.pen_down,
                'touched_dot': touched}
        return obs, self.compute_reward(), False, info

    def _mark_touched(self) -> int:
        """Mark any dot the pen is currently in contact with. Returns its index, or -1."""
        d = np.linalg.norm(self.layout.dots - self.pen[None, :2], axis=1)
        hit = int(np.argmin(d))
        if d[hit] <= self.dot_radius:
            self.dots_visited[hit] = True
            return hit
        return -1

    # -- metrics -------------------------------------------------------------------------

    def dot_coverage(self) -> float:
        """Recall: fraction of dots touched pen-down. The headline success number."""
        if self.dots_visited is None or len(self.dots_visited) == 0:
            return 0.0
        return float(self.dots_visited.mean())

    def ink_precision(self) -> float:
        """
        Precision: fraction of drawn ink lying inside the allowed region.

        This is the term `SimpleDrawEnv` structurally could not have -- it is what makes drawing in
        the wrong place cost something. 1.0 when nothing has been drawn yet, so an empty canvas is
        not scored as maximally wrong.
        """
        if self.ink is None:
            return 1.0
        total = int(self.ink.sum())
        if total == 0:
            return 1.0
        stray = int((self.ink.astype(bool) & ~self.allowed.astype(bool)).sum())
        return float(1.0 - stray / total)

    def n_strokes(self) -> int:
        """Pen-down runs. Splits {CONNECT, CURVE} (1) from {TOUCH, PARALLEL} (n_dots)."""
        runs, prev = 0, False
        for _, down, _ in self.dot_trace:
            if down and not prev:
                runs += 1
            prev = down
        return runs

    def compute_reward(self) -> float:
        """
        Coverage only.

        Precision is deliberately NOT folded in here. `DrawRunner` aggregates per-step rewards with
        `np.max` over the episode; coverage is monotone so that is harmless, but precision is not --
        under `max`, a policy that touched every dot and then scribbled would score identically to
        one that stopped cleanly. `DrawDotRunner` reads precision from terminal state instead.
        """
        return self.dot_coverage()

    # -- rendering -----------------------------------------------------------------------

    def _get_obs(self) -> dict:
        canvas = compose_canvas(self.ink, self.layout, self.canvas_size)
        dots = np.zeros((N_DOTS, 2), dtype=np.float32)
        dots[:self.layout.n_dots] = self.layout.dots.astype(np.float32)
        return {'canvas': canvas,
                'dots': dots,
                'pen_pose': self.pen.astype(np.float32)}

    def _compose_render(self) -> np.ndarray:
        size = self.render_size
        img = np.empty((size, size, 3), dtype=np.uint8)
        img[:] = np.asarray(BACKGROUND_COLOR, dtype=np.uint8)

        if self.layout.walls is not None:
            for seg in self.layout.walls:
                px = to_px(seg, size)
                cv2.line(img, tuple(px[0]), tuple(px[1]), WALL_COLOR, 3, cv2.LINE_8)

        # dots first so ink draws over them
        r = max(2, int(round(self.dot_radius * (size - 1))))
        for p in to_px(self.layout.dots, size):
            cv2.circle(img, tuple(p), r, DOT_COLOR, -1, cv2.LINE_AA)

        ink_big = cv2.resize(self.ink * 255, (size, size), interpolation=cv2.INTER_NEAREST)
        img[ink_big > 127] = INK_COLOR

        c = tuple(int(v) for v in to_px(self.pen[None, :2], size)[0])
        color = PEN_DOWN_COLOR if self.pen_down else PEN_UP_COLOR
        cv2.circle(img, c, 7, color, -1, cv2.LINE_AA)
        cv2.circle(img, c, 7, BACKGROUND_COLOR, 1, cv2.LINE_AA)

        cv2.putText(img, f'cov {self.dot_coverage():.2f}  prec {self.ink_precision():.2f}  '
                         f'z {self.pen[2]:.2f}',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def render(self, mode: str = 'rgb_array', **kwargs) -> np.ndarray:
        frame = self._compose_render()
        self.render_cache = frame
        if mode == 'human':
            self._show(frame)
        return frame

    def _show(self, frame: np.ndarray) -> None:
        """
        Human window.

        Built in from the start deliberately: `SimpleDrawEnv` had no human render mode, which is
        exactly why the live-demo path could never be used with it and `load_env.py`'s `live_demo`
        branch had to be asserted off. The Phase-2 ungrounded demo collector needs this.
        """
        if self._window is None:
            self._window = 'DrawingDotEnv'
            cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self._window, frame[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        if self._window is not None:
            cv2.destroyWindow(self._window)
            self._window = None
