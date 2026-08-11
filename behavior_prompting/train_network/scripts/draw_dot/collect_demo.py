"""
Collect a human demonstration of a *manner* on a `DrawingDotEnv` layout, with the mouse.

What this is for
----------------
The policy has no manner conditioning: at fixed dots it samples a manner from its own residual
multimodality (see `docs/drawanything_dots.md` §The multimodality experiment). The intended steering
signal is a human demonstration of the manner you want -- given on a **different dot layout** than
the one being steered, so that what transfers is the manner and not the solution.

That last point is why the demo layout here is freshly sampled and unrelated to any eval instance. A
demo that shared the target's dots would be a solution, and a policy that copied it would tell us
nothing about whether the *manner* generalized.

⚠️ This diverges from what `docs/drawanything_dots.md` §Steering originally specified, which was an
**ungrounded** collector: blank canvas, no dots, pure motion. Grounding the demo on a layout is
strictly more informative -- because there are dots, the recorded demo can be replayed through the
env and run through `classify_manner`, so the tool reports which of the four manners you actually
demonstrated instead of leaving you to hope. An ungrounded path cannot be classified at all.

How capture works
-----------------
The env is driven **live** from the mouse: each frame the pen is commanded toward the cursor, clipped
to `max_delta`, with `z` taken from the left button (down = inking). So the ink you see accumulating
is the demonstration, `dot_coverage` updates as you touch dots, and the recorded actions are real env
actions rather than something replayed afterwards.

The capture loop runs much faster than the policy's 10 Hz control rate, which is deliberate: at
`capture_fps=60` the `max_delta` clip corresponds to about 4.8 canvas widths per second, so it
effectively never binds and the drawn shape is not distorted by the speed limit. The cost is an
action stream at capture rate rather than control rate, so **two** streams are saved:

    action_live  what was actually executed, at capture rate -- preserves the human's pacing
    action       the same path re-emitted at `strategies.DEFAULT_SPEED` -- matches how all four
                 scripted manners are generated, and so is the comparable one

Neither is privileged here. Which one a steering controller should consume is an open question, so
both are kept along with the raw timestamped path.

Usage
-----
Standalone:

    python scripts/draw_dot/collect_demo.py --out demos/ --seed 0

Or press `d` inside `scripts/draw_dot/interactive_rollout.py`.

Keys: drag = draw · `space`/Enter finish · `r` restart this layout · `l` new layout · `q`/Esc cancel.
"""

import argparse
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from behavior_prompting.train_network.env.draw_dot.draw_dot_env import PEN_DOWN_Z, DrawingDotEnv
from behavior_prompting.train_network.env.draw_dot.layout import (
    MANNERS,
    DotLayout,
    sample_layout,
)
from behavior_prompting.train_network.scripts.draw_dot import strategies as ST
from behavior_prompting.train_network.utils import dot_strategy_metrics as DM

WINDOW = 'draw_dot demonstration'
Z_DOWN, Z_UP = 0.0, 1.0
DEFAULT_CAPTURE_FPS = 60


class _Mouse:
    """Cursor position in scene coordinates, plus button state, from a cv2 mouse callback."""

    def __init__(self, img_size: int):
        self.img_size = int(img_size)
        self.pos: Optional[np.ndarray] = None      # None until the cursor enters the canvas
        self.down = False

    def __call__(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.down = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.down = False
        # Ignore anything below the canvas (the HUD panel), but keep the last good position.
        if y >= self.img_size:
            return
        s = max(self.img_size - 1, 1)
        self.pos = np.clip(np.array([x / s, y / s], dtype=np.float64), 0.0, 1.0)


def _compose(env: DrawingDotEnv, scale: int, hud: List[str]) -> np.ndarray:
    img = np.ascontiguousarray(env.render(mode='rgb_array')[:, :, ::-1])
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    panel = np.full((22 * len(hud) + 12, img.shape[1], 3), 255, np.uint8)
    for i, line in enumerate(hud):
        cv2.putText(panel, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, panel])


def _nearest_dot_index(layout: DotLayout, xy: np.ndarray) -> np.ndarray:
    """Per-sample nearest dot. A best-effort stand-in for the scripted `target_dot_index`."""
    d = np.linalg.norm(layout.dots[None, :, :] - np.asarray(xy)[:, None, :], axis=-1)
    return np.argmin(d, axis=1).astype(np.int64)


def capture_once(env: DrawingDotEnv, layout: DotLayout, scale: int = 3,
                 capture_fps: int = DEFAULT_CAPTURE_FPS,
                 title: str = '') -> Tuple[str, Optional[Dict]]:
    """
    Drive the env from the mouse until the user finishes.

    Returns `('done', record)`, or `(command, None)` for `restart` / `relayout` / `cancel`.
    """
    img_size = env.render_size * scale
    mouse = _Mouse(img_size)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, mouse)
    delay = max(1, round(1000 / capture_fps))

    # Reset once up front purely so the wait screen has something to render -- `_compose_render`
    # dereferences `layout` and `ink`, both of which are None until a reset. The pen start is
    # provisional and is replaced below.
    env.reset(layout=layout, pen_start=np.array([0.5, 0.5, Z_UP]))

    # The pen has to start somewhere, and starting it at the cursor avoids an opening jump across the
    # canvas that would be recorded as part of the demonstration.
    while mouse.pos is None:
        cv2.imshow(WINDOW, _compose(env, scale, ['move the cursor onto the canvas to begin',
                                                 'drag to draw, release to lift', 'q = cancel']))
        if (cv2.waitKey(delay) & 0xFF) in (ord('q'), 27):
            return 'cancel', None
    pen_start = np.array([mouse.pos[0], mouse.pos[1], Z_UP], dtype=np.float64)
    env.reset(layout=layout, pen_start=pen_start)

    raw_xy: List[np.ndarray] = [pen_start[:2].copy()]
    raw_z: List[float] = [Z_UP]
    raw_t: List[float] = [0.0]
    live_actions: List[np.ndarray] = []
    t0 = time.perf_counter()

    while True:
        target_xy = mouse.pos
        target_z = Z_DOWN if mouse.down else Z_UP
        delta_xy = np.clip(target_xy - env.pen[:2], -env.max_delta, env.max_delta)
        action = np.array([delta_xy[0], delta_xy[1], target_z - env.pen[2]], dtype=np.float64)
        env.step(action)

        live_actions.append(action)
        raw_xy.append(target_xy.copy())
        raw_z.append(target_z)
        raw_t.append(time.perf_counter() - t0)

        hud = [title or 'demonstrate a manner',
               f'cov {env.dot_coverage():.2f}  prec {env.ink_precision():.2f}  '
               f'strokes {env.n_strokes()}  steps {len(live_actions)}  '
               f'{"DOWN" if env.pen_down else "up"}',
               'space/enter = finish   r = restart   l = new layout   q = cancel']
        cv2.imshow(WINDOW, _compose(env, scale, hud))
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord(' '), 13, 10):
            break
        if key == ord('r'):
            return 'restart', None
        if key == ord('l'):
            return 'relayout', None
        if key in (ord('q'), 27):
            return 'cancel', None

    if len(live_actions) < 2:
        print('  demo too short, nothing recorded')
        return 'restart', None

    return 'done', {
        'raw_xy': np.asarray(raw_xy, dtype=np.float32),
        'raw_z': np.asarray(raw_z, dtype=np.float32),
        'raw_t': np.asarray(raw_t, dtype=np.float32),
        'action_live': np.asarray(live_actions, dtype=np.float32),
        'pen_start': pen_start.astype(np.float32),
    }


def finalize(record: Dict, layout: DotLayout, canvas_size: int,
             speed: float = ST.DEFAULT_SPEED) -> Dict:
    """
    Re-emit the captured path at the canonical speed, replay it, and classify the result.

    The replay is what makes the demo self-checking: it is run through a *fresh* env, so the ink being
    classified is the ink the canonical action stream actually produces, not the ink the live capture
    left behind. If those two disagree the demo is not reproducible and the printed coverage will say
    so.
    """
    raw_xy = record['raw_xy'].astype(np.float64)
    raw_z = record['raw_z'].astype(np.float64)
    pen_start = record['pen_start'].astype(np.float64)

    demo = ST.path_to_demo(raw_xy, raw_z, pen_start,
                           target_dot_index=_nearest_dot_index(layout, raw_xy),
                           manner='HUMAN', speed=speed)

    replay = DrawingDotEnv(canvas_size=canvas_size)
    replay.reset(layout=layout, pen_start=pen_start)
    for a in demo.actions:
        replay.step(a)
    result = DM.evaluate_env(replay, layout)

    out = dict(record)
    out.update({
        'xy_path': demo.xy_path.astype(np.float32),
        'z_path': demo.z_path.astype(np.float32),
        'action': demo.actions.astype(np.float32),
        'target_dot_index': demo.target_dot_index.astype(np.int64),
        'dots': layout.dots.astype(np.float32),
        'dot_order': np.asarray(layout.order, dtype=np.int64),
        'ink': replay.ink.astype(np.uint8),
        'manner': np.asarray(result.manner or '', dtype='U16'),
        'manner_names': np.asarray(MANNERS, dtype='U16'),
        'manner_ious': np.asarray([result.manner_ious.get(m, np.nan) for m in MANNERS],
                                  dtype=np.float32),
        'dot_coverage': np.float32(result.dot_coverage),
        'ink_precision': np.float32(result.ink_precision),
        'n_strokes': np.int64(result.n_strokes),
        'speed': np.float32(speed),
        'canvas_size': np.int64(canvas_size),
        'pen_down_z': np.float32(PEN_DOWN_Z),
    })
    return out


def describe(final: Dict) -> str:
    ious = ' '.join(f'{m}:{v:.2f}' for m, v in zip(final['manner_names'], final['manner_ious']))
    return (f"looks like {str(final['manner']) or '-'}  "
            f"cov {float(final['dot_coverage']):.3f}  prec {float(final['ink_precision']):.3f}  "
            f"strokes {int(final['n_strokes'])}  [{ious}]  "
            f"{len(final['action'])} steps ({len(final['action_live'])} captured)")


def save_demo(final: Dict, out_dir: str, index: int) -> str:
    manner = str(final['manner']) or 'UNCLASSIFIED'
    path = pathlib.Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    fn = path / f'demo_{index:04d}_{manner}.npz'
    np.savez_compressed(fn, **final)
    return str(fn)


def collect_demo(out_dir: str, rng: np.random.Generator, canvas_size: int = 96,
                 render_size: int = 256, scale: int = 3,
                 capture_fps: int = DEFAULT_CAPTURE_FPS, start_index: int = 0,
                 max_demos: Optional[int] = None) -> List[str]:
    """
    Interactive loop: sample a layout, capture, classify, confirm, save. Returns the paths written.

    Confirmation is a separate step from capture on purpose. The classification is the only signal
    that the intended manner is the one that came out, and it does not exist until the path has been
    re-emitted and replayed -- so the user needs a chance to see it and reject the demo.
    """
    env = DrawingDotEnv(canvas_size=canvas_size, render_size=render_size)
    saved: List[str] = []
    layout = sample_layout(rng)
    idx = start_index
    try:
        while max_demos is None or len(saved) < max_demos:
            title = f'demo {len(saved) + 1}' + (f'/{max_demos}' if max_demos else '')
            cmd, record = capture_once(env, layout, scale=scale, capture_fps=capture_fps,
                                       title=title)
            if cmd == 'cancel':
                break
            if cmd == 'relayout':
                layout = sample_layout(rng)
                continue
            if cmd == 'restart':
                continue

            final = finalize(record, layout, canvas_size)
            print(f'  {describe(final)}')

            # Show the canonical replay, not the live capture, so what is confirmed is what is saved.
            hud = [describe(final)[:78],
                   'enter/s = save   r = redo   l = new layout   q = discard and quit']
            replay = DrawingDotEnv(canvas_size=canvas_size, render_size=render_size)
            replay.reset(layout=layout, pen_start=final['pen_start'].astype(np.float64))
            for a in final['action']:
                replay.step(a)
            decision = None
            while decision is None:
                cv2.imshow(WINDOW, _compose(replay, scale, hud))
                key = cv2.waitKey(50) & 0xFF
                decision = {13: 'save', 10: 'save', ord('s'): 'save', ord('r'): 'redo',
                            ord('l'): 'relayout', ord('q'): 'quit', 27: 'quit'}.get(key)
            if decision == 'save':
                fn = save_demo(final, out_dir, idx)
                saved.append(fn)
                idx += 1
                print(f'  saved {fn}')
                layout = sample_layout(rng)      # a fresh layout for the next manner
            elif decision == 'relayout':
                layout = sample_layout(rng)
            elif decision == 'quit':
                break
    finally:
        cv2.destroyWindow(WINDOW)
        env.close()
    return saved


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='demos/draw_dot', help='directory for the .npz demos')
    p.add_argument('--seed', type=int, default=0, help='seeds the demo layout sampling')
    p.add_argument('--canvas-size', type=int, default=96)
    p.add_argument('--render-size', type=int, default=256)
    p.add_argument('--scale', type=int, default=3, help='window magnification')
    p.add_argument('--capture-fps', type=int, default=DEFAULT_CAPTURE_FPS)
    p.add_argument('--n-demos', type=int, default=None, help='stop after this many saved demos')
    p.add_argument('--start-index', type=int, default=None,
                   help='first filename index; default continues from what is in --out')
    a = p.parse_args(argv)

    start = a.start_index
    if start is None:
        existing = sorted(pathlib.Path(a.out).glob('demo_*.npz')) if pathlib.Path(a.out).is_dir() \
            else []
        start = 1 + max((int(f.name.split('_')[1]) for f in existing), default=-1)

    print(f'Collecting into {a.out} starting at index {start}')
    print('drag = draw   space/enter = finish   r = restart   l = new layout   q = cancel')
    saved = collect_demo(a.out, np.random.default_rng(a.seed), canvas_size=a.canvas_size,
                         render_size=a.render_size, scale=a.scale, capture_fps=a.capture_fps,
                         start_index=start, max_demos=a.n_demos)
    print(f'\n{len(saved)} demo(s) saved')
    for s in saved:
        print(f'  {s}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
