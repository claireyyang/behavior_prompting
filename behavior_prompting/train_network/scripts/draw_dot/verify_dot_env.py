"""
Verification for `DrawingDotEnv` and the four manners.

Two things this confirms, and nothing else:

  1. **The env behaves as specified** -- delta and pose clipping, ink iff the pen is down, dots
     marked only on pen-down contact, and the executed-action buffer holding the last 8.
  2. **The four manners are actually distinct**, in the specific ways the experiment depends on:
     CONNECT/TOUCH separable only by contact, CONNECT/CURVE separable by kinematics alone, and all
     four leaving mutually distinguishable ink.

(2) gates everything downstream: if it fails, the dataset does not encode the separability the
experiment needs and no amount of training or steering will recover it.

Also included is one metric-calibration assertion -- that a generated demo scores `dot_coverage`
and `ink_precision` of exactly 1.0. That is not testing the demo (which touches every dot by
construction); it is testing that `ink_tol` is wide enough that rasterizing a continuous path at
96 px does not itself register as stray ink.

    python verify_dot_env.py --n-instances 200
"""

import argparse
import sys

import numpy as np

from behavior_prompting.train_network.env.draw_dot.draw_dot_env import DrawingDotEnv
from behavior_prompting.train_network.env.draw_dot.layout import (
    ink_templates,
    mask_iou,
    sample_layout,
    sample_pen_start,
)
from behavior_prompting.train_network.scripts.draw_dot import strategies as ST


def _fail(msg: str) -> bool:
    print(f'  FAIL: {msg}')
    return False


def check_env_mechanics() -> bool:
    """(1) The env does what the spec says."""
    ok = True
    rng = np.random.default_rng(0)
    layout = sample_layout(rng)
    env = DrawingDotEnv(layout=layout, max_delta=0.08)

    # -- delta clipping and pose clipping
    env.reset(pen_start=np.array([0.5, 0.5, 1.0]))
    env.step(np.array([10.0, 0.0, 0.0]))            # absurd delta -> clipped to max_delta
    if not np.isclose(env.pen[0], 0.58):
        ok = _fail(f'delta not clipped to max_delta: x={env.pen[0]} (expected 0.58)')

    env.reset(pen_start=np.array([0.98, 0.5, 1.0]))
    env.step(np.array([0.08, 0.0, 0.0]))            # would leave the scene -> clipped to bounds
    if env.pen[0] > 1.0:
        ok = _fail(f'pose not clipped to scene bounds: x={env.pen[0]}')

    env.reset(pen_start=np.array([0.5, 0.5, 1.0]))
    env.step(np.array([0.0, 0.0, -5.0]))
    if env.pen[2] < 0.0:
        ok = _fail(f'z not clipped to [0,1]: z={env.pen[2]}')

    # -- ink appears iff the pen is down
    env.reset(pen_start=np.array([0.3, 0.3, 1.0]))
    env.step(np.array([0.05, 0.0, 0.0]))            # pen up
    if env.ink.sum() != 0:
        ok = _fail('ink drawn while the pen was up')
    env.step(np.array([0.05, 0.0, -1.0]))           # pen down
    if env.ink.sum() == 0:
        ok = _fail('no ink drawn while the pen was down')

    # -- a dot is marked only on pen-down contact
    env.reset(pen_start=np.array([*layout.dots[0], 1.0]))
    env.step(np.array([0.0, 0.0, 0.0]))             # sitting on a dot, pen UP
    if env.dots_visited.any():
        ok = _fail('dot marked visited with the pen up')
    env.step(np.array([0.0, 0.0, -1.0]))            # same place, pen DOWN
    if not env.dots_visited.any():
        ok = _fail('dot not marked visited on pen-down contact')

    # -- executed-action buffer holds the last 8, and starts empty
    env.reset(pen_start=np.array([0.5, 0.5, 1.0]))
    if len(env.executed_action_buffer) != 0:
        ok = _fail('executed_action_buffer not cleared on reset')
    for i in range(12):
        env.step(np.array([0.01 * i, 0.0, 0.0]))
    if len(env.executed_action_buffer) != 8:
        ok = _fail(f'executed_action_buffer len={len(env.executed_action_buffer)}, expected 8')
    last = np.stack(list(env.executed_action_buffer))
    if not np.allclose(last[-1, 0], min(0.01 * 11, env.max_delta)):
        ok = _fail(f'executed_action_buffer tail wrong: {last[-1]}')

    print(f'  env mechanics: {"OK" if ok else "FAILED"}')
    return ok


def check_separability(n_instances: int, iou_thresh: float, verbose: bool) -> bool:
    """(2) The four manners are distinct, on every instance."""
    ok = True
    n_bad_pair, n_bad_xy, n_bad_contact = 0, 0, 0
    worst_iou = 0.0

    for i in range(n_instances):
        rng = np.random.default_rng([i, 0xD07])
        layout = sample_layout(rng)
        pen_start = sample_pen_start(rng)
        demos = ST.make_all(layout, pen_start)

        contact = {m: (d.z_path <= 0.5) for m, d in demos.items()}
        strokes = {m: int(np.sum(c[1:] & ~c[:-1]) + (1 if c[0] else 0)) for m, c in contact.items()}

        # contact-only pair: identical xy, different contact
        if not np.array_equal(demos['CONNECT'].xy_path, demos['TOUCH'].xy_path):
            n_bad_xy += 1
            ok = _fail(f'instance {i}: CONNECT.xy != TOUCH.xy (contact-only pair broken)')
        if np.array_equal(contact['CONNECT'], contact['TOUCH']):
            n_bad_contact += 1
            ok = _fail(f'instance {i}: CONNECT and TOUCH have identical contact')

        # kinematics-only pair: identical contact structure, different xy
        if strokes['CONNECT'] != 1 or strokes['CURVE'] != 1:
            ok = _fail(f'instance {i}: CONNECT/CURVE strokes = '
                       f'{strokes["CONNECT"]}/{strokes["CURVE"]}, expected 1/1')
        if strokes['TOUCH'] != layout.n_dots or strokes['PARALLEL'] != layout.n_dots:
            ok = _fail(f'instance {i}: TOUCH/PARALLEL strokes = '
                       f'{strokes["TOUCH"]}/{strokes["PARALLEL"]}, expected {layout.n_dots}')
        n = min(len(demos['CONNECT'].xy_path), len(demos['CURVE'].xy_path))
        if np.allclose(demos['CONNECT'].xy_path[:n], demos['CURVE'].xy_path[:n]):
            ok = _fail(f'instance {i}: CONNECT.xy == CURVE.xy (kinematics-only pair broken)')

        # all four leave mutually distinguishable ink
        tpl = ink_templates(layout, 96)
        for a in range(len(ST.MANNERS)):
            for b in range(a + 1, len(ST.MANNERS)):
                ma, mb = ST.MANNERS[a], ST.MANNERS[b]
                iou = mask_iou(tpl[ma], tpl[mb])
                worst_iou = max(worst_iou, iou)
                if iou > iou_thresh:
                    n_bad_pair += 1
                    ok = _fail(f'instance {i}: {ma} vs {mb} ink IoU {iou:.3f} > {iou_thresh}')
        if verbose and i < 3:
            print(f'    instance {i}: strokes={strokes} '
                  f'steps={{{", ".join(f"{m}:{d.n_steps}" for m, d in demos.items())}}}')

    print(f'  separability over {n_instances} instances: {"OK" if ok else "FAILED"} '
          f'(worst cross-manner ink IoU {worst_iou:.3f}, threshold {iou_thresh})')
    return ok


def check_metric_calibration(n_instances: int) -> bool:
    """A perfect demo must score 1.0 / 1.0, or `ink_tol` is miscalibrated for 96 px."""
    ok = True
    worst_cov, worst_prec, worst_case = 1.0, 1.0, ''
    for i in range(n_instances):
        rng = np.random.default_rng([i, 0xCA11])
        layout = sample_layout(rng)
        pen_start = sample_pen_start(rng)
        env = DrawingDotEnv(layout=layout)
        for manner, demo in ST.make_all(layout, pen_start).items():
            env.reset(layout=layout, pen_start=pen_start)
            for a in demo.actions:
                env.step(a)
            cov, prec = env.dot_coverage(), env.ink_precision()
            if cov < worst_cov:
                worst_cov, worst_case = cov, f'{manner}@{i}'
            worst_prec = min(worst_prec, prec)
            if cov < 1.0:
                ok = _fail(f'instance {i} {manner}: dot_coverage {cov:.3f} < 1.0')
            if prec < 1.0:
                ok = _fail(f'instance {i} {manner}: ink_precision {prec:.4f} < 1.0')
    print(f'  metric calibration over {n_instances} instances: {"OK" if ok else "FAILED"} '
          f'(worst coverage {worst_cov:.3f} [{worst_case}], worst precision {worst_prec:.4f})')
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-instances', type=int, default=200)
    p.add_argument('--n-calibration', type=int, default=50)
    p.add_argument('--iou-threshold', type=float, default=0.5,
                   help='max tolerated ink IoU between two different manners')
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args()

    print('DrawingDotEnv verification')
    results = [
        check_env_mechanics(),
        check_separability(a.n_instances, a.iou_threshold, a.verbose),
        check_metric_calibration(a.n_calibration),
    ]
    if all(results):
        print('\nAll checks passed.')
        return 0
    print('\nFAILURES above.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
