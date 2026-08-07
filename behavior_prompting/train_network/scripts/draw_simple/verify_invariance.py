"""
Verify that `SimpleDrawEnv` renders BIT-IDENTICAL drawings across permuted strategies.

For each sampled task this draws the same geometry `--n-strategies` times with independently
sampled stroke orders, traversal directions, velocity profiles and speeds, then checks:

  1. every render equals the reference render exactly       (`np.array_equal`)
  2. every render equals the canonical goal image exactly    -- i.e. rasterising the task's
     geometry directly, never from a rollout
  3. edge coverage is exactly 1.0

This is the counterpart to `scripts/draw/verify_strategy_invariance.py`, which reports IoU
against a 0.98 threshold because the PD-tracked env can only approach invariance. Here the
threshold is equality, so any nonzero pixel difference is a bug in the geometry/env/strategy
stack rather than an expected noise floor. The script exits nonzero if anything differs.

    python verify_invariance.py --n-tasks 5 --n-strategies 12 --verbose
"""

import sys

import click
import numpy as np

from behavior_prompting.train_network.env.draw_simple.draw_simple_env import (
    BOUNDARY_ANGLE_HIGH,
    BOUNDARY_ANGLE_LOW,
    SimpleDrawEnv,
)
import shapes as SH
import strategy as ST


def draw_with_strategy(env, geom, strategy, boundary_angle, control_hz,
                       rng, noise_std, noise_bounds):
    """Execute a strategy with no recording and return the finished drawing image."""
    env.reset(boundary_angle=boundary_angle)
    for stroke_idx, reverse, profile, speed in ST.realized_strokes(strategy):
        actions, _ = ST.build_stroke_actions(
            geom, stroke_idx, reverse, speed, control_hz, profile, boundary_angle,
            rng=rng, noise_std=noise_std, noise_bounds=noise_bounds,
            max_step_len=env.step_cap)
        target = actions[0, :2].astype(np.float64)
        guard = int(np.ceil(2 * env.window_size / env.step_cap)) + 4
        for _ in range(guard):
            if float(np.linalg.norm(env.cursor - target)) <= 1e-9:
                break
            env.step(np.array([target[0], target[1], 0.0]))
        else:
            raise ValueError('pen-up approach failed')
        for action in actions:
            env.step(action)
    return env.get_drawing_image(), env.get_goal_image(), env.coverage()


@click.command()
@click.option('--n-tasks', default=5, type=int)
@click.option('--n-strategies', default=10, type=int)
@click.option('--n-strokes', default=3, type=int)
@click.option('--min-parts-per-stroke', default=1, type=int)
@click.option('--max-parts-per-stroke', default=2, type=int)
@click.option('--primitives', default='line,arc,bezier')
@click.option('--ds', default=1.0, type=float)
@click.option('--min-stroke-separation', default=0.0, type=float)
@click.option('--speed-min', default=100.0, type=float)
@click.option('--speed-max', default=400.0, type=float)
@click.option('--profile-families', default=','.join(ST.PROFILE_FAMILIES))
@click.option('--profile-strength', default=0.7, type=float)
@click.option('--stroke-speed-ratio', default=2.0, type=float)
@click.option('--noise-std', default=0.5, type=float)
@click.option('--noise-bounds', default=1.5, type=float)
@click.option('--snap-tol', default=3.0, type=float)
@click.option('--control-hz', default=10, type=int)
@click.option('--vary-angle-per-strategy', is_flag=True,
              help='Also resample the board angle per strategy. Off by default because the '
                   'angle is a property of the demo, not of the strategy, so varying it is '
                   'expected to change pixels.')
@click.option('--seed', default=0, type=int)
@click.option('--verbose', is_flag=True)
def main(**kw):
    primitives = tuple(p.strip() for p in kw['primitives'].split(',') if p.strip())
    families = tuple(p.strip() for p in kw['profile_families'].split(',') if p.strip())

    failures = 0
    comparisons = 0

    for task_idx in range(kw['n_tasks']):
        geom_rng = np.random.default_rng([kw['seed'], task_idx, 0xC0FFEE])
        geom = SH.sample_task_geometry(
            geom_rng, n_strokes=kw['n_strokes'],
            parts_per_stroke=(kw['min_parts_per_stroke'], kw['max_parts_per_stroke']),
            ds=kw['ds'], allowed=primitives,
            min_stroke_separation=kw['min_stroke_separation'])

        env = SimpleDrawEnv(geometry=geom, boundary_angle=0.0, snap_tol=kw['snap_tol'],
                            control_hz=kw['control_hz'])
        assert kw['noise_bounds'] < env.snap_tol

        base_angle = float(np.random.default_rng([kw['seed'], task_idx, 1]).uniform(
            BOUNDARY_ANGLE_LOW, BOUNDARY_ANGLE_HIGH))

        print(f'\ntask {task_idx}: {geom!r}')
        reference = None
        for k in range(kw['n_strategies']):
            rng = np.random.default_rng([kw['seed'], task_idx, k, 7])
            angle = (float(rng.uniform(BOUNDARY_ANGLE_LOW, BOUNDARY_ANGLE_HIGH))
                     if kw['vary_angle_per_strategy'] else base_angle)
            strategy = ST.sample_strategy(
                rng, geom.n_strokes, float(rng.uniform(kw['speed_min'], kw['speed_max'])),
                profile_families=families, profile_strength=kw['profile_strength'],
                profile_per_stroke=True, stroke_speed_ratio=kw['stroke_speed_ratio'])

            drawing, goal, coverage = draw_with_strategy(
                env, geom, strategy, angle, kw['control_hz'], rng,
                kw['noise_std'], kw['noise_bounds'])

            if reference is None:
                reference = drawing.copy()

            same_ref = np.array_equal(drawing, reference)
            same_goal = np.array_equal(drawing, goal)
            full = coverage == 1.0
            ok = same_ref and same_goal and full
            comparisons += 1
            if not ok:
                failures += 1

            if kw['verbose'] or not ok:
                n_ref = int((drawing != reference).any(axis=2).sum())
                n_goal = int((drawing != goal).any(axis=2).sum())
                print(f'  strategy {k:2d}: {"OK " if ok else "FAIL"} '
                      f'coverage={coverage:.6f} px_diff_vs_ref={n_ref} px_diff_vs_goal={n_goal} '
                      f'| {strategy!r}')

    print(f'\n{comparisons} comparisons, {failures} failures')
    if failures:
        print('Renders are NOT strategy-invariant. This is a bug, not a noise floor.')
        sys.exit(1)
    print('All renders are bit-identical to the reference AND to the canonical goal image.')


if __name__ == '__main__':
    main()
