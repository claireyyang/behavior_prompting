"""
Check that per-demo strategy variation does not change the rendered drawing.

The premise of a goal-image-conditioned strategy experiment is that the goal image says WHAT
to draw and nothing about HOW. If two demos of a task render different images, the goal image
partially reveals the strategy and the what/how split leaks.

Two things make that non-trivial in DrawEnv:

* The canvas records the agent's ACHIEVED position, not the commanded action, and the agent is
  a PD-tracked body (k_p=100, k_v=20). So rendering depends on the dynamics, not just the
  commanded geometry.
* Tangential lag (k_v/k_p)*v is a pure time delay and repaints the same pixels, but tracking a
  curve of radius R at speed v needs centripetal acceleration v^2/R, which the P term supplies
  only by sitting off-path by about v^2/(R*k_p). That radial error is what changes the image,
  and it scales with speed and with curvature.

So ORDER and DIRECTION should be pixel-clean (no part's speed changes), while PROFILE and
PER-PART SPEED should not be, unless --curvature-tol-px bounds them. This script measures which
is true rather than assuming.

Usage:
    python verify_strategy_invariance.py --n-strategies 12
    python verify_strategy_invariance.py --profile-families linear,ease_in_out,two_peak \
                                         --part-speed-ratio 1.5 --curvature-tol-px 3.0
    python verify_strategy_invariance.py --save-dir /tmp/strategy_check   # dump the renders

Reports, against the first (reference) strategy: IoU of the inked masks, the fraction of
disagreeing pixels, and the predicted worst-case radial error. IoU below --iou-threshold is a
failure and exits non-zero.
"""

import os
import sys

import click
import numpy as np

from behavior_prompting.train_network.env.draw.draw_env import DrawEnv

import strategy_variation as SV
from procedural_generate_drawings import (
    generate_procedural_trajectory,
    generate_movement_control_points,
    compute_bezier_point,
)

BOARD_CENTER = 256.0


def ink_mask(drawing_image: np.ndarray) -> np.ndarray:
    """
    Boolean mask of inked pixels. The pen is opaque blue (0,0,255) on white, so 'not white'
    isolates the ink without depending on the exact anti-aliasing.
    """
    img = np.asarray(drawing_image)
    if img.ndim == 2:
        return img < 250
    return np.any(img < 250, axis=-1)


def execute_demo(env, strokes, strategy, control_hz, noise_std, part_delay_min, part_delay_max,
                 curvature_tol_px, settle_tolerance=1.0, settle_max_steps=200,
                 settle_hold_steps=10, end_hold_steps=10, max_total_steps=8000):
    """
    Run one strategy in the env and return (inked mask, worst predicted radial error).

    Mirrors the generator's executor: every stroke is approached with the pen up and settled to
    rest before drawing, which is what makes order permutation renderable identically.
    """
    realized = SV.apply_strategy(strokes, strategy)
    SV.assert_geometry_preserved(strokes, realized)

    obs = env.reset(no_rotation=True)
    steps = 0
    worst_radial = 0.0

    def step(act):
        nonlocal obs, steps
        obs, _, _, _ = env.step(np.asarray(act, dtype=np.float32))
        steps += 1
        if steps > max_total_steps:
            raise RuntimeError(f'demo exceeded {max_total_steps} steps')

    def approach_and_settle(target):
        current = np.array(obs['agent_pos'], dtype=np.float64)
        c1, c2 = generate_movement_control_points(current, target, env.board_length, 20.0)
        for i in range(20):
            current = np.array(obs['agent_pos'], dtype=np.float64)
            if np.linalg.norm(current - target) < settle_tolerance:
                break
            t = i / 19.0
            p = compute_bezier_point(current, c1, c2, target, t)
            step([p[0], p[1], 0.0])
        arrived = False
        for _ in range(settle_max_steps):
            current = np.array(obs['agent_pos'], dtype=np.float64)
            if np.linalg.norm(current - target) < settle_tolerance:
                arrived = True
                break
            step([target[0], target[1], 0.0])
        if not arrived:
            raise RuntimeError(f'failed to settle at {target}')
        # hold the exact target so position AND velocity are driven to ~zero, making the
        # stroke's starting state independent of the randomized approach path
        for _ in range(settle_hold_steps):
            step([target[0], target[1], 0.0])

    for exec_idx, stroke in enumerate(realized):
        family, strength = strategy.profiles[exec_idx]
        acts, info = SV.build_stroke_actions(
            stroke, control_hz, strategy.base_speed,
            speed_mults=strategy.speed_mults[exec_idx],
            profile=(family, strength),
            noise_std=noise_std, noise_bounds=20.0,
            part_delay_min=part_delay_min, part_delay_max=part_delay_max,
            curvature_tol_px=curvature_tol_px,
            end_hold_steps=end_hold_steps)  # every stroke, not just the last
        worst_radial = max(worst_radial, max((pi['radial_error_px'] for pi in info), default=0.0))

        approach_and_settle(np.array([acts[0][0], acts[0][1]], dtype=np.float64))
        for act in acts:
            step(act)

    return ink_mask(env.get_drawing_image()), worst_radial


@click.command()
@click.option('--n-strategies', default=12, type=int, help='Strategies to render and compare.')
@click.option('--n-tasks', default=3, type=int, help='Distinct task shapes to test.')
@click.option('--min-parts', default=4, type=int)
@click.option('--max-parts', default=6, type=int)
@click.option('--connection-probability', default=0.5, type=float,
              help='Chance a part ends on an earlier endpoint. Does NOT affect stroke count: '
                   'consecutive parts are always chained start-to-end.')
@click.option('--parts', default=None,
              help="Comma-separated allowed part types. Stroke count is 1 + the number of "
                   "'movement' parts, so e.g. 'oval,movement' or 'straight,movement' forces many "
                   "strokes (movements cannot be consecutive, so ~half the parts become breaks).")
@click.option('--control-hz', default=10, type=int)
@click.option('--speed', default=200.0, type=float, help='Base speed in px/s.')
@click.option('--noise-std', default=0.0, type=float,
              help='Keep 0 to isolate strategy effects; per-step noise alone changes pixels.')
@click.option('--part-delay-min', default=0, type=int,
              help='Pen-down dwell at interior part boundaries. Measured near-irrelevant to '
                   'render invariance; the direction leak is radial tracking error, not corners.')
@click.option('--part-delay-max', default=0, type=int)
@click.option('--vary-order/--no-vary-order', default=True)
@click.option('--vary-direction/--no-vary-direction', default=False,
              help='OFF by default: measured not image-invariant. Enable to re-measure the leak.')
@click.option('--profile-families', default='linear')
@click.option('--profile-strength', default=0.7, type=float)
@click.option('--part-speed-ratio', default=1.0, type=float)
@click.option('--curvature-tol-px', default=None, type=float)
@click.option('--settle-tolerance', default=1.0, type=float,
              help='Pixel radius the cursor must reach before drawing starts.')
@click.option('--settle-hold-steps', default=10, type=int,
              help='Extra hold steps after arriving. Set 0 to reproduce the ~0.99 IoU noise floor.')
@click.option('--end-hold-steps', default=10, type=int,
              help='Pen-down hold at each stroke end. Set 0 to reproduce the truncation bug.')
@click.option('--iou-threshold', default=0.98, type=float,
              help='Minimum acceptable IoU against the reference render.')
@click.option('--seed', default=0, type=int)
@click.option('--save-dir', default=None, help='Write reference/variant/diff PNGs here.')
def main(n_strategies, n_tasks, min_parts, max_parts, connection_probability, parts, control_hz,
         speed, noise_std, part_delay_min, part_delay_max, vary_order, vary_direction, profile_families,
         profile_strength, part_speed_ratio, curvature_tol_px, settle_tolerance,
         settle_hold_steps, end_hold_steps, iou_threshold, seed, save_dir):
    families = tuple(f.strip() for f in profile_families.split(',') if f.strip())
    varies_speed = families != ('linear',) or part_speed_ratio > 1.0

    print(f'axes: order={vary_order} direction={vary_direction} '
          f'profiles={list(families)} part_speed_ratio={part_speed_ratio} '
          f'curvature_tol_px={curvature_tol_px}')
    if varies_speed:
        print('NOTE: profile / per-part-speed variation changes local speed, which moves the')
        print('      ACHIEVED path on curved parts. Expect IoU < 1 unless curvature_tol_px is set.')
    print()

    env = DrawEnv(boundary_angle=0, render_mode='rgb_array')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_ious = []
    failures = 0

    for task_idx in range(n_tasks):
        np.random.seed(seed * 1000 + task_idx)
        allowed = [t.strip() for t in parts.split(',')] if parts else None
        trajectory_parts, _ = generate_procedural_trajectory(
            control_hz, env.board_length, 20.0, min_parts, max_parts,
            connection_probability=connection_probability, min_distance=50.0,
            partial_oval_probability=0.5, allowed_parts=allowed,
            movement_allowed_at_end_of_episode=False)
        strokes = SV.group_parts_into_strokes(trajectory_parts)

        n_perms = 1
        for i in range(1, len(strokes) + 1):
            n_perms *= i
        print(f'task {task_idx}: {len(trajectory_parts)} parts -> {len(strokes)} strokes '
              f'({[len(s) for s in strokes]} parts each), {n_perms} possible orders')
        if len(strokes) < 2:
            print('  single stroke -- order permutation is a no-op here, skipping')
            continue

        ref_mask = None
        ref_strategy = None
        for k in range(n_strategies):
            np.random.seed(seed * 1000 + task_idx * 100 + k)
            if k == 0:
                strategy = SV.DemoStrategy(list(range(len(strokes))), [False] * len(strokes),
                                           [('linear', 0.0)] * len(strokes),
                                           [[1.0] * len(s) for s in strokes], speed)
            else:
                strategy = SV.sample_demo_strategy(
                    strokes, speed, vary_order=vary_order, vary_direction=vary_direction,
                    profile_families=families, profile_strength=profile_strength,
                    part_speed_ratio=part_speed_ratio)

            mask, worst_radial = execute_demo(
                env, strokes, strategy, control_hz, noise_std, part_delay_min,
                part_delay_max, curvature_tol_px,
                settle_tolerance=settle_tolerance, settle_hold_steps=settle_hold_steps,
                end_hold_steps=end_hold_steps)

            if k == 0:
                ref_mask, ref_strategy = mask, strategy
                print(f'  reference: {int(mask.sum())} inked px, '
                      f'predicted worst radial error {worst_radial:.1f} px')
                if save_dir:
                    _save(os.path.join(save_dir, f'task{task_idx}_ref.png'), mask)
                continue

            inter = int(np.logical_and(mask, ref_mask).sum())
            union = int(np.logical_or(mask, ref_mask).sum())
            iou = inter / union if union else 1.0
            disagree = int(np.logical_xor(mask, ref_mask).sum())
            all_ious.append(iou)
            ok = iou >= iou_threshold
            failures += int(not ok)
            print(f'  strategy {k:2d}: IoU={iou:.4f} disagreeing_px={disagree:5d} '
                  f'worst_radial={worst_radial:5.1f}px  order={strategy.order} '
                  f'rev={[int(b) for b in strategy.reversed_flags]} '
                  f'prof={strategy.profiles[0][0]}  {"OK" if ok else "MISMATCH"}')
            if save_dir and not ok:
                _save(os.path.join(save_dir, f'task{task_idx}_strategy{k}.png'), mask)
                _save(os.path.join(save_dir, f'task{task_idx}_strategy{k}_diff.png'),
                      np.logical_xor(mask, ref_mask))

    env.close()

    print()
    if all_ious:
        arr = np.array(all_ious)
        print(f'{len(arr)} comparisons: IoU min={arr.min():.4f} mean={arr.mean():.4f} '
              f'median={np.median(arr):.4f}')
    else:
        print('no comparisons made -- every task was single-stroke. Lower '
              '--connection-probability or raise --min-parts.')
        sys.exit(1)

    if failures:
        print(f'\n{failures} of {len(all_ious)} renders differ from the reference beyond '
              f'IoU {iou_threshold}.')
        print('If only order/direction were varied, this is a real bug: strategy is leaking into')
        print('the goal image. If profile or per-part speed were varied, this is the expected')
        print('radial-error leak -- set --curvature-tol-px to bound it.')
        sys.exit(1)

    print(f'\nAll renders match the reference within IoU {iou_threshold}. Strategy variation is '
          f'invisible in the goal image for these axes.')


def _save(path, mask):
    try:
        import imageio.v2 as imageio
    except ImportError:
        return
    imageio.imwrite(path, (np.asarray(mask) * 255).astype(np.uint8))


if __name__ == '__main__':
    main()
