"""
Show what varied across the demos of each task, and what differs between tasks.

Answers two different questions that are easy to conflate:

1. "Did the STRATEGY vary within a task?"  Look at strategy_order_rank / order_inversions and
   at the recovered stroke-onset order. Within a task these SHOULD differ across demos.

2. "Do the tasks draw different SHAPES?"  Look at the ink signature. Across tasks this should
   differ; within a task it should NOT -- the whole point of order permutation is that the
   finished drawing is identical, so comparing final images or drawing videos within a task
   will correctly show no difference. That is the invariance, not a failure.

Usage:
    python inspect_strategy.py --dataset datasets/draw/procedural_order
    python inspect_strategy.py --dataset datasets/draw/procedural_order --show-strokes
"""

import glob
import os

import click
import numpy as np


def load_task(store_path, labels_wanted):
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
    from behavior_prompting.common.replay_buffer import ReplayBuffer
    register_codecs(verbose=False)

    rb = ReplayBuffer.create_from_path(store_path, mode='r')
    task_names = np.asarray(rb.task_names[:])
    task_lengths = np.asarray(rb.task_lengths[:])
    task_data_ends = np.asarray(rb.task_data_ends[:])
    task_labels_ends = np.asarray(rb.task_labels_ends[:])
    available = [k for k in labels_wanted if k in rb.labels]

    demos = []
    for t in range(len(task_names)):
        d_end = int(task_data_ends[t]); d_start = d_end - int(task_lengths[t])
        l_end = int(task_labels_ends[t]); l_start = l_end - int(task_lengths[t])
        row = {'task': str(task_names[t]), 'n_steps': d_end - d_start,
               'action': np.asarray(rb.data['action'][d_start:d_end], dtype=np.float64)}
        for k in available:
            vals = np.asarray(rb.labels[k][l_start:l_end], dtype=np.float64).reshape(-1)
            row[k] = float(vals[0]) if vals.size else float('nan')
        demos.append(row)
    return demos, available


def stroke_onsets(action, boundary_angle):
    """
    Recover the execution order of strokes as the sequence of pen-down run centroids, in the
    upright frame. This is what the order axis actually is, read back off the trajectory.
    """
    xy, pen = action[:, :2], action[:, 2] > 0.5
    c, s = np.cos(-boundary_angle), np.sin(-boundary_angle)
    rel = xy - 256.0
    up = np.stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c], axis=1) + 256.0

    edges = np.diff(pen.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    stops = list(np.where(edges == -1)[0] + 1)
    if pen[0]:
        starts = [0] + starts
    if pen[-1]:
        stops = stops + [pen.size]
    return [up[a:b].mean(axis=0) for a, b in zip(starts, stops) if b - a >= 2]


def ink_signature(action, boundary_angle, grid=32, step_px=2.0):
    """
    Occupancy grid of the drawn path in the upright frame -- an order- and direction-independent
    fingerprint of WHAT was drawn. Identical within a task, different across tasks.

    The polyline between consecutive pen-down samples is RASTERIZED, not just point-sampled.
    Point sampling looks correct but is not: at 200 px/s and 10 Hz consecutive commanded samples
    are ~20 px apart while a grid cell is 16 px, so cells between samples are missed, and two
    demos of the same task that were sampled at different speeds get different signatures. That
    produced a spurious within-task IoU of 0.73-0.88 for drawings that are in fact identical.
    """
    xy, pen = action[:, :2], action[:, 2] > 0.5
    occ = np.zeros((grid, grid), dtype=bool)

    edges = np.diff(pen.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    stops = list(np.where(edges == -1)[0] + 1)
    if pen[0]:
        starts = [0] + starts
    if pen[-1]:
        stops = stops + [pen.size]

    c, s = np.cos(-boundary_angle), np.sin(-boundary_angle)
    for a, b in zip(starts, stops):
        seg = xy[a:b]
        if seg.shape[0] < 2:
            continue
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        dense = [seg[0]]
        for i, dist in enumerate(d):
            n = max(int(dist / step_px), 1)
            t = np.linspace(0.0, 1.0, n + 1)[1:, None]
            dense.append(seg[i] + t * (seg[i + 1] - seg[i]))
        pts = np.vstack([np.atleast_2d(p) for p in dense])
        rel = pts - 256.0
        up = np.stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c], axis=1)
        idx = np.clip(((up + 256.0) / 512.0 * grid).astype(int), 0, grid - 1)
        occ[idx[:, 1], idx[:, 0]] = True

    # Dilate by one cell. The real pen is 12 px wide against a 16 px cell, so a hairline
    # occupancy grid is the wrong model and is hypersensitive at cell boundaries: two identical
    # drawings sampled at different densities landed at IoU 0.94 purely from sub-cell
    # registration. Dilating both sides makes the comparison tolerant to that while still
    # separating genuinely different shapes.
    padded = np.pad(occ, 1)
    dilated = np.zeros_like(occ)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            dilated |= padded[dy:dy + grid, dx:dx + grid]
    return dilated


LABELS = ['boundary_angle', 'strategy_n_strokes', 'strategy_order_rank',
          'strategy_order_first', 'strategy_order_last', 'strategy_order_inversions',
          'strategy_frac_reversed', 'strategy_base_speed', 'strategy_realized_speed',
          'strategy_max_radial_error_px', 'strategy_n_parts_capped', 'strategy_n_parts_floored']


@click.command()
@click.option('--dataset', required=True, help='Directory of per-task .zarr stores, or one store.')
@click.option('--show-strokes', is_flag=True, help='Print each demo\'s stroke centroid sequence.')
def main(dataset, show_strokes):
    if os.path.isdir(os.path.join(dataset, 'meta')):
        stores = [dataset]
    else:
        stores = sorted(glob.glob(os.path.join(dataset, '*.zarr')))
    if not stores:
        raise click.ClickException(f'no .zarr stores found under {dataset}')

    task_signatures = {}

    for store in stores:
        demos, available = load_task(store, LABELS)
        name = demos[0]['task'] if demos else os.path.basename(store)
        print(f'\n=== {os.path.basename(store)}  ({len(demos)} demos, task "{name}") ===')

        if 'strategy_order_rank' not in available:
            print('  NO strategy_* labels in this store -- it predates the strategy-variation')
            print('  generator, so its demos cannot have varied in order. Regenerate to get them.')

        cols = [k for k in LABELS if k in available]
        hdr = f"{'demo':<6}{'steps':>7}" + ''.join(f'{k.replace("strategy_", ""):>22}' for k in cols)
        print(hdr)
        print('-' * len(hdr))
        for i, d in enumerate(demos):
            print(f'{i:<6}{d["n_steps"]:>7}' + ''.join(f'{d[k]:>22.3f}' for k in cols))

        # did the strategy actually vary?
        if 'strategy_order_rank' in available:
            ranks = [d['strategy_order_rank'] for d in demos]
            n_strokes = {d['strategy_n_strokes'] for d in demos}
            print(f'\n  distinct order_rank values: {len(set(ranks))} of {len(ranks)} demos '
                  f'-> {sorted(set(ranks))}')
            if n_strokes == {1.0}:
                print('  ** every demo has ONE stroke, so there is only one possible order and')
                print('     --vary-order is a no-op. Stroke count is 1 + the number of movement')
                print('     parts: raise --max-parts, or pass --parts oval,movement.')
            elif len(set(ranks)) == 1:
                print('  ** order_rank is constant: order did NOT vary across these demos.')
            else:
                print('  ** order VARIED across demos. This is the strategy variation you want.')

        # recovered onset order, independent of the labels
        onsets = [stroke_onsets(d['action'], d.get('boundary_angle', 0.0)) for d in demos]
        counts = {len(o) for o in onsets}
        print(f'  recovered pen-down runs per demo: {sorted(len(o) for o in onsets)}')
        if show_strokes:
            for i, o in enumerate(onsets):
                seq = ' -> '.join(f'({p[0]:.0f},{p[1]:.0f})' for p in o)
                print(f'    demo {i}: {seq}')
        if len(counts) == 1 and counts != {1} and len(onsets) > 1:
            first = [tuple(np.round(o[0]).astype(int)) for o in onsets if o]
            print(f'  first stroke drawn, per demo (upright centroid): {first}')
            print(f'    -> {len(set(first))} distinct starting strokes across {len(first)} demos')

        # WHAT was drawn: should be identical within a task
        sigs = [ink_signature(d['action'], d.get('boundary_angle', 0.0)) for d in demos]
        ious = []
        for s in sigs[1:]:
            inter = np.logical_and(s, sigs[0]).sum()
            union = np.logical_or(s, sigs[0]).sum()
            ious.append(inter / union if union else 1.0)
        if ious:
            print(f'  ink signature IoU vs demo 0: min={min(ious):.3f} mean={np.mean(ious):.3f}')
            print('    (should be ~1.0 -- the drawing is meant to be identical across demos;')
            print('     that is the invariance, not a missing difference)')
        task_signatures[os.path.basename(store)] = sigs[0]

    if len(task_signatures) > 1:
        print('\n=== across tasks: are the SHAPES different? ===')
        names = list(task_signatures)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = task_signatures[names[i]], task_signatures[names[j]]
                union = np.logical_or(a, b).sum()
                iou = np.logical_and(a, b).sum() / union if union else 1.0
                verdict = 'DIFFERENT shapes' if iou < 0.5 else ('similar' if iou < 0.95
                                                                else '** SUSPICIOUSLY IDENTICAL')
                print(f'  {names[i]} vs {names[j]}: ink IoU={iou:.3f}  {verdict}')


if __name__ == '__main__':
    main()
