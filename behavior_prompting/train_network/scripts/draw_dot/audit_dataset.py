"""
Audit a generated `DrawingDotEnv` dataset: is it what we think we generated?

This is a data report, not a correctness proof of the recovery code -- the generator writes the
labels directly, so "can we read back what we wrote" verifies very little. What matters is whether
the dataset actually contains the strategy coverage the experiment depends on:

  * **Balance** -- episodes per manner. The multimodality measurement is read against a uniform
    reference, so a manner that is under-represented is a hole the policy will inherit and the
    reference will be wrong.
  * **Dot placement** -- confirm layouts are spread over the scene rather than clustered by a
    sampling bug, since that spread is the across-instance generalization axis.
  * **Separability, empirically** -- the manners are distinct by construction, but re-check it over
    a sample of REAL layouts (including any near-collinear draws that legitimately occur) rather
    than trusting the argument.
  * **Delta-action sanity** -- that a zero delta normalizes to exactly 0, which is otherwise a
    silent failure.

    python audit_dataset.py -d ../../datasets/draw_dot/dots_train.zarr
"""

import argparse
import sys
from collections import Counter

import numpy as np
import torch

from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.env.draw_dot.layout import (
    MANNERS,
    DotLayout,
    ink_templates,
    mask_iou,
)
from behavior_prompting.train_network.scripts.draw_dot import strategies as ST


def audit(dataset_path: str, n_separability: int, iou_thresh: float) -> bool:
    rb = ReplayBuffer.create_from_path(dataset_path)
    names = np.asarray(rb.task_names[:])
    unique = np.unique(names)
    ok = True

    print(f'{dataset_path}')
    print(f'  {rb.n_episodes} episodes / {rb.n_steps} steps / {len(unique)} instances')

    # -- balance --------------------------------------------------------------------------
    strategy_id = np.asarray(rb.labels['strategy_id']).ravel()
    lab_ends = np.asarray(rb.meta['task_labels_ends'])
    lab_starts = np.concatenate([[0], lab_ends[:-1]])
    per_episode = [int(strategy_id[lo]) for lo, hi in zip(lab_starts, lab_ends)]
    counts = Counter(per_episode)
    print('  episodes per manner:')
    for i, m in enumerate(MANNERS):
        print(f'    {m:9s} {counts.get(i, 0)}')
    expected = rb.n_episodes / len(MANNERS)
    for i, m in enumerate(MANNERS):
        if abs(counts.get(i, 0) - expected) > 0.02 * expected + 1:
            ok = False
            print(f'  FAIL: {m} is unbalanced ({counts.get(i, 0)} vs expected ~{expected:.0f})')

    # episode length by manner -- PARALLEL is legitimately longer (four approaches)
    lengths = {m: [] for m in MANNERS}
    for (lo, hi), sid in zip(zip(lab_starts, lab_ends), per_episode):
        lengths[MANNERS[sid]].append(hi - lo)
    print('  mean episode length: ' +
          '  '.join(f'{m}:{np.mean(v):.0f}' for m, v in lengths.items() if v))

    # -- dot placement --------------------------------------------------------------------
    first_step = [int(rb.task_data_ends[i]) - int(rb.task_lengths[i]) for i in range(rb.n_tasks)]
    dots = np.asarray(rb.data['dots'])[first_step].reshape(-1, 2)
    print(f'  dot placement: x [{dots[:, 0].min():.3f}, {dots[:, 0].max():.3f}] '
          f'mean {dots[:, 0].mean():.3f} | y [{dots[:, 1].min():.3f}, {dots[:, 1].max():.3f}] '
          f'mean {dots[:, 1].mean():.3f}')
    # a uniform sample over the margin box should sit near its centre
    if not (0.4 < dots.mean() < 0.6):
        ok = False
        print(f'  FAIL: dot placement looks clustered (overall mean {dots.mean():.3f})')

    # -- separability over REAL layouts ----------------------------------------------------
    worst = 0.0
    idxs = np.linspace(0, len(unique) - 1, min(n_separability, len(unique))).astype(int)
    for k in idxs:
        i = int(np.where(names == unique[k])[0][0])
        start = int(rb.task_data_ends[i]) - int(rb.task_lengths[i])
        layout = DotLayout(dots=np.asarray(rb.data['dots'][start]).reshape(-1, 2).astype(np.float64))
        pen_start = np.asarray(rb.labels['pen_start'][
            int(rb.task_labels_ends[i]) - int(rb.task_lengths[i])]).reshape(3).astype(np.float64)

        demos = ST.make_all(layout, pen_start)
        if not np.array_equal(demos['CONNECT'].xy_path, demos['TOUCH'].xy_path):
            ok = False
            print(f'  FAIL: {unique[k]}: CONNECT.xy != TOUCH.xy (contact-only pair broken)')
        tpl = ink_templates(layout, 96)
        for a in range(len(MANNERS)):
            for b in range(a + 1, len(MANNERS)):
                iou = mask_iou(tpl[MANNERS[a]], tpl[MANNERS[b]])
                worst = max(worst, iou)
                if iou > iou_thresh:
                    ok = False
                    print(f'  FAIL: {unique[k]}: {MANNERS[a]} vs {MANNERS[b]} ink IoU {iou:.3f}')
    print(f'  separability over {len(idxs)} real layouts: worst cross-manner ink IoU {worst:.3f} '
          f'(threshold {iou_thresh})')

    # -- delta-action sanity ---------------------------------------------------------------
    from behavior_prompting.train_network.model.common.normalize_util import (
        array_to_stats,
        get_symmetric_range_normalizer_from_stat,
    )
    norm = get_symmetric_range_normalizer_from_stat(array_to_stats(rb.data['action']))
    with torch.no_grad():
        zero = norm.normalize(torch.zeros(1, 3)).numpy()
    if not np.allclose(zero, 0.0):
        ok = False
        print(f'  FAIL: zero delta normalizes to {zero} rather than 0')
    else:
        print('  zero delta normalizes to exactly 0')

    action = np.asarray(rb.data['action'])
    print(f'  action deltas: |dxy| max {np.abs(action[:, :2]).max():.4f}  '
          f'dz values {np.unique(np.round(action[:, 2], 3))}')

    print(f'\n{"AUDIT PASSED" if ok else "AUDIT FAILED"}')
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', required=True)
    p.add_argument('--n-separability', type=int, default=50,
                   help='how many real layouts to re-check separability on')
    p.add_argument('--iou-threshold', type=float, default=0.5)
    a = p.parse_args()
    return 0 if audit(a.dataset, a.n_separability, a.iou_threshold) else 1


if __name__ == '__main__':
    sys.exit(main())
