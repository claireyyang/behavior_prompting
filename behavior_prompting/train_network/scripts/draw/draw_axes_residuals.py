"""
Recompute the strategy axes of `draw_axes.py` as RESIDUALS against per-task geometric
baselines, and report the within/between decomposition before and after.

Why
---
Most raw axes are contaminated by the drawing's geometry, which is exactly the part the
goal image already reveals. `transit_path_len` is a good example: a demo that visits three
strokes pen-up travels further when the strokes are far apart, so its between-task variance
is mostly board layout, and its ICC measures "are these different drawings" rather than
"is routing a strategy choice". Dividing by the pen-up travel the task's own layout implies
leaves a dimensionless number -- routing cost relative to what this geometry costs -- that
is comparable across tasks and cannot be produced by rescaling a drawing.

Three consequences, and they are the reason to do this:
  * the component the goal image explains is removed, so what is left is HOW not WHAT;
  * the axes become dimensionless, so they transfer to a dataset with a different scale;
  * a baseline that predicts an axis by trivially resampling one trajectory per task can no
    longer score well by matching the geometry, since the geometry has been divided out.

The one trap
------------
A "per-task baseline" must NOT be the per-task mean of the axis being residualized. That
sets the between-task variance to zero by construction and drives the ICC to 0 -- it looks
like a dramatic result and means nothing. Every baseline here is built from a DIFFERENT,
geometry-only quantity (stroke lengths, stroke endpoints, the reset pose, stroke count).
Ratios that do divide by the per-task median of their own numerator are kept, because they
are useful invariance checks, but they are reported in a separate table with no ICC.

Because the geometry is identical across the demos of a task, dividing by a per-task
baseline is a constant rescale WITHIN a task. So the ICC moves for exactly one reason:
heteroscedasticity. Where an axis's within-task spread scales with the geometry, the rescale
shrinks the pooled within-variance and the ICC rises; where its between-task spread was the
geometry, the rescale shrinks between-variance and the ICC falls. Both directions are
informative, and which happens is not predictable in advance -- that is why this measures
rather than asserts.

Usage
-----
    python draw_axes.py --dataset datasets/draw/procedural_order_3stroke --out OUT
    python draw_axes_residuals.py --axes-dir OUT

    # cross-check the recovered stroke order against the generator's own order labels
    python draw_axes_residuals.py --axes-dir OUT --labels-csv strategy_labels.csv

`--labels-csv` wants one row per demo with columns `task`, `demo` and
`strategy_order_rank`, read straight off the store's `strategy_*` labels.
"""

import csv
import itertools
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from draw_axes import curved_only, decompose, kendall_tau

CONTROL_HZ = 10.0


# --------------------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------------------

def read_csv(path: str) -> List[Dict]:
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ('task', 'episode'):
                continue
            try:
                r[k] = float(v) if v != '' else float('nan')
            except (TypeError, ValueError):
                r[k] = float('nan')
    return rows


def nanmedian(vals: Sequence[float]) -> float:
    a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float('nan')


# --------------------------------------------------------------------------------------
# per-task geometric baselines
# --------------------------------------------------------------------------------------

class TaskGeometry:
    """
    Everything about a task that the goal image determines, and nothing about execution.

    Estimated as the per-demo median of geometry-only quantities. That is legitimate --
    within a task every demo draws the same upright geometry, so the median is a
    measurement of the geometry, not a summary of the strategy. It is NOT legitimate for
    execution quantities (speed, dwell, routing), and none are taken from here.

    slot_len[s]     arc length of stroke slot s (px)
    slot_p0/p1[s]   its start / end point, upright frame (px)
    ink_len         total inked arc length (px)
    ink_scale       RMS radius of the stroke endpoints about their centroid (px)
    home            the reset pose the approach starts from (px)
    e_transit       pen-up travel between strokes, averaged over ALL orders
    min_transit     ... and its minimum, i.e. the optimal route
    e_approach      home -> first-stroke-start distance, averaged over the choice of first
    n_slots         stroke count
    """

    def __init__(self, slot_len, slot_p0, slot_p1, home):
        self.slot_len = slot_len
        self.slot_p0 = slot_p0
        self.slot_p1 = slot_p1
        self.home = home
        self.n_slots = len(slot_len)
        self.ink_len = float(np.nansum(slot_len)) if self.n_slots else float('nan')

        pts = np.array([p for p in list(slot_p0) + list(slot_p1) if np.all(np.isfinite(p))])
        self.ink_scale = (float(np.sqrt(np.mean(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1))))
                          if pts.shape[0] >= 2 else float('nan'))

        self.e_transit, self.min_transit = self._transit_stats()
        d0 = [float(np.linalg.norm(p - home)) for p in slot_p0
              if np.all(np.isfinite(p)) and np.all(np.isfinite(home))]
        self.e_approach = float(np.mean(d0)) if d0 else float('nan')

    def _transit_stats(self) -> Tuple[float, float]:
        """
        Pen-up travel implied by an order, over every order.

        A transit leg runs from where one stroke's pen lifted to where the next goes down,
        so it is end(prev) -> start(next). Enumerating all n! orders is fine at n <= 5 and
        gives the exact mean and minimum rather than an estimate. Straight-line legs are
        the right reference: the generator's pen-up moves are straight (transit
        straightness measures ~1 on this data).
        """
        n = self.n_slots
        if n < 2:
            return float('nan'), float('nan')
        p0, p1 = np.asarray(self.slot_p0), np.asarray(self.slot_p1)
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            return float('nan'), float('nan')
        totals = []
        for order in itertools.permutations(range(n)):
            totals.append(sum(float(np.linalg.norm(p0[order[i + 1]] - p1[order[i]]))
                              for i in range(n - 1)))
        return float(np.mean(totals)), float(np.min(totals))


def build_geometry(episode_rows: List[Dict], stroke_rows: List[Dict]) -> Dict[str, TaskGeometry]:
    by_task_slot: Dict[str, Dict[int, List[Dict]]] = {}
    for r in stroke_rows:
        slot = r.get('stroke_slot', float('nan'))
        if not np.isfinite(slot) or slot < 0:
            continue
        by_task_slot.setdefault(r['task'], {}).setdefault(int(slot), []).append(r)

    home_by_task: Dict[str, List[Tuple[float, float]]] = {}
    for r in episode_rows:
        home_by_task.setdefault(r['task'], []).append((r.get('start_x'), r.get('start_y')))

    geom: Dict[str, TaskGeometry] = {}
    for task, slots in by_task_slot.items():
        keys = sorted(slots)
        slot_len = [nanmedian([r['path_len'] for r in slots[s]]) for s in keys]
        slot_p0 = [np.array([nanmedian([r['start_x'] for r in slots[s]]),
                             nanmedian([r['start_y'] for r in slots[s]])]) for s in keys]
        slot_p1 = [np.array([nanmedian([r['end_x'] for r in slots[s]]),
                             nanmedian([r['end_y'] for r in slots[s]])]) for s in keys]
        hs = home_by_task.get(task, [])
        home = np.array([nanmedian([h[0] for h in hs]), nanmedian([h[1] for h in hs])])
        geom[task] = TaskGeometry(slot_len, slot_p0, slot_p1, home)
    return geom


# --------------------------------------------------------------------------------------
# canonical stroke order -- reference-free
# --------------------------------------------------------------------------------------

def canonical_slot_order(g: TaskGeometry) -> List[int]:
    """
    Order the task's stroke slots by upright reading order (y, then x) of their midpoints.

    `draw_axes.order_tau` compares each demo against DEMO 0 of its task, which makes the
    axis reference-dependent: it is undefined for demo 0 itself, and the same execution
    order scores differently depending on which demo happened to be first in the store. A
    canonical geometric order fixes both -- every demo gets a value, and the value means
    the same thing in every task, which is what makes an order axis transferable.
    """
    mids = [(p0 + p1) / 2.0 for p0, p1 in zip(g.slot_p0, g.slot_p1)]
    return sorted(range(g.n_slots), key=lambda s: (round(float(mids[s][1]), 1),
                                                   round(float(mids[s][0]), 1)))


def perm_rank(order: Sequence[int]) -> int:
    """Lexicographic rank of a permutation, matching strategy_variation.order_rank."""
    o = list(order)
    n = len(o)
    rank, avail = 0, sorted(o)
    for i, v in enumerate(o):
        idx = avail.index(v)
        rank += idx * math.factorial(n - 1 - i)
        avail.pop(idx)
    return rank


def add_order_axes(episode_rows: List[Dict], stroke_rows: List[Dict],
                   geom: Dict[str, TaskGeometry]) -> None:
    """
    Write `order_tau_canon` and `order_rank_canon` onto each episode row.

    The execution order is read off `stroke_slot` (which stroke slot was drawn i-th),
    re-expressed in canonical slot indices. Episodes whose strokes did not all match a
    distinct slot are left NaN rather than partially scored.
    """
    strokes_by_ep: Dict[Tuple[str, str], List[Dict]] = {}
    for r in stroke_rows:
        strokes_by_ep.setdefault((r['task'], r['episode']), []).append(r)

    for ep in episode_rows:
        ep['order_tau_canon'] = float('nan')
        ep['order_rank_canon'] = float('nan')
        g = geom.get(ep['task'])
        rows = strokes_by_ep.get((ep['task'], ep['episode']))
        if g is None or not rows or g.n_slots < 2:
            continue
        rows = sorted(rows, key=lambda r: r['stroke_idx'])
        slots = [int(r['stroke_slot']) for r in rows
                 if np.isfinite(r.get('stroke_slot', float('nan'))) and r['stroke_slot'] >= 0]
        if len(slots) != g.n_slots or len(set(slots)) != g.n_slots:
            continue
        canon = canonical_slot_order(g)
        pos = {s: i for i, s in enumerate(canon)}
        seq = [pos[s] for s in slots]
        ep['order_tau_canon'] = kendall_tau(seq)
        ep['order_rank_canon'] = float(perm_rank(seq))


# --------------------------------------------------------------------------------------
# residual axes
# --------------------------------------------------------------------------------------

def safe_div(a, b, lo=1e-9):
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)) or abs(b) < lo:
        return float('nan')
    return a / b


def add_episode_residuals(episode_rows: List[Dict], geom: Dict[str, TaskGeometry],
                          v_ref: float, control_hz: float = CONTROL_HZ) -> None:
    for ep in episode_rows:
        g = geom.get(ep['task'])
        if g is None:
            continue

        # --- routing: how far pen-up, relative to what this layout implies ---
        # ~1 means the demo's route costs what an average order costs; the spread across
        # demos of one task IS the order axis, expressed as a cost rather than a label.
        ep['transit_vs_mean_order'] = safe_div(ep.get('transit_path_len'), g.e_transit)
        # Where the demo's route sits between the best and the average order: 0 = optimal,
        # 1 = as costly as an average order. NOT transit/min_transit -- parts of a chain
        # share endpoints, so the optimal route through some layouts costs ~0 px and that
        # ratio diverges (it reached a mean of 12 with a within-task std of 19 on this
        # dataset, which is a division artifact, not a strategy signal).
        span = g.e_transit - g.min_transit if np.isfinite(g.e_transit) else float('nan')
        ep['transit_excess_norm'] = safe_div(
            (ep.get('transit_path_len', float('nan')) - g.min_transit), span,
            lo=0.05 * g.e_transit if np.isfinite(g.e_transit) else 1e-9)

        # --- approach: detour factor, against the straight line it could have taken ---
        # baseline averages over WHICH stroke is first, so the choice of first stroke stays
        # in the residual; that choice is part of the order strategy.
        ep['approach_vs_mean_first'] = safe_div(ep.get('approach_path_len'), g.e_approach)
        # No second approach residual against "the leg this demo actually flew": that leg is
        # approach_path_len * approach_straightness, so the ratio collapses to
        # 1/straightness, which draw_axes already reports as `approach_shape`.

        # --- time: episode length against the length this geometry implies at v_ref ---
        travel = g.ink_len + (g.e_transit if np.isfinite(g.e_transit) else 0.0) + \
            (g.e_approach if np.isfinite(g.e_approach) else 0.0)
        implied_steps = safe_div(travel * control_hz, v_ref)
        ep['time_vs_geometry'] = safe_div(ep.get('total_steps'), implied_steps)
        # ... and with the stationary steps removed. `time_vs_geometry` still carries a
        # geometry-dependent overhead: dwell is sampled in STEPS and the final hold is a
        # fixed 10, so a short drawing pays proportionally more of it and the ratio keeps a
        # between-task component that is not speed. Netting dwell out isolates traversal.
        ep['move_time_vs_geometry'] = safe_div(
            ep.get('total_steps', float('nan')) - ep.get('dwell_steps', float('nan')),
            implied_steps)

        # --- speed, dimensionless against the dataset's own reference speed ---
        ep['speed_rel'] = safe_div(ep.get('speed_px_s'), v_ref)

        # --- pause: dwell steps per part boundary the geometry implies ---
        # The generator inserts an inter-part delay at each part boundary and holds 10 steps
        # at the end, so the count of boundaries is geometry and the length of each is
        # strategy. n_slots-1 transits plus the final hold is the boundary count that does
        # not depend on the demo.
        ep['dwell_per_boundary'] = safe_div(ep.get('dwell_steps'), max(g.n_slots, 1))

        # --- noise, in units of the demo's own step displacement ---
        # noise_sigma in px is coupled to speed through the estimator's residual path term,
        # and speed is per-demo, so px is the wrong unit for asking "is noise an axis".
        step_px = safe_div(ep.get('speed_px_s'), control_hz)
        ep['noise_per_step'] = safe_div(ep.get('noise_sigma'), step_px)

        # --- geometry invariance checks (numerator and baseline are the same quantity) ---
        ep['ink_len_rel'] = safe_div(ep.get('drawn_path_len'), g.ink_len)
        ep['scale_norm_ink'] = safe_div(g.ink_len, g.ink_scale)


def add_stroke_residuals(stroke_rows: List[Dict], episode_rows: List[Dict],
                         geom: Dict[str, TaskGeometry]) -> None:
    ep_by_key = {(r['task'], r['episode']): r for r in episode_rows}
    for r in stroke_rows:
        g = geom.get(r['task'])
        slot = r.get('stroke_slot', float('nan'))
        ep = ep_by_key.get((r['task'], r['episode']))

        # per-stroke speed with the demo's own global speed divided out. Removes both the
        # geometry and the per-demo speed draw, leaving only WITHIN-episode speed
        # modulation -- the thing "per-part speed" would be if it were an axis here.
        r['speed_rel_episode'] = safe_div(r.get('speed_px_s'),
                                          ep.get('speed_px_s') if ep else float('nan'))

        # turning per unit arc length, made dimensionless by the task's own scale:
        # mean curvature * ink_scale. Baseline is the slot's LENGTH, not its turning, so
        # this is a real residual and its ICC is meaningful.
        if g is not None and np.isfinite(slot) and 0 <= slot < g.n_slots:
            rel_len = safe_div(g.slot_len[int(slot)], g.ink_scale, lo=0.1)
            r['turning_per_len'] = safe_div(r.get('total_turning'), rel_len)
            r['len_rel'] = safe_div(r.get('path_len'), g.slot_len[int(slot)])
            # Fraction of the task's ink this slot carries. Used to gate the turning
            # residual: the bottom percentiles of slot length on this dataset are ~12 px,
            # and normalizing radians by such a slot amplifies noise-driven turning by 5-8x,
            # which showed up as a within-task std of 10 rad on an axis whose raw within-task
            # std is 1.75. That is a division artifact, not a strategy signal.
            r['slot_len_frac'] = safe_div(g.slot_len[int(slot)], g.ink_len)
        else:
            r['turning_per_len'] = float('nan')
            r['len_rel'] = float('nan')
            r['slot_len_frac'] = float('nan')


def curved_and_substantial(row: Dict) -> bool:
    """`curved_only`, plus a slot that carries enough ink for turning to be normalizable."""
    f = row.get('slot_len_frac', float('nan'))
    return curved_only(row) and bool(np.isfinite(f) and f >= 0.10)


# residual axes: (feature, label, raw counterpart, row_filter)
# The filter is applied to BOTH the residual and its raw counterpart, so the dICC compares
# the same population of rows rather than crediting the residual with a filtering effect.
EPISODE_RESIDUALS: List[Tuple[str, str, Optional[str], object]] = [
    ('transit_vs_mean_order', 'routing cost / mean-order cost', 'transit_path_len', None),
    ('transit_excess_norm', 'routing cost, best->mean normalized', 'transit_path_len', None),
    ('approach_vs_mean_first', 'approach / mean first-leg distance', 'approach_path_len', None),
    ('time_vs_geometry', 'episode length / geometry-implied length', 'total_steps', None),
    ('move_time_vs_geometry', 'moving steps / geometry-implied length', 'total_steps', None),
    ('speed_rel', 'speed / reference speed', 'speed_px_s', None),
    ('dwell_per_boundary', 'dwell steps per stroke boundary', 'dwell_steps', None),
    ('noise_per_step', 'noise / step displacement', 'noise_sigma', None),
    ('order_tau_canon', 'stroke order (tau vs canonical geometry)', 'order_tau', None),
    ('order_agreement', 'order+direction (path vs demo 0)', None, None),
]

STROKE_RESIDUALS: List[Tuple[str, str, Optional[str], object]] = [
    ('speed_rel_episode', 'stroke speed / episode speed', 'speed_px_s', None),
    ('turning_per_len', 'turning per unit length (curved, >=10% of ink)', 'total_turning',
     curved_and_substantial),
]

# same-quantity ratios: useful as invariance checks, ICC not meaningful
INVARIANCE_CHECKS: List[Tuple[str, str, str]] = [
    ('ink_len_rel', 'episode', 'inked length / task inked length'),
    ('len_rel', 'stroke', 'stroke length / slot length'),
]


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------

def fmt(x, w, p=3):
    return f'{x:>{w}.{p}f}' if isinstance(x, float) and np.isfinite(x) else f'{"-":>{w}}'


def report(episode_rows: List[Dict], stroke_rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    hdr = (f"{'residual axis':<44}{'lvl':<7}{'mean':>9}{'within':>9}{'between':>9}"
           f"{'ratio':>8}{'ICC':>7}{'ICC_raw':>9}{'dICC':>8}")
    print('\n' + hdr)
    print('-' * len(hdr))

    for level, specs, rows, keys in (
            ('ep', EPISODE_RESIDUALS, episode_rows, ('task',)),
            ('str', STROKE_RESIDUALS, stroke_rows, ('task', 'stroke_slot'))):
        for feat, label, raw, row_filter in specs:
            d = decompose(rows, feat, keys, row_filter=row_filter)
            icc_raw = (decompose(rows, raw, keys, row_filter=row_filter)['icc']
                       if raw else float('nan'))
            d.update({'label': label, 'level': level, 'raw_feature': raw or '',
                      'icc_raw': icc_raw,
                      'delta_icc': (d['icc'] - icc_raw) if np.isfinite(icc_raw) else float('nan')})
            out.append(d)
            print(f"{label:<44}{level:<7}{fmt(d['mean'], 9)}{fmt(d['within_std'], 9)}"
                  f"{fmt(d['between_std'], 9)}{fmt(d['ratio'], 8)}{fmt(d['icc'], 7, 2)}"
                  f"{fmt(icc_raw, 9, 2)}{fmt(d['delta_icc'], 8, 2)}")

    print('\nratio = within-task std / between-task std.  ICC = between / (between + within).')
    print('ICC_raw is the same decomposition on the un-residualized axis; dICC is the change.')
    print('dICC < 0 means the geometry was carrying the between-task variance (the axis was')
    print('partly reading off the goal image); dICC > 0 means the geometry was inflating the')
    print('within-task variance and the residual is the cleaner measurement.')

    print('\ngeometry invariance checks (baseline is the same quantity -- ICC meaningless):')
    print(f"  {'check':<40}{'median':>10}{'IQR':>10}{'within-task IQR':>18}")
    for feat, level, label in INVARIANCE_CHECKS:
        rows = episode_rows if level == 'episode' else stroke_rows
        v = np.array([r.get(feat, np.nan) for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        q75, q25 = np.percentile(v, [75, 25])
        print(f'  {label:<40}{np.median(v):>10.4f}{q75 - q25:>10.4f}'
              f'{"(should be ~1.000)":>18}')
    return out


def check_order_labels(episode_rows: List[Dict], labels_csv: str) -> None:
    """
    Cross-check the recovered canonical order against the generator's own order label.

    Not a residual, but the thing that makes the residual believable: if `order_rank_canon`
    recovered from the trajectory is a consistent relabelling of `strategy_order_rank`, the
    whole segmentation -> slot-matching -> order chain is verified against ground truth.
    Consistency, not equality, is what to expect: the generator ranks a permutation of its
    own stroke indices, and the canonical order indexes strokes by upright position, so the
    two labellings differ by a per-task relabelling of the strokes.
    """
    lab = read_csv(labels_csv)
    by_key = {(r['task'], f"episode_{int(r['demo']):04d}"): r for r in lab}
    per_task: Dict[str, List[Tuple[float, float]]] = {}
    for ep in episode_rows:
        r = by_key.get((ep['task'], ep['episode']))
        if r is None or not np.isfinite(ep.get('order_rank_canon', float('nan'))):
            continue
        per_task.setdefault(ep['task'], []).append(
            (float(r['strategy_order_rank']), float(ep['order_rank_canon'])))

    consistent = inconsistent = 0
    for task, pairs in per_task.items():
        # a per-task bijection must exist between the two rank labellings
        fwd: Dict[float, float] = {}
        rev: Dict[float, float] = {}
        ok = True
        for a, b in pairs:
            if fwd.setdefault(a, b) != b or rev.setdefault(b, a) != a:
                ok = False
                break
        consistent += ok
        inconsistent += (not ok)
    total = consistent + inconsistent
    print(f'\norder recovery vs generator labels: {consistent}/{total} tasks consistent '
          f'({100.0 * consistent / max(total, 1):.1f}%)')
    if inconsistent:
        print(f'  {inconsistent} tasks where the recovered order is NOT a bijective relabelling')
        print('  of strategy_order_rank -- segmentation or slot matching failed on those.')


def main():
    import click

    @click.command()
    @click.option('--axes-dir', required=True, help='output directory of a draw_axes.py run')
    @click.option('--labels-csv', default=None,
                  help='CSV of strategy_* labels (task, demo, strategy_order_rank) to verify against')
    @click.option('--control-hz', default=CONTROL_HZ, show_default=True)
    def cli(axes_dir, labels_csv, control_hz):
        episode_rows = read_csv(os.path.join(axes_dir, 'episodes.csv'))
        stroke_rows = read_csv(os.path.join(axes_dir, 'strokes.csv'))
        print(f'{len(episode_rows)} episodes, {len(stroke_rows)} strokes from {axes_dir}')

        geom = build_geometry(episode_rows, stroke_rows)
        n_slots = np.array([g.n_slots for g in geom.values()])
        print(f'{len(geom)} tasks with geometry; stroke-slot counts: '
              f'{dict(zip(*np.unique(n_slots, return_counts=True)))}')

        # reference speed: the dataset's own median, so `speed_rel` is centred at 1 and
        # dimensionless without importing the generator's sampling range.
        sp = np.array([r.get('speed_px_s', np.nan) for r in episode_rows], dtype=np.float64)
        v_ref = float(np.nanmedian(sp))
        print(f'reference speed v_ref = {v_ref:.2f} px/s (dataset median)')

        add_order_axes(episode_rows, stroke_rows, geom)
        add_episode_residuals(episode_rows, geom, v_ref, control_hz)
        add_stroke_residuals(stroke_rows, episode_rows, geom)

        rep = report(episode_rows, stroke_rows)
        if labels_csv:
            check_order_labels(episode_rows, labels_csv)

        out = os.path.join(axes_dir, 'residual_axes_report.csv')
        keys = ['label', 'level', 'feature', 'raw_feature', 'mean', 'within_std', 'between_std',
                'ratio', 'icc', 'icc_raw', 'delta_icc', 'n_groups', 'n_obs']
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(rep)
        print(f'\nwrote {out}')

    cli()


if __name__ == '__main__':
    main()
