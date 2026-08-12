"""
Export scripted demos of all four manners as `collect_demo.py`-schema `.npz` files.

Purpose: the separability gate for the motion-axis space (`fluency_steering`'s
`analysis/draw_dot_separation.py`). Before a human demonstration is worth collecting, the four
manners must be distinguishable in the axes a steering signal would be built from -- and that is a
property of the *manner geometry*, testable without a human. Scripted demos are the controlled
corpus for it: constant `DEFAULT_SPEED` by construction, so path shape and contact structure alone
carry any separation, and pacing cannot confound the result.

Files are written per (layout, manner) as `demo_<layout_idx>_<MANNER>.npz` with the canonical-stream
fields of the `collect_demo.py` schema (`xy_path`, `z_path`, `pen_down_z`, `manner`, ...). There is
deliberately no live stream -- a script has no human pacing -- so downstream ingestion must use
`--stream canonical`. The layout index in the filename is the grouping key for
leave-one-layout-out evaluation.

    python scripts/draw_dot/export_scripted_tracks.py --out demos/draw_dot_scripted --n-layouts 25
"""

import argparse
import pathlib
import sys

import numpy as np

from behavior_prompting.train_network.env.draw_dot.draw_dot_env import PEN_DOWN_Z
from behavior_prompting.train_network.env.draw_dot.layout import sample_layout, sample_pen_start
from behavior_prompting.train_network.scripts.draw_dot import strategies as ST


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='demos/draw_dot_scripted')
    p.add_argument('--n-layouts', type=int, default=25)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--speed', type=float, default=ST.DEFAULT_SPEED)
    a = p.parse_args(argv)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    n = 0
    for li in range(a.n_layouts):
        layout = sample_layout(rng)
        pen_start = sample_pen_start(rng)
        for manner, demo in ST.make_all(layout, pen_start, speed=a.speed).items():
            np.savez_compressed(
                out / f'demo_{li:04d}_{manner}.npz',
                xy_path=demo.xy_path.astype(np.float32),
                z_path=demo.z_path.astype(np.float32),
                action=demo.actions.astype(np.float32),
                target_dot_index=demo.target_dot_index.astype(np.int64),
                pen_start=np.asarray(demo.pen_start, dtype=np.float32),
                dots=layout.dots.astype(np.float32),
                dot_order=np.asarray(layout.order, dtype=np.int64),
                manner=np.asarray(manner, dtype='U16'),
                speed=np.float32(a.speed),
                pen_down_z=np.float32(PEN_DOWN_Z),
            )
            n += 1
    print(f'wrote {n} scripted demos ({a.n_layouts} layouts x {n // a.n_layouts} manners) to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
