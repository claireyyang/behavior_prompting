"""
Generate the `DrawingDotEnv` dataset: many instances, each demonstrated in all four manners.

    python generate_dot_demos.py -o ../../datasets/draw_dot/dots_train.zarr \
        --num-instances 1000 --repeats-per-strategy 4

Layout of the output
--------------------
One zarr, written directly. `scripts/draw/group_demos.py` is not used: it requires a
`draw_<x>_lower.zarr` filename convention and hardcodes `agent_pos`/`pen_down`/`image` data keys,
neither of which apply here. Building one `ReplayBuffer` in memory and saving once is simpler than
bending either side.

    data/   canvas     (T, S, S, 3) uint8   -- ch0 ink, ch1 dots, ch2 walls, as the env composes it
            pen_pose   (T, 3)      float32  -- absolute, scene frame
            dots       (T, 4 * 2)  float32  -- the task conditioning, flat; constant within an episode
            action     (T, 3)      float32  -- deltas, stored natively
    labels/ pen_start         (T, 3)    float32
            strategy_id       (T,)      float32
            target_dot_index  (T,)      float32

`dots` lives in `data/`, not `labels/`, because it is a policy INPUT -- `SequenceSampler` only
indexes `data` when assembling observation keys. It is stored flat as `(T, 8)` and reshaped to
`(4, 2)` by `DotDataset`, since shape_meta obs entries are flattened by the encoder anyway. The
per-step repetition is redundant but costs 32 bytes a step and keeps the observation path uniform.

`canvas` is stored exactly as the env emits it, in the repo's usual (T, H, W, C) uint8 image
convention, rather than storing the ink plane alone and recomposing the dot/wall channels at load
time. Recomposition would save about a third of the (already tiny) footprint at the cost of a second
implementation of channel composition that could drift out of step with the env's -- and any drift
between training observations and rollout observations is exactly the kind of bug that is invisible
until it quietly ruins a training run. The constant dot channel costs almost nothing under Blosc.

There is no `is_drag` label: pen state is exactly `action[:, 2] <= 0.5` and the action stream is
already stored. `target_dot_index` is kept because it is the ground-truth segmentation into
per-transit spans, which per-transit analysis needs and which is genuinely awkward to recover from
position alone wherever the path crosses itself.
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import zarr

from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.env.draw_dot.draw_dot_env import DrawingDotEnv
from behavior_prompting.train_network.env.draw_dot.layout import (
    MANNER_TO_ID,
    MANNERS,
    N_DOTS,
    sample_layout,
    sample_pen_start,
)
from behavior_prompting.train_network.scripts.draw_dot import strategies as ST

CANVAS_CHUNK_STEPS = 64     # image chunking: keep random access cheap without tiny chunks


def instance_name(base_seed: int, idx: int) -> str:
    """
    One instance = one dot layout. Conceptually still a single task -- see the docs.

    The base seed is part of the name on purpose. Layouts are sampled from `[base_seed, idx]`, so
    two datasets generated with different seeds hold entirely different layouts -- but if the names
    depended only on `idx` they would collide, and `load_env.py` asserts train and eval instance
    names are disjoint. Worse, a collision that slipped through would silently evaluate on instances
    whose names claim they were trained on. Carrying the seed makes that impossible to forget.
    """
    return f'dots_{base_seed:05d}_{idx:05d}'


def rollout_demo(env: DrawingDotEnv, layout, pen_start: np.ndarray,
                 demo: ST.Demo) -> Dict[str, np.ndarray]:
    """
    Replay a demo through the env, recording the observation each action was taken FROM.

    Recording obs-before-step gives the standard (obs_t, action_t) pairing the sampler expects.
    """
    obs = env.reset(layout=layout, pen_start=pen_start)
    canvas, pen, acts = [], [], []
    for a in demo.actions:
        # (3, H, W) float in [0,1] -> (H, W, 3) uint8, the repo's storage convention for images
        canvas.append((np.moveaxis(obs['canvas'], 0, -1) * 255).astype(np.uint8))
        pen.append(env.pen.astype(np.float32).copy())
        acts.append(np.asarray(a, dtype=np.float32))
        obs, _, _, _ = env.step(a)

    T = len(acts)
    return {
        'canvas': np.stack(canvas).astype(np.uint8),
        'pen_pose': np.stack(pen).astype(np.float32),
        'dots': np.repeat(layout.dots.reshape(1, -1).astype(np.float32), T, axis=0),
        'action': np.stack(acts).astype(np.float32),
    }


def generate(output: str, num_instances: int, repeats: int, base_seed: int,
             canvas_size: int, speed: float, noise_std: float,
             strict: bool, verbose: bool, observe_ink: bool = True) -> ReplayBuffer:
    rb = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    # `canvas` is recorded straight off the env, so this flag decides what the stored observations
    # contain. A dataset built one way cannot train a policy rolled out the other way.
    env = DrawingDotEnv(canvas_size=canvas_size, observe_ink=observe_ink)
    n_bad = 0

    for i in range(num_instances):
        rng = np.random.default_rng([base_seed, i])
        layout = sample_layout(rng)
        pen_start = sample_pen_start(rng)
        name = instance_name(base_seed, i)

        for manner in MANNERS:
            for rep in range(repeats):
                demo_rng = np.random.default_rng([base_seed, i, MANNER_TO_ID[manner], rep])
                demo = ST.make_demo(manner, layout, pen_start, speed=speed,
                                    rng=demo_rng, noise_std=noise_std if rep > 0 else 0.0)
                data = rollout_demo(env, layout, pen_start, demo)
                T = len(data['action'])

                # A demo that fails to touch every dot would poison the reference distribution the
                # multimodality measurement is read against, so surface it rather than shipping it.
                cov = env.dot_coverage()
                if cov < 1.0:
                    n_bad += 1
                    msg = (f'{name} {manner} rep{rep}: dot_coverage {cov:.3f} < 1.0 '
                           f'(noise_std={noise_std} may be too large)')
                    if strict:
                        raise ValueError(msg)
                    print(f'  WARNING: {msg}')

                labels = {
                    'pen_start': np.repeat(pen_start.astype(np.float32)[None], T, axis=0),
                    'strategy_id': np.full(T, MANNER_TO_ID[manner], dtype=np.float32),
                    'target_dot_index': demo.target_dot_index.astype(np.float32),
                }
                rb.add_episode(
                    data=data,
                    tasks=[{'name': name, 'start_idx': 0, 'end_idx': T, 'labels': labels}],
                    episode_name=f'{name}_{manner.lower()}_{rep}',
                )

        if verbose and (i + 1) % 50 == 0:
            print(f'  {i + 1}/{num_instances} instances, {rb.n_episodes} episodes, '
                  f'{rb.n_steps} steps')

    print(f'Writing {rb.n_episodes} episodes / {rb.n_steps} steps to {output}')
    rb.save_to_path(output,
                    chunks={'canvas': (CANVAS_CHUNK_STEPS, canvas_size, canvas_size, 3)},
                    compressors={'canvas': 'disk'})
    if n_bad:
        print(f'WARNING: {n_bad} demos did not reach full coverage')
    return rb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--num-instances', type=int, default=1000)
    p.add_argument('--repeats-per-strategy', type=int, default=4,
                   help='rep 0 is noiseless; later reps get action noise')
    p.add_argument('--base-seed', type=int, default=0)
    p.add_argument('--canvas-size', type=int, default=96)
    p.add_argument('--speed', type=float, default=ST.DEFAULT_SPEED)
    p.add_argument('--noise-std', type=float, default=0.004,
                   help='waypoint jitter in scene units; must stay well under the dot radius')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--no-strict', dest='strict', action='store_false',
                   help='warn instead of failing when a demo misses a dot')
    p.add_argument('--no-observe-ink', dest='observe_ink', action='store_false',
                   help='observation channel 0 becomes visited-dot discs instead of the accumulated '
                        'ink, removing the action history from the policy input')
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args()

    if os.path.exists(a.output):
        if not a.overwrite:
            print(f'{a.output} exists; pass --overwrite to replace it')
            return 1
        import shutil
        shutil.rmtree(a.output)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)

    print(f'Generating {a.num_instances} instances x {len(MANNERS)} manners x '
          f'{a.repeats_per_strategy} repeats = '
          f'{a.num_instances * len(MANNERS) * a.repeats_per_strategy} episodes')
    print(f'observation channel 0: {"ink (action history)" if a.observe_ink else "visited dots"}')
    generate(a.output, a.num_instances, a.repeats_per_strategy, a.base_seed,
             a.canvas_size, a.speed, a.noise_std, a.strict, a.verbose,
             observe_ink=a.observe_ink)
    return 0


if __name__ == '__main__':
    sys.exit(main())
