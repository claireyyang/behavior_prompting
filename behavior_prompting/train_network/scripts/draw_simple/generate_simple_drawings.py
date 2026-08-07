"""
Generate drawing demonstrations in `SimpleDrawEnv`, where every demo of a task renders a
BIT-IDENTICAL goal image regardless of the strategy used to draw it.

Output layout is identical to `scripts/draw/procedural_generate_drawings.py`, so the datasets
drop straight into the existing diffusion-policy training path:

    <output>/
      draw_simple_001_lower.zarr/          one zarr per task, one episode per demo
      draw_simple_001_lower_geometry.npz   sidecar: the task's canonical polyline (see below)
      tmp_draw_simple_001_lower_draw_video.mp4   only with --visualize
      ...

Per-step data keys match the old generator exactly: `image`, `agent_pos`, `pen_down`, `action`.
Labels are `boundary_angle`, `drawing_image`, `goal_image`, the per-step `stroke_index` and
`profile_id`, and the `strategy_*` scalars (whose names match `scripts/draw/strategy_variation.py`,
so existing analysis scripts still work).

What varies within a task
-------------------------
Two axes by default, both of which leave the goal image bit-identical:

  1. STROKE ORDER -- which stroke is drawn first, second, ...
  2. MOTION DISTRIBUTION IN TIME along each stroke -- constant pace (`linear`), easing in
     (`ease_in`, slow start / fast finish), or easing out (`ease_out`, fast start / slow finish),
     sampled independently per stroke.

A profile only redistributes WHEN the pen is at each point along a fixed path; it never changes
which path. Since every profile is monotone and pinned at both endpoints, each stroke is covered
exactly, so all of this is ink-invariant (see `strategy.py`).

Speed is held constant by default (`--speed-min == --speed-max`) so pacing variation comes from
the profile rather than from an overall speed knob, and direction reversal is off. The episode's
whole-trajectory profile is then a joint consequence of the order and the per-stroke profiles;
`stroke_index` labels which stroke each step belongs to so that can be segmented and extracted.

Also implemented and verified invariant, just off by default: `--vary-direction`,
`--stroke-speed-ratio`, and a non-degenerate `--speed-min/--speed-max`.

The geometry sidecar has no equivalent in the old pipeline and is not part of the zarr: it is
what you need to roll a trained policy back out in this env, since the env's ink rule is defined
against the task's canonical polyline. Load it with `StrokeGeometry.load(path)`.

What this generator does NOT need
--------------------------------
No settle loop, no settle tolerance, no curvature-derived speed cap, no per-part minimum sample
count, no end-hold steps, and no post-hoc "realized speed" relabelling. Those all existed in the
old generator to bound how much PD lag leaked into the render. Here the leak is structurally
absent, so each demo simply asserts the thing outright:

    coverage == 1.0  and  get_drawing_image() == get_goal_image()  (exact array equality)

Example
-------
    python generate_simple_drawings.py -o /tmp/simple_ds --num-tasks 20 --demos-per-task 10 \
        --n-strokes 3 --profile-families linear,ease_in,ease_out,ease_in_out --visualize
"""

import hashlib
import multiprocessing
import os
import shutil
import time
from typing import Dict, List, Optional, Sequence

import click
import numpy as np
import zarr
from tqdm import tqdm

from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.common.replay_buffer_util import print_replay_buffer_draw
from behavior_prompting.train_network.env.draw_simple.draw_simple_env import (
    BOUNDARY_ANGLE_HIGH,
    BOUNDARY_ANGLE_LOW,
    SimpleDrawEnv,
)
from behavior_prompting.train_network.env.draw_simple.geometry import StrokeGeometry
import shapes as SH
import strategy as ST

TASK_PREFIX = 'draw simple_'
LABEL_CHUNK_STEPS = 16   # steps per chunk for the image-valued labels; see save_to_path below


# ---------------------------------------------------------------------------------------
# output paths -- kept byte-compatible with scripts/draw/demo_draw.py, reimplemented here so
# this generator has no dependency on the old draw pipeline
# ---------------------------------------------------------------------------------------

def task_name_to_dataset_name(task_name: str) -> str:
    safe = task_name.replace(' ', '_')
    ending = safe[len('draw_'):]
    if ending.upper() == ending:
        safe += '_upper'
    elif ending.lower() == ending:
        safe += '_lower'
    else:
        raise ValueError(f'Task name {task_name!r} is not valid')
    return safe + '.zarr'


def get_task_dataset_path(output_dir: str, task_name: str) -> str:
    return os.path.join(output_dir, task_name_to_dataset_name(task_name))


def get_geometry_path(output_dir: str, task_name: str) -> str:
    stem = task_name_to_dataset_name(task_name)[:-len('.zarr')]
    return os.path.join(output_dir, stem + '_geometry.npz')


def get_video_path(output_dir: str, task_name: str) -> str:
    stem = task_name_to_dataset_name(task_name)[:-len('.zarr')]
    return os.path.join(output_dir, f'tmp_{stem}_draw_video.mp4')


def task_name_for(task_idx: int, num_tasks: int) -> str:
    return TASK_PREFIX + str(task_idx + 1).zfill(len(str(num_tasks)))


def seed_from_ints(parts: Sequence) -> int:
    """Stable across processes and runs, unlike hash() -- which python salts per interpreter."""
    digest = hashlib.sha256(','.join(map(str, parts)).encode()).hexdigest()
    return int(digest, 16) % (2 ** 32)


# ---------------------------------------------------------------------------------------
# one demo
# ---------------------------------------------------------------------------------------

def rollout_demo(env: SimpleDrawEnv, geom: StrokeGeometry, strategy: ST.Strategy,
                 boundary_angle: float, rng: np.random.Generator,
                 control_hz: int, noise_std: float, noise_bounds: float,
                 approach_dwell_max: int, final_hold_steps: int,
                 record: bool = True) -> Dict[str, List]:
    """
    Execute one demo and return the recorded per-step arrays.

    Each stroke is reached with a pen-up approach that lands EXACTLY on the stroke's start (the
    kinematic step makes that trivial: commanding a target within `step_cap` sets the cursor to
    it bit-for-bit). No settling is needed because there is no velocity to settle.
    """
    obs = env.reset(boundary_angle=boundary_angle)
    rec: Dict[str, List] = {'image': [], 'agent_pos': [], 'pen_down': [], 'action': [],
                            'drawing_image': [], 'stroke_index': [], 'profile_id': []}

    def execute(action: np.ndarray, stroke_index: int = -1, profile_id: int = -1) -> None:
        nonlocal obs
        if record:
            rec['stroke_index'].append(np.float32(stroke_index))
            rec['profile_id'].append(np.float32(profile_id))
            rec['image'].append((np.transpose(obs['image'], (1, 2, 0)) * 255).astype(np.uint8))
            rec['agent_pos'].append(np.asarray(obs['agent_pos'], dtype=np.float32))
            rec['pen_down'].append(np.asarray(obs['pen_down'], dtype=np.float32))
            rec['action'].append(np.asarray(action, dtype=np.float32))
            rec['drawing_image'].append(env.get_drawing_image())
        obs, _, _, _ = env.step(action)

    def approach(target: np.ndarray, dwell: int) -> None:
        max_steps = int(np.ceil(2 * env.window_size / env.step_cap)) + 4
        for _ in range(max_steps):
            if float(np.linalg.norm(env.cursor - target)) <= 1e-9:
                break
            execute(np.array([target[0], target[1], 0.0], dtype=np.float32))
        else:
            raise ValueError(f'pen-up approach failed to reach {target} in {max_steps} steps')
        for _ in range(max(1, dwell)):
            execute(np.array([target[0], target[1], 0.0], dtype=np.float32))

    stroke_infos: List[dict] = []
    for stroke_idx, reverse, profile, speed in ST.realized_strokes(strategy):
        actions, info = ST.build_stroke_actions(
            geom, stroke_idx, reverse, speed, control_hz, profile, boundary_angle,
            rng=rng, noise_std=noise_std, noise_bounds=noise_bounds,
            max_step_len=env.step_cap)
        stroke_infos.append(info)
        approach(actions[0, :2].astype(np.float64),
                 int(rng.integers(1, approach_dwell_max + 1)))
        pid = ST.PROFILE_FAMILIES.index(profile[0])
        for action in actions:
            execute(action, stroke_index=stroke_idx, profile_id=pid)

    # Pen-up hold so the completed drawing appears in the recorded `drawing_image` labels --
    # every step records the canvas as it was BEFORE that step, so without this the last stroke's
    # final edge would never be observed. Pen up, so it inks nothing.
    hold = np.array([env.cursor[0], env.cursor[1], 0.0], dtype=np.float32)
    for _ in range(max(1, final_hold_steps)):
        execute(hold)

    # The point of the whole exercise, asserted rather than measured.
    coverage = env.coverage()
    if coverage < 1.0:
        raise ValueError(f'demo covered only {coverage:.6f} of the canonical edge set; '
                         f'the strategy layer should always cover it exactly')
    drawing, goal = env.get_drawing_image(), env.get_goal_image()
    if not np.array_equal(drawing, goal):
        n_diff = int((drawing != goal).any(axis=2).sum())
        raise ValueError(f'completed drawing differs from the canonical goal image in '
                         f'{n_diff} pixels; ink is no longer strategy-invariant')

    rec['_info'] = stroke_infos
    return rec


# ---------------------------------------------------------------------------------------
# one task (runs in its own process)
# ---------------------------------------------------------------------------------------

def generate_single_task(a: dict):
    task_idx, num_tasks = a['task_idx'], a['num_tasks']
    output, verbose = a['output'], a['verbose']
    task_name = task_name_for(task_idx, num_tasks)

    # The task's geometry is sampled once and shared by every demo -- it is the "what".
    geom_rng = np.random.default_rng(seed_from_ints([a['base_seed'], task_idx, num_tasks, 'geom']))
    geom = SH.sample_task_geometry(
        geom_rng,
        n_strokes=a['n_strokes'],
        parts_per_stroke=(a['min_parts_per_stroke'], a['max_parts_per_stroke']),
        board_length=a['board_length'], margin=a['margin'], ds=a['ds'],
        min_part_len=a['min_part_len'], max_part_len=a['max_part_len'],
        arc_radius_range=(a['arc_radius_min'], a['arc_radius_max']),
        allowed=a['primitives'], min_stroke_separation=a['min_stroke_separation'])

    if verbose:
        print(f'\n--- {task_name}: {geom!r} ---')

    env = SimpleDrawEnv(geometry=geom, boundary_angle=0.0,
                        render_size=a['render_size'], control_hz=a['control_hz'],
                        max_speed=a['max_speed'], pen_radius=a['pen_radius'],
                        snap_tol=a['snap_tol'], board_length=a['board_length'])
    assert a['noise_bounds'] < env.snap_tol, (
        f'noise_bounds ({a["noise_bounds"]}) must stay under the env snap_tol ({env.snap_tol}) '
        f'or noise can push the cursor off the canonical path and lose ink')

    replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())

    # The board angle is a TASK property by default. It has to be, for demos of a task to render
    # the IDENTICAL goal image: ink is invariant to the strategy but not to the board rotation, so
    # resampling the angle per demo (as the old generator does) makes each demo's goal image a
    # rotation of the others rather than the same array. --vary-angle-per-demo restores that
    # rotation augmentation, and gives up the identity.
    task_angle = float(np.random.default_rng(
        seed_from_ints([a['base_seed'], task_idx, num_tasks, 'angle'])
    ).uniform(BOUNDARY_ANGLE_LOW, BOUNDARY_ANGLE_HIGH))

    first_goal_image = None
    for demo_idx in range(a['demos_per_task']):
        rng = np.random.default_rng(seed_from_ints([a['base_seed'], task_idx, num_tasks, demo_idx]))
        boundary_angle = (float(rng.uniform(BOUNDARY_ANGLE_LOW, BOUNDARY_ANGLE_HIGH))
                          if a['vary_angle_per_demo'] else task_angle)
        base_speed = float(rng.uniform(a['speed_min'], a['speed_max']))

        strategy = ST.sample_strategy(
            rng, geom.n_strokes, base_speed,
            vary_order=a['vary_order'], vary_direction=a['vary_direction'],
            reverse_probability=a['reverse_probability'],
            profile_families=a['profile_families'], profile_strength=a['profile_strength'],
            profile_per_stroke=a['profile_per_stroke'],
            stroke_speed_ratio=a['stroke_speed_ratio'])

        rec = rollout_demo(env, geom, strategy, boundary_angle, rng,
                           control_hz=a['control_hz'], noise_std=a['noise_std'],
                           noise_bounds=a['noise_bounds'],
                           approach_dwell_max=a['approach_dwell_max'],
                           final_hold_steps=a['final_hold_steps'])

        # Close the loop at the DATASET level: rollout_demo already asserted this demo's drawing
        # equals its own goal image, so matching the first demo's goal image proves every demo of
        # this task drew a bit-identical picture.
        goal_image = env.get_goal_image()
        if first_goal_image is None:
            first_goal_image = goal_image
        elif not a['vary_angle_per_demo'] and not np.array_equal(goal_image, first_goal_image):
            n_diff = int((goal_image != first_goal_image).any(axis=2).sum())
            raise ValueError(f'{task_name} demo {demo_idx}: goal image differs from demo 0 in '
                             f'{n_diff} pixels despite a fixed board angle')

        n_steps = len(rec['image'])
        episode_data = {'image': np.stack(rec['image']),
                        'agent_pos': np.stack(rec['agent_pos']),
                        'pen_down': np.stack(rec['pen_down']),
                        'action': np.stack(rec['action'])}

        labels: Dict[str, np.ndarray] = {
            'boundary_angle': np.full(n_steps, boundary_angle, dtype=np.float32),
            'drawing_image': _maybe_resize(np.stack(rec['drawing_image']), a['label_image_size']),
            # Which stroke each step belongs to (-1 while the pen is up). Ground truth for
            # segmenting an episode by stroke; recoverable from pen_down + geometry, but only
            # awkwardly, and a downstream profile extractor wants it directly.
            'stroke_index': np.asarray(rec['stroke_index'], dtype=np.float32),
            # Per-step pacing ground truth: index into ST.PROFILE_FAMILIES for the stroke being
            # drawn, -1 while the pen is up. The scalar `strategy_profile_id` only reports the
            # FIRST drawn stroke's family, which does not describe a demo when profiles are
            # sampled per stroke (the default) -- use this instead.
            'profile_id': np.asarray(rec['profile_id'], dtype=np.float32),
        }
        if a['save_goal_image']:
            goal = _maybe_resize(goal_image[None], a['label_image_size'])
            labels['goal_image'] = np.repeat(goal, n_steps, axis=0)
        # Label what the demo actually did, not only what was asked for: a profile whose peak
        # step exceeds the env's step cap gets refined, so its realized cadence is slower than
        # its requested `strategy_base_speed`. `strategy_n_strokes_capped` says how often that
        # happened, so a downstream model is never conditioned on a speed that did not occur.
        infos = rec['_info']
        realized = {
            'strategy_realized_speed': float(np.mean([i['realized_speed'] for i in infos])),
            'strategy_max_step_px': float(max(i['max_step_px'] for i in infos)),
            'strategy_n_strokes_capped': float(sum(i['speed_capped'] for i in infos)),
        }
        for key, value in {**strategy.summary_labels(), **realized}.items():
            labels[key] = np.full(n_steps, value, dtype=np.float32)

        replay_buffer.add_episode(
            data=episode_data,
            tasks=[{'name': task_name, 'start_idx': 0, 'end_idx': n_steps, 'labels': labels}],
            episode_name=f'episode_{demo_idx:04d}')

        if verbose:
            print(f'  demo {demo_idx + 1}/{a["demos_per_task"]}: {n_steps:4d} steps, '
                  f'{strategy!r}, angle {boundary_angle:+.3f} rad')

    # Compress and chunk the image-valued labels explicitly. ReplayBuffer.add_episode creates
    # label arrays with compressor=None and one chunk spanning the episode, and save_to_store then
    # inherits both -- so `drawing_image`/`goal_image` land on disk UNCOMPRESSED despite being
    # mostly-white canvases (goal_image is even the same frame repeated every step). Measured on a
    # 6-demo task at 128px: 36.8 MB -> 0.2 MB per label, and 216 GB -> 21 GB projected over
    # 2000 tasks x 8 demos. The 16-step chunking also matters for training: a whole-episode chunk
    # forces a full-episode decompress to fetch one random sample.
    size = a['label_image_size'] or 512
    image_labels = ['drawing_image'] + (['goal_image'] if a['save_goal_image'] else [])
    output_path = get_task_dataset_path(output, task_name)
    replay_buffer.save_to_path(
        output_path,
        chunks={k: (LABEL_CHUNK_STEPS, size, size, 3) for k in image_labels},
        compressors={k: 'disk' for k in image_labels})
    geom.save(get_geometry_path(output, task_name))

    if a['visualize']:
        print_replay_buffer_draw(output_path, replay_buffer=replay_buffer, print_summary=False,
                                 vis_video=True, enable_print=verbose)

    return task_name, a['demos_per_task']


def _maybe_resize(images: np.ndarray, size: Optional[int]) -> np.ndarray:
    """Optionally downsample (T, H, W, 3) uint8 label images to save disk."""
    if size is None or size == images.shape[1]:
        return images
    import cv2
    return np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
                     for im in images])


# ---------------------------------------------------------------------------------------
# resume / summary
# ---------------------------------------------------------------------------------------

def find_missing_tasks(output: str, num_tasks: int, demos_per_task: int,
                       check_videos: bool) -> List[int]:
    """
    Which task indices still need generating. Partial or corrupt tasks are deleted so they get
    regenerated cleanly rather than appended to.
    """
    missing = []
    for task_idx in range(num_tasks):
        task_name = task_name_for(task_idx, num_tasks)
        path = get_task_dataset_path(output, task_name)
        complete = False
        if os.path.exists(path):
            try:
                rb = ReplayBuffer.create_from_path(path, mode='r')
                complete = rb.n_episodes >= demos_per_task
            except Exception:
                complete = False
            if complete and not os.path.exists(get_geometry_path(output, task_name)):
                complete = False
            if complete and check_videos and not os.path.exists(get_video_path(output, task_name)):
                complete = False
            if not complete:
                shutil.rmtree(path, ignore_errors=True)
        if not complete:
            missing.append(task_idx)
    return missing


def print_task_summary(output: str, num_tasks: int) -> None:
    header = f'======== Task Summary for {output} ========'
    print(f'\n{header}')
    total = 0
    found = 0
    for task_idx in range(num_tasks):
        task_name = task_name_for(task_idx, num_tasks)
        path = get_task_dataset_path(output, task_name)
        if not os.path.exists(path):
            continue
        n = ReplayBuffer.create_from_path(path, mode='r').n_episodes
        total += n
        found += 1
        print(f'  "{task_name}": {n} demos')
    print(f'\nTotal: {found} tasks, {total} demos')
    print('=' * len(header) + '\n')


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

@click.command()
@click.option('-o', '--output', required=True, help='Output directory (one .zarr per task).')
@click.option('--num-tasks', default=10, type=int)
@click.option('--demos-per-task', default=5, type=int)
@click.option('--overwrite', is_flag=True, help='Delete the output directory first.')
@click.option('--max-workers', default=None, type=int, help='Worker processes (default: auto).')
@click.option('--base-seed', default=0, type=int)
@click.option('--visualize', is_flag=True, help='Render one video per task.')
@click.option('--verbose', is_flag=True)
# --- geometry (the "what") ---
@click.option('--n-strokes', default=3, type=int, help='Strokes per task; the unit of ordering.')
@click.option('--min-parts-per-stroke', default=1, type=int)
@click.option('--max-parts-per-stroke', default=2, type=int)
@click.option('--primitives', default='line,arc,bezier',
              help=f'Comma-separated subset of {",".join(SH.PRIMITIVES)}.')
@click.option('--board-length', default=350.0, type=float)
@click.option('--margin', default=30.0, type=float)
@click.option('--ds', default=1.0, type=float,
              help='Canonical vertex spacing in px. Sets the granularity of the edge set.')
@click.option('--min-part-len', default=60.0, type=float)
@click.option('--max-part-len', default=220.0, type=float)
@click.option('--arc-radius-min', default=30.0, type=float)
@click.option('--arc-radius-max', default=80.0, type=float)
@click.option('--min-stroke-separation', default=0.0, type=float,
              help='Reject strokes closer than this to an already-placed one. 0 allows crossing '
                   'strokes; the env resolves crossings by stroke continuity. Raise it to also '
                   'rule out two strokes passing within --snap-tol of a pen-down point.')
# --- strategy (the "how") ---
@click.option('--speed-min', default=200.0, type=float,
              help='Base speed range in px/s. Equal min/max (the default) keeps speed out of '
                   'the picture so stroke order is the only thing that varies.')
@click.option('--speed-max', default=200.0, type=float)
@click.option('--vary-order/--no-vary-order', default=True)
@click.option('--vary-angle-per-demo', is_flag=True,
              help='Resample the board rotation per demo instead of per task. Off by default: a '
                   'per-demo angle makes each goal image a ROTATION of the others rather than the '
                   'same array, which is the one thing that breaks within-task goal identity.')
@click.option('--vary-direction/--no-vary-direction', default=False,
              help='Off by default: stroke ORDER is the intended axis. Direction reversal is '
                   'also goal-image-invariant here, so turning it on is safe when wanted.')
@click.option('--reverse-probability', default=0.5, type=float)
@click.option('--profile-families', default='linear,ease_in,ease_out',
              help=f'Comma-separated subset of {",".join(ST.PROFILE_FAMILIES)}. How the motion '
                   f'is distributed in time along a stroke: constant pace, easing in, or easing '
                   f'out. All of them cover the stroke exactly, so the goal image is unchanged.')
@click.option('--profile-strength', default=0.7, type=float)
@click.option('--profile-per-stroke/--profile-per-demo', default=True,
              help='Per-stroke (default) gives each stroke of a demo its own pacing; per-demo '
                   'makes the profile one demo-level cadence signature.')
@click.option('--stroke-speed-ratio', default=1.0, type=float,
              help='>1 varies speed per stroke, log-uniformly in [1/r, r].')
@click.option('--noise-std', default=0.5, type=float,
              help='Positional action noise in px. Ink-neutral while under --snap-tol.')
@click.option('--noise-bounds', default=1.5, type=float, help='Noise clip; must be < --snap-tol.')
@click.option('--approach-dwell-max', default=4, type=int)
@click.option('--final-hold-steps', default=4, type=int)
# --- env ---
@click.option('--control-hz', default=10, type=int)
@click.option('--max-speed', default=1200.0, type=float, help='Per-step displacement cap * hz.')
@click.option('--pen-radius', default=6, type=int, help='Half the pen width (6 -> 12px stroke).')
@click.option('--snap-tol', default=3.0, type=float,
              help='Off-path tolerance, pen down. Keep well under --pen-radius.')
@click.option('--render-size', default=224, type=int)
@click.option('--label-image-size', default=512, type=int,
              help='Side length of the drawing_image / goal_image labels. 512 matches the old '
                   'generator; lower it to shrink the dataset on disk.')
@click.option('--save-goal-image/--no-save-goal-image', default=True)
def main(**kw):
    """Generate strategy-invariant drawing demos into a directory of per-task zarr datasets."""
    output = kw['output']
    num_tasks, demos_per_task = kw['num_tasks'], kw['demos_per_task']

    if os.path.exists(output) and kw['overwrite']:
        print(f"Removing existing output directory '{output}' (--overwrite)...")
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)

    tasks_to_generate = find_missing_tasks(output, num_tasks, demos_per_task, kw['visualize'])
    if not tasks_to_generate:
        print('All tasks already complete. Nothing to do.')
        print_task_summary(output, num_tasks)
        return
    if len(tasks_to_generate) < num_tasks:
        print(f'RESUME: {num_tasks - len(tasks_to_generate)} tasks already complete, '
              f'{len(tasks_to_generate)} to generate.')

    primitives = tuple(p.strip() for p in kw['primitives'].split(',') if p.strip())
    profile_families = tuple(p.strip() for p in kw['profile_families'].split(',') if p.strip())
    for p in primitives:
        assert p in SH.PRIMITIVES, f'unknown primitive {p!r}'
    for p in profile_families:
        assert p in ST.PROFILE_FAMILIES, f'unknown profile family {p!r}'

    max_workers = kw['max_workers'] or min(len(tasks_to_generate), multiprocessing.cpu_count())

    def build_args(task_idx: int, verbose: bool) -> dict:
        args = {k: v for k, v in kw.items()
                if k not in ('overwrite', 'max_workers', 'primitives', 'profile_families')}
        args.update({'task_idx': task_idx, 'num_tasks': num_tasks, 'verbose': verbose,
                     'primitives': primitives, 'profile_families': profile_families})
        return args

    print(f'Generating {len(tasks_to_generate)} tasks x {demos_per_task} demos '
          f'({kw["n_strokes"]} strokes/task) with {max_workers} worker(s)...')
    start = time.time()
    total_episodes = 0

    if max_workers == 1:
        for task_idx in tasks_to_generate:
            _, n = generate_single_task(build_args(task_idx, verbose=True))
            total_episodes += n
    else:
        task_args = [build_args(i, verbose=kw['verbose']) for i in tasks_to_generate]
        with multiprocessing.Pool(processes=max_workers) as pool:
            with tqdm(total=len(task_args), desc='Generating tasks', unit='task') as pbar:
                for _, n in pool.imap_unordered(generate_single_task, task_args):
                    total_episodes += n
                    pbar.update(1)

    elapsed = time.time() - start
    print(f'\n{"=" * 60}\nGENERATION COMPLETE\n{"=" * 60}')
    print(f'Episodes generated this run: {total_episodes}')
    print(f'Total time: {elapsed:.1f}s '
          f'({elapsed / max(total_episodes, 1):.2f}s per episode)')
    print_task_summary(output, num_tasks)


if __name__ == '__main__':
    main()
