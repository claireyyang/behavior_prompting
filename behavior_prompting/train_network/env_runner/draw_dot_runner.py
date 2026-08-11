"""
Closed-loop rollout for `DrawingDotEnv`.

One runner per INSTANCE (dot layout), mirroring how `DrawRunner` works per drawing task, with a
shared vector env reused across instances. `DrawRunner` itself is not subclassed: it hardcodes the
`image`/`agent_pos`/`pen_down` schema, a 3-D absolute action sliced as `action[:, :, :2]` for its
cursor-stall check, and a `draw_` task-name prefix -- none of which apply.

Two things this does differently from `DrawRunner`, both deliberate:

  * **Metrics come from terminal env state, never from reward aggregation.** `DrawRunner` logs
    `np.max(all_rewards[i])` over the episode. Coverage is monotone so max would be harmless, but
    `ink_precision` is not: under `max` a policy that touched every dot and then scribbled would
    score identically to one that stopped cleanly.
  * **`fix_initial_state`** pins the pen start across every env of an instance. Strategy is defined
    as the variation a policy exhibits at FIXED conditioning, so leaving the start randomized would
    conflate the policy's own multimodality with its sensitivity to initial conditions.
"""

import os
import pathlib
from typing import Dict, List, Optional

import dill
import numpy as np
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wv

from behavior_prompting.common.pytorch_util import dict_apply
from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.env.draw_dot.draw_dot_env import DrawingDotEnv
from behavior_prompting.train_network.env.draw_dot.instances import load_instance
from behavior_prompting.train_network.env.draw_dot.layout import (
    DotLayout,
    allowed_ink_mask,
    sample_pen_start,
)
from behavior_prompting.train_network.env_runner.base_runner import BaseRunner
from behavior_prompting.train_network.gym_util.async_vector_env import AsyncVectorEnv
from behavior_prompting.train_network.gym_util.multistep_wrapper import MultiStepWrapper
from behavior_prompting.train_network.gym_util.video_recording_wrapper import VideoRecordingWrapper
from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.train_network.utils import dot_strategy_metrics as DM
from behavior_prompting.train_network.utils.video_recorder import VideoRecorder


def get_dot_env(shape_meta, n_train, n_test, fps, crf, exec_action_horizon, max_steps,
                canvas_size=96, render_size=256, observe_ink=True, use_async_vector_env=True,
                **kwargs):
    """
    Build the shared vector env. Mirrors `draw_runner.get_draw_env`; only the base env differs.

    `observe_ink` must match the dataset the policy was trained on -- it changes what channel 0 of
    the observation contains, not just how it looks. See `DrawingDotEnv._get_obs`.
    """
    n_envs = n_train + n_test
    max_obs_horizon = max(attr['horizon'] for attr in shape_meta['obs'].values())

    def env_fn():
        return MultiStepWrapper(
            VideoRecordingWrapper(
                DrawingDotEnv(canvas_size=canvas_size, render_size=render_size,
                              observe_ink=observe_ink),
                video_recoder=VideoRecorder.create_h264(
                    fps=fps, codec='h264', input_pix_fmt='rgb24', crf=crf,
                    thread_type='AUTO', thread_count=0),
                mode='rgb_array',
                file_path=None,
                steps_per_render=1,
            ),
            n_obs_steps=max_obs_horizon,
            n_action_steps=exec_action_horizon,
            max_episode_steps=max_steps,
        )

    env_fns = [env_fn] * n_envs
    if use_async_vector_env:
        return AsyncVectorEnv(env_fns)
    assert n_envs == 1, 'if use_async_vector_env is False, then n_envs must be 1'
    return env_fns[0]()


class DotEnvSetup:
    """
    Per-env configuration, as a picklable object holding **plain data only**.

    It is `dill.dumps`-ed and shipped to the `AsyncVectorEnv` workers; capturing the runner would
    drag its `ReplayBuffer` (and the zarr behind it) through pickle on every env init.
    """

    def __init__(self, dots: np.ndarray, pen_start: np.ndarray):
        self.dots = np.asarray(dots, dtype=np.float64).copy()
        self.pen_start = np.asarray(pen_start, dtype=np.float64).copy()

    def __call__(self, base_env: DrawingDotEnv):
        base_env.set_layout(DotLayout(dots=self.dots))
        base_env.pending_pen_start = self.pen_start


class DrawDotRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 env,
                 replay_buffer: ReplayBuffer,
                 task_name: str,
                 shape_meta: dict,
                 is_eval_dataset: bool,
                 n_train: int = 8,
                 n_train_vis: int = 1,
                 train_start_seed: int = 0,
                 n_test: int = 16,
                 n_test_vis: int = 1,
                 test_start_seed: int = 10000,
                 max_steps: int = 200,
                 fps: int = 10,
                 crf: int = 22,
                 tqdm_interval_sec: float = 1.0,
                 exec_action_horizon: int = 8,
                 canvas_size: int = 96,           # used by get_dot_env
                 render_size: int = 256,          # used by get_dot_env
                 fix_initial_state: bool = False,
                 strategy_coverage_threshold: float = DM.DEFAULT_COVERAGE_THRESHOLD,
                 **kwargs):
        super().__init__(output_dir)
        assert exec_action_horizon <= shape_meta['action']['horizon'], \
            'cannot execute more steps than the policy predicts'

        self.env = env
        self.is_async_vector_env = isinstance(env, AsyncVectorEnv)
        self.replay_buffer = replay_buffer
        self.task_name = task_name
        self.vis_task_name = task_name.replace(' ', '_')
        self.shape_meta = shape_meta
        self.is_eval_dataset = is_eval_dataset
        self.n_train, self.n_test = n_train, n_test
        self.n_train_vis, self.n_test_vis = n_train_vis, n_test_vis
        self.train_start_seed, self.test_start_seed = train_start_seed, test_start_seed
        self.n_envs = n_train + n_test
        self.max_steps = max_steps
        self.fps, self.crf = fps, crf
        self.tqdm_interval_sec = tqdm_interval_sec
        self.exec_action_horizon = exec_action_horizon
        self.fix_initial_state = fix_initial_state
        self.strategy_coverage_threshold = strategy_coverage_threshold

        self.obs_horizons = {k: attr['horizon'] for k, attr in shape_meta['obs'].items()}
        self.layout, self.recorded_pen_start = self._load_instance()
        self.allowed = allowed_ink_mask(self.layout, canvas_size)

    def _load_instance(self):
        """Recover this instance's dot layout and recorded pen start from the replay buffer."""
        return load_instance(self.replay_buffer, self.task_name)

    def _slice_obs(self, obs: dict) -> dict:
        """
        Take the last `horizon` frames of each key.

        `MultiStepWrapper` stacks every observation to the same `n_obs_steps` (the max over keys),
        but `dots` is declared with horizon 1 while `canvas`/`pen_pose` use 2. `TimmObsEncoder`
        asserts the exact declared shape, so the extra frame has to come off here.
        """
        return {k: v[:, -self.obs_horizons[k]:] for k, v in obs.items() if k in self.obs_horizons}

    def run(self, policy: BasePolicy, enable_expensive_vis: bool = True) -> Dict:
        print(f'\n=== DrawDotRunner: instance "{self.task_name}" ===')
        device, env = policy.device, self.env
        n_envs = self.n_envs

        # -- per-env setup ---------------------------------------------------------------
        env_seeds, env_prefixs, init_dills = [], [], []
        for split, count, vis, start_seed in (('train', self.n_train, self.n_train_vis,
                                               self.train_start_seed),
                                              ('test', self.n_test, self.n_test_vis,
                                               self.test_start_seed)):
            for i in range(count):
                seed = start_seed + i
                if self.fix_initial_state:
                    pen_start = self.recorded_pen_start
                else:
                    pen_start = sample_pen_start(np.random.default_rng(seed))
                setup = DotEnvSetup(self.layout.dots, pen_start)
                enable_render = i < vis
                output_dir, vis_name = self.output_dir, self.vis_task_name

                def init_fn(e, seed=seed, enable_render=enable_render, setup=setup,
                            split=split, output_dir=output_dir, vis_name=vis_name):
                    assert isinstance(e.env, VideoRecordingWrapper)
                    e.env.video_recoder.stop()
                    e.env.file_path = None
                    if enable_render:
                        fn = pathlib.Path(output_dir).joinpath(
                            'media', f'{split}_{vis_name}_{wv.util.generate_id()}.mp4')
                        fn.parent.mkdir(parents=True, exist_ok=True)
                        e.env.file_path = str(fn)
                    setup(e.env.env)
                    assert isinstance(e, MultiStepWrapper)
                    e.seed(seed)

                env_seeds.append(seed)
                env_prefixs.append(f'{split}/{self.vis_task_name}_')
                init_dills.append(dill.dumps(init_fn))

        if self.is_async_vector_env:
            env.call_each('run_dill_function', args_list=[(x,) for x in init_dills])
        else:
            dill.loads(init_dills[0])(env)

        # -- rollout ----------------------------------------------------------------------
        obs = env.reset()
        policy.reset(action_exec_horizon=self.exec_action_horizon)
        pbar = tqdm.tqdm(total=self.max_steps, desc=f'Eval "{self.task_name}"',
                         leave=False, mininterval=self.tqdm_interval_sec)
        done, steps = False, 0
        while not done:
            obs_dict = dict_apply(self._slice_obs(dict(obs)),
                                  lambda x: torch.from_numpy(x).to(device=device))
            with torch.inference_mode():
                action = policy.predict_action(obs_dict)['action']
            action = action.detach().to('cpu').numpy()[:, :self.exec_action_horizon]
            obs, _, done, _ = env.step(action)
            done = np.all(done)
            steps += action.shape[1]
            pbar.update(action.shape[1])
            if steps >= self.max_steps:
                done = True
        pbar.close()

        # -- score from TERMINAL state, before the reset wipes it --------------------------
        if self.is_async_vector_env:
            inks = env.call('get_attr', 'ink')[:n_envs]
            visited = env.call('get_attr', 'dots_visited')[:n_envs]
            strokes = env.call('n_strokes')[:n_envs]
            videos = env.render()[:n_envs]
        else:
            inks, visited, strokes = [env.ink], [env.dots_visited], [env.n_strokes()]
            videos = [env.render()]

        results = [DM.evaluate_state(inks[i], visited[i], strokes[i], self.layout, self.allowed)
                   for i in range(n_envs)]

        _ = env.reset()
        policy.reset(action_exec_horizon=self.exec_action_horizon)

        # -- log ---------------------------------------------------------------------------
        log_data: Dict = {}
        for split, lo, hi in (('train', 0, self.n_train), ('test', self.n_train, n_envs)):
            split_results = results[lo:hi]
            if not split_results:
                continue
            # `mean_score` is dot_coverage: the headline success number, and what
            # checkpoint.topk.monitor_key watches via mean_scores/{split}/all.
            coverages = [r.dot_coverage for r in split_results]
            log_data[f'{split}/rewards/{self.vis_task_name}/mean_score'] = float(np.mean(coverages))
            for i, r in enumerate(split_results):
                log_data[f'{split}/rewards/{self.vis_task_name}/seed_{env_seeds[lo + i]}'] = \
                    r.dot_coverage
            for name, value in DM.manner_distribution(
                    split_results, self.strategy_coverage_threshold).items():
                log_data[f'{split}/strategy/{self.vis_task_name}/{name}'] = value
            print(f'  {split}: {DM.summarize(split_results, self.strategy_coverage_threshold)}')

        for i, path in enumerate(videos):
            if path is not None:
                prefix = env_prefixs[i].replace('train/', 'train/videos/').replace(
                    'test/', 'test/videos/')
                log_data[prefix + f'seed_{env_seeds[i]}'] = wandb.Video(path, format='mp4')

        return log_data

    def close(self):
        pass
