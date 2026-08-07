"""
Closed-loop rollout for policies trained on `env/draw_simple` data.

`DrawRunner` hardcodes the PD-controlled `env/draw/draw_env.py::DrawEnv`, so scoring a policy
trained on `scripts/draw_simple/generate_simple_drawings.py` data with it would measure sim2sim
transfer rather than policy quality. This runner swaps in the matching `SimpleDrawEnv` and
otherwise inherits `DrawRunner` wholesale -- goal-image loading, the rollout loop, the seen/unseen
split handling and the reward metrics are all env-agnostic.

Two things it adds:

  * geometry. `SimpleDrawEnv` rasterises its goal from the task's `StrokeGeometry` rather than from
    a target image, and that geometry is NOT in the zarr -- the generator writes it to a sidecar
    next to the per-task store. Hence `geometry_dir`.
  * strategy metrics. The dataset's whole point is that stroke order and pacing vary under a
    bit-identical goal image, so "did it draw the shape" is only half the evaluation. See
    `utils/draw_strategy_metrics.py`.
"""

import os
from typing import Optional

from behavior_prompting.train_network.env.draw_simple.draw_simple_env import SimpleDrawEnv
from behavior_prompting.train_network.env.draw_simple.geometry import StrokeGeometry
from behavior_prompting.train_network.env_runner.draw_runner import DrawRunner
from behavior_prompting.train_network.gym_util.async_vector_env import AsyncVectorEnv
from behavior_prompting.train_network.gym_util.multistep_wrapper import MultiStepWrapper
from behavior_prompting.train_network.gym_util.video_recording_wrapper import VideoRecordingWrapper
from behavior_prompting.train_network.utils import draw_strategy_metrics as strategy_metrics_util
from behavior_prompting.train_network.utils.video_recorder import VideoRecorder


def get_simple_draw_env(shape_meta, n_train, n_test, fps, crf, exec_action_horizon, max_steps,
                        boundary_angle=None, use_async_vector_env=True, render_cache_size=None,
                        overlay_target_drawing=True, **kwargs):
    """
    Build the shared vector env. Mirrors `draw_runner.get_draw_env`; only the base env differs.

    `overlay_target_drawing` maps onto `SimpleDrawEnv.overlay_goal`. The other draw-env overlay
    flags (`overlay_action_cross`, `overlay_reward`) have no counterpart -- this env's HUD is
    unconditional -- so they are absorbed by `**kwargs` rather than plumbed through as dead args.
    """
    n_envs = n_train + n_test
    steps_per_render = 1

    use_prompting = shape_meta['use_prompting']
    prompt_sample_mode = shape_meta['prompt_sample_mode'] if use_prompting else None
    if prompt_sample_mode == 'sequence':
        max_obs_horizon = exec_action_horizon
    else:
        max_obs_horizon = max(attr['horizon'] for attr in shape_meta['obs'].values())

    render_size = shape_meta['image_resolution']

    def env_fn():
        return MultiStepWrapper(
            VideoRecordingWrapper(
                SimpleDrawEnv(
                    boundary_angle=boundary_angle,
                    render_size=render_size,
                    render_cache_size=render_cache_size,
                    overlay_goal=overlay_target_drawing,
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=fps,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=crf,
                    thread_type='AUTO',
                    thread_count=0
                ),
                mode='rgb_array',
                file_path=None,
                steps_per_render=steps_per_render
            ),
            n_obs_steps=max_obs_horizon,
            n_action_steps=exec_action_horizon,
            max_episode_steps=max_steps
        )

    env_fns = [env_fn] * n_envs
    if use_async_vector_env:
        return AsyncVectorEnv(env_fns)
    assert n_envs == 1, 'if use_async_vector_env is False, then n_envs must be 1'
    return env_fns[0]()


class SimpleDrawEnvSetup:
    """
    Per-env configuration for `SimpleDrawEnv`. Picklable -- it is dilled to the vector-env workers.

    The board angle comes from the goal image the policy is being conditioned on, so the goal ink
    and the canvas share an orientation. Randomisation is switched off deliberately: it would
    confound rotation generalisation with task success, which are worth measuring separately.
    """

    def __init__(self, geometry: StrokeGeometry, target_boundary_angle: float):
        self.geometry = geometry
        self.target_boundary_angle = float(target_boundary_angle)

    def __call__(self, base_env: SimpleDrawEnv):
        base_env.set_geometry(self.geometry)
        base_env.boundary_angle = self.target_boundary_angle
        base_env.randomize_boundary_angle = False


class SimpleDrawRunner(DrawRunner):
    def __init__(self, *args, geometry_dir: Optional[str] = None,
                 strategy_coverage_threshold: float = strategy_metrics_util.DEFAULT_COVERAGE_THRESHOLD,
                 **kwargs):
        super().__init__(*args, **kwargs)

        assert geometry_dir is not None, (
            'SimpleDrawRunner needs geometry_dir: SimpleDrawEnv rasterises its goal from the '
            "task's StrokeGeometry, which is not stored in the zarr. Point it at the directory of "
            '*_geometry.npz sidecars written by scripts/draw_simple/generate_simple_drawings.py.')
        assert self.save_canvas_step is None and self.restore_canvas_step is None, (
            'save_canvas_step/restore_canvas_step are not supported by SimpleDrawEnv, which has no '
            'save_canvas_state or skip_next_video_reset.')

        # `vis_task_name` is the task name with spaces replaced and a case suffix appended -- which
        # is exactly the sidecar stem the generator writes, so no separate naming scheme is needed.
        self.geometry_path = os.path.join(geometry_dir, f'{self.vis_task_name}_geometry.npz')
        assert os.path.exists(self.geometry_path), (
            f'no geometry sidecar for task "{self.task_name}" at {self.geometry_path}')
        self.geometry = StrokeGeometry.load(self.geometry_path)
        self.strategy_coverage_threshold = strategy_coverage_threshold

    def _make_env_setup(self, target_drawing, target_boundary_angle, boundary_angle_from_task):
        # target_drawing is ignored: this env renders the goal from geometry, which is the property
        # the whole env exists to provide.
        return SimpleDrawEnvSetup(self.geometry, target_boundary_angle)

    def _extra_log_data(self, env, n_inits: int) -> dict:
        """
        Score which strategy the policy realised, per env split.

        Traces are read straight off the base envs. They cannot come from the returned `info`:
        `MultiStepWrapper` keeps info in a `deque(maxlen=n_obs_steps+1)`, so all but the last
        couple of steps of each action chunk are dropped. `get_attr` reaches the base env because
        `stroke_trace` exists on no wrapper and gym's `Wrapper.__getattr__` forwards it down.
        """
        if self.is_async_vector_env:
            traces = env.call('get_attr', 'stroke_trace')[:n_inits]
        else:
            traces = [env.stroke_trace]

        log_data = {}
        for split, lo, hi in (('train', 0, self.n_train), ('test', self.n_train, n_inits)):
            split_traces = traces[lo:hi]
            if not split_traces:
                continue
            realized = [strategy_metrics_util.recover_strategy(t) for t in split_traces]
            metrics = strategy_metrics_util.strategy_metrics(
                realized,
                n_strokes=self.geometry.n_strokes,
                coverage_threshold=self.strategy_coverage_threshold)
            for name, value in metrics.items():
                log_data[f'{split}/strategy/{self.vis_task_name}/{name}'] = value
        return log_data
