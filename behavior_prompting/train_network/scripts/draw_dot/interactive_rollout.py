"""
Watch a trained `DrawingDotEnv` policy draw, one episode at a time, in a live window.

This is the qualitative companion to the batch multimodality run (`train.py +rollout=draw_dot`,
see `docs/drawanything_dots.md`). That job reports `manner_entropy` over 40 rollouts per instance;
this one lets you actually see what those rollouts look like, and prints a running manner histogram
so the distribution is legible while it accumulates.

It deliberately does NOT go through `RolloutPolicyWorkspace`/`DrawDotRunner`:

  * **Config comes from the checkpoint, not from Hydra.** The payload carries the full training cfg,
    so there is no way for a model-shaping override to drift out of sync -- which is the failure mode
    `draw_dot_policy_dunet.yaml` warns about for the workspace path. No Accelerator, no wandb.
  * **The raw env is stepped directly, without `MultiStepWrapper`.** That wrapper runs all
    `exec_action_horizon` inner steps inside one `step()` call with no hook in between, so a live
    view would freeze for the whole chunk and then teleport, and a keypress would only be sampled
    once per chunk. Stepping the raw env renders every step and reads keys every step. It also lets
    `reset(layout=..., pen_start=...)` be called directly -- `pending_pen_start` exists only because
    the wrapper's `reset()` takes no arguments.
  * **`render(mode='rgb_array')` plus our own `imshow`.** `DrawingDotEnv._show` hardcodes
    `cv2.waitKey(1)` and returns nothing, so it cannot carry keyboard control and would race with
    ours. Composing the window here also buys the HUD panel and upscaling, and needs no env change.

⚠️ `--fix-initial-state` defaults to ON here, the opposite of `config/task/draw_dot.yaml`. Strategy
is defined as the variation the policy shows at FIXED conditioning; a resampled pen start would
conflate the policy's own multimodality with its sensitivity to initial conditions. The task config
leaves it off because that config is shaped for success-rate runs, where breadth is what matters.

Usage
-----
    cd behavior_prompting/train_network
    python scripts/draw_dot/interactive_rollout.py \
        --checkpoint runs/.../checkpoints/'epoch=0040-eval_test_mean_score=0.924.ckpt' \
        --n-instances 5 --episodes-per-instance 8

Keys: `n` next instance · `r` replay this instance with fresh sampling noise · space pause ·
`q`/Esc quit.
"""

import argparse
import itertools
import pathlib
import sys
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.env.draw_dot.draw_dot_env import DrawingDotEnv
from behavior_prompting.train_network.env.draw_dot.instances import (
    list_instance_names,
    load_instance,
)
from behavior_prompting.train_network.env.draw_dot.layout import (
    MANNERS,
    DotLayout,
    sample_pen_start,
)
from behavior_prompting.train_network.gym_util.multistep_wrapper import stack_last_n_obs
from behavior_prompting.train_network.model.common.base_policy import BasePolicy
from behavior_prompting.train_network.utils import dot_strategy_metrics as DM

REPO_TRAIN_NETWORK = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CKPT = REPO_TRAIN_NETWORK / (
    'runs/2026.08.06/15.39.26_dots_v1_policy_draw_dot_diffusion_unet/checkpoints/latest.ckpt')

# What `LiveViewer.show` reports back to the rollout loop.
CMD_GO, CMD_PAUSE, CMD_NEXT, CMD_REPLAY, CMD_QUIT = 'go', 'pause', 'next', 'replay', 'quit'
_KEYMAP = {ord('q'): CMD_QUIT, 27: CMD_QUIT, ord('n'): CMD_NEXT,
           ord('r'): CMD_REPLAY, ord(' '): CMD_PAUSE}


# ---------------------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------------------

def load_policy(ckpt_path: str, device: torch.device) -> Tuple[BasePolicy, DictConfig]:
    """
    Rebuild the policy from the config stored inside the checkpoint.

    `state_dicts` holds only `model` and `optimizer` -- there is no separate `ema_model` entry
    because `TrainPolicyWorkspace` copies the EMA weights into `self.model` before every save. So
    `state_dicts['model']` already *is* the EMA policy, and it carries the normalizer with it.
    Loading is strict on purpose: a key mismatch means the config and the weights disagree, and a
    rollout under `strict=False` would be quietly meaningless rather than loudly broken.
    """
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']

    policy: BasePolicy = hydra.utils.instantiate(cfg.model)
    policy.load_state_dict(payload['state_dicts']['model'])
    policy.eval()
    policy.to(device)

    epoch = dill.loads(payload['pickles']['epoch']) if 'epoch' in payload.get('pickles', {}) else '?'
    print(f'Loaded {ckpt_path}\n  epoch {epoch}  model {cfg.model._target_.split(".")[-1]}  '
          f'device {device}')
    return policy, cfg


def build_obs_batch(obs_hist: Sequence[dict], obs_horizons: Dict[str, int],
                    n_obs_steps: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Stack the observation history into the batched, per-key-horizon form the encoder asserts on.

    Two separate things happen here. `stack_last_n_obs` left-pads by repeating the oldest frame,
    which is what makes the very first step (history of length 1) well defined -- it is imported
    rather than reimplemented so this matches `MultiStepWrapper` exactly. Then each key is trimmed
    to *its own* declared horizon: `dots` is horizon 1 while `canvas`/`pen_pose` are 2, and
    `TimmObsEncoder` checks the exact shape. Same trim as `DrawDotRunner._slice_obs`.
    """
    out = {}
    for key, horizon in obs_horizons.items():
        stacked = stack_last_n_obs([o[key] for o in obs_hist], n_obs_steps)
        arr = np.ascontiguousarray(stacked[None, -horizon:])
        out[key] = torch.from_numpy(arr).to(device)
    return out


# ---------------------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------------------

class LiveViewer:
    """The cv2 window, plus the keyboard. Disabled (`enabled=False`) it is an inert no-op."""

    WINDOW = 'draw_dot policy rollout'

    def __init__(self, scale: int = 2, delay_ms: int = 100, enabled: bool = True):
        self.scale, self.delay_ms, self.enabled = int(scale), max(1, int(delay_ms)), enabled
        self._opened = False

    def show(self, env: DrawingDotEnv, hud_lines: Sequence[str],
             delay_ms: Optional[int] = None) -> str:
        if not self.enabled:
            return CMD_GO
        # `mode='rgb_array'` composes the frame without touching the env's own window.
        frame = env.render(mode='rgb_array')
        img = np.ascontiguousarray(frame[:, :, ::-1])          # _compose_render returns RGB
        if self.scale != 1:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_NEAREST)

        panel = np.full((22 * len(hud_lines) + 12, img.shape[1], 3), 255, np.uint8)
        for i, line in enumerate(hud_lines):
            cv2.putText(panel, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 0), 1, cv2.LINE_AA)

        if not self._opened:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
            self._opened = True
        cv2.imshow(self.WINDOW, np.vstack([img, panel]))
        key = cv2.waitKey(self.delay_ms if delay_ms is None else max(1, delay_ms)) & 0xFF
        return _KEYMAP.get(key, CMD_GO)

    def wait_while_paused(self, env: DrawingDotEnv, hud_lines: Sequence[str]) -> str:
        """Spin on the window until the user resumes or asks to leave the episode."""
        hud = list(hud_lines) + ['[PAUSED - space to resume]']
        while True:
            cmd = self.show(env, hud, delay_ms=50)
            if cmd in (CMD_PAUSE, CMD_NEXT, CMD_REPLAY, CMD_QUIT):
                return CMD_GO if cmd == CMD_PAUSE else cmd

    def close(self):
        if self._opened:
            cv2.destroyWindow(self.WINDOW)
            self._opened = False


# ---------------------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------------------

def run_episode(policy: BasePolicy, env: DrawingDotEnv, layout: DotLayout,
                pen_start: np.ndarray, obs_horizons: Dict[str, int], exec_horizon: int,
                max_steps: int, device: torch.device, viewer: LiveViewer,
                hud_lines: Sequence[str]) -> Tuple[str, Optional[DM.RolloutResult]]:
    """
    One episode. Returns `('done', result)`, or `(command, None)` if the user cut it short.

    A truncated episode is deliberately not scored: its terminal ink reflects where the user pressed
    a key, not what the policy would have drawn.
    """
    obs = env.reset(layout=layout, pen_start=pen_start)
    policy.reset(action_exec_horizon=exec_horizon)

    n_obs_steps = max(obs_horizons.values())
    hist = deque([obs], maxlen=n_obs_steps)
    steps = 0

    # `DrawingDotEnv.step` always returns done=False, and there is no MultiStepWrapper truncation
    # here, so this step budget is the only thing that ends the episode.
    while steps < max_steps:
        obs_dict = build_obs_batch(hist, obs_horizons, n_obs_steps, device)
        with torch.inference_mode():
            action = policy.predict_action(obs_dict)['action']       # (1, action_horizon, 3)
        chunk = action[0, :exec_horizon].detach().to('cpu').numpy()

        for a in chunk:
            if steps >= max_steps:      # inside the chunk, else the budget overshoots by up to 7
                break
            obs, _, _, _ = env.step(a)
            hist.append(obs)
            steps += 1

            hud = list(hud_lines) + [f'step {steps}/{max_steps}']
            cmd = viewer.show(env, hud)
            if cmd == CMD_PAUSE:
                cmd = viewer.wait_while_paused(env, hud)
            if cmd in (CMD_QUIT, CMD_NEXT, CMD_REPLAY):
                return cmd, None

    # Score from terminal state, before anything resets it -- see `DrawDotRunner`'s docstring on
    # why this is terminal rather than aggregated over the episode.
    return 'done', DM.evaluate_env(env, layout)


# ---------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------

def format_episode(name: str, ep: int, r: DM.RolloutResult, threshold: float) -> str:
    ious = ' '.join(f'{m}:{r.manner_ious.get(m, float("nan")):.2f}' for m in MANNERS)
    flag = '' if r.is_valid(threshold) else '  (INVALID: below coverage threshold)'
    return (f'{name} ep{ep:02d}  cov {r.dot_coverage:.3f}  prec {r.ink_precision:.3f}  '
            f'strokes {r.n_strokes:2d}  manner {r.manner or "-":<9s}[{ious}]{flag}')


def print_histogram(results: Sequence[DM.RolloutResult], threshold: float) -> None:
    """The running manner distribution -- the whole point of the exercise."""
    if not results:
        return
    d = DM.manner_distribution(results, threshold)
    n_valid = int(d.get('n_valid', 0))
    print(f'  running: {len(results)} eps, {n_valid} valid (cov>={threshold:g}), '
          f'H(manner) {d.get("manner_entropy", float("nan")):.3f}, '
          f'{int(d.get("manner_n_distinct", 0))}/4 modes, '
          f'H(contact) {d.get("contact_entropy", float("nan")):.3f}, '
          f'continuous {d.get("contact_frac_continuous", float("nan")):.2f}')
    if n_valid == 0:
        # manner_distribution omits every manner key when nothing cleared the threshold; saying so
        # is more useful than printing four zero bars that read as "collapsed".
        print('    (no valid rollouts yet -- manner is undefined until the task is solved)')
        return
    for m in MANNERS:
        frac = d.get(f'manner_frac_{m}', 0.0)
        print(f'    {m:<9s} {frac:.2f} {"#" * int(round(frac * 30))}')


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------

def select_instances(replay_buffer: ReplayBuffer, args) -> List[str]:
    names = list_instance_names(replay_buffer)
    if args.instances:
        missing = [n for n in args.instances if n not in set(names)]
        assert not missing, f'instances not in dataset: {missing}'
        return list(args.instances)
    if args.shuffle_instances:
        np.random.default_rng(args.seed).shuffle(names)
    return names[args.instance_start:args.instance_start + args.n_instances]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=str(DEFAULT_CKPT))
    p.add_argument('--eval-dataset', default=None,
                   help='eval zarr; defaults to task.eval_dataset_path from the checkpoint cfg')
    p.add_argument('--instances', nargs='+', default=None, help='explicit instance names')
    p.add_argument('--n-instances', type=int, default=5)
    p.add_argument('--instance-start', type=int, default=0)
    p.add_argument('--shuffle-instances', action='store_true')
    p.add_argument('--episodes-per-instance', type=int, default=8)
    p.add_argument('--random-pen-start', dest='fix_initial_state', action='store_false',
                   help='resample the pen start each episode instead of using the recorded one')
    p.set_defaults(fix_initial_state=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fps', type=float, default=None)
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--exec-action-horizon', type=int, default=None)
    p.add_argument('--render-size', type=int, default=None)
    p.add_argument('--scale', type=int, default=2)
    p.add_argument('--loop', action='store_true', help='cycle the instance list forever')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--coverage-threshold', type=float, default=DM.DEFAULT_COVERAGE_THRESHOLD)
    p.add_argument('--no-window', action='store_true', help='headless; print only')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    policy, cfg = load_policy(args.checkpoint, device)

    obs_horizons = {k: int(v['horizon']) for k, v in cfg.task.shape_meta.obs.items()}
    er = cfg.task.env_runner
    exec_horizon = args.exec_action_horizon or int(er.exec_action_horizon)
    max_steps = args.max_steps or int(er.max_steps)
    fps = args.fps or float(er.fps)
    render_size = args.render_size or int(er.render_size)
    zarr_path = args.eval_dataset or cfg.task.eval_dataset_path
    assert zarr_path is not None, 'no eval dataset in the checkpoint cfg; pass --eval-dataset'

    replay_buffer = ReplayBuffer.create_from_path(str(zarr_path))
    names = select_instances(replay_buffer, args)
    print(f'{len(names)} instance(s) from {zarr_path}: {", ".join(names)}')
    print(f'{args.episodes_per_instance} episodes each, '
          f'{"fixed" if args.fix_initial_state else "resampled"} pen start, '
          f'{max_steps} steps, exec horizon {exec_horizon}')

    env = DrawingDotEnv(canvas_size=int(er.canvas_size), render_size=render_size)
    viewer = LiveViewer(scale=args.scale, delay_ms=round(1000 / fps), enabled=not args.no_window)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    all_results: List[DM.RolloutResult] = []
    quit_requested = False
    try:
        for _ in itertools.count():
            for name in names:
                layout, recorded_pen_start = load_instance(replay_buffer, name)
                pen_start = (recorded_pen_start if args.fix_initial_state
                             else sample_pen_start(rng))
                ep = 0
                while ep < args.episodes_per_instance:
                    hud = [f'{name}  ep {ep + 1}/{args.episodes_per_instance}  '
                           f'(total {len(all_results)})',
                           DM.summarize(all_results, args.coverage_threshold)
                           if all_results else 'no results yet',
                           'n=next  r=replay  space=pause  q=quit']
                    # Sampling noise is drawn fresh inside predict_action on every call, so a
                    # replay at the same conditioning is simply another draw from p(tau | goal).
                    cmd, result = run_episode(policy, env, layout, pen_start, obs_horizons,
                                              exec_horizon, max_steps, device, viewer, hud)
                    if cmd == CMD_QUIT:
                        quit_requested = True
                        break
                    if cmd == CMD_REPLAY:
                        continue
                    if cmd == CMD_NEXT:
                        break
                    all_results.append(result)
                    print(format_episode(name, ep, result, args.coverage_threshold))
                    print_histogram(all_results, args.coverage_threshold)
                    ep += 1
                    if not args.fix_initial_state:
                        pen_start = sample_pen_start(rng)
                if quit_requested:
                    break
            if quit_requested or not args.loop:
                break
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        viewer.close()
        env.close()

    print(f'\n=== {len(all_results)} episodes over {len(names)} instance(s) ===')
    print('  ' + DM.summarize(all_results, args.coverage_threshold))
    print_histogram(all_results, args.coverage_threshold)
    return 0


if __name__ == '__main__':
    sys.exit(main())
