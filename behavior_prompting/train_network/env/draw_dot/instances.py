"""
Recovering a `DrawingDotEnv` instance (dot layout + recorded pen start) from a replay buffer.

Kept out of `layout.py` on purpose: that module owns pure geometry and must not import a
`ReplayBuffer` or know that datasets exist. Kept out of `draw_dot_runner.py` too, so that a caller
who only wants an instance -- the interactive rollout script -- does not drag in `wandb`, `dill` and
`AsyncVectorEnv` through the runner's import.

The index arithmetic below is the reason this is a shared function rather than six duplicated lines:
`data` and `labels` have *separate* end offsets, so the per-episode start of each has to be computed
from its own `*_ends` array against the shared `task_lengths`.
"""

from typing import List, Tuple

import numpy as np

from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.env.draw_dot.layout import DotLayout


def list_instance_names(replay_buffer: ReplayBuffer) -> List[str]:
    """Every distinct instance (dot layout) in the buffer, in a stable sorted order."""
    return [str(n) for n in np.unique(np.asarray(replay_buffer.task_names[:]))]


def load_instance(replay_buffer: ReplayBuffer, task_name: str) -> Tuple[DotLayout, np.ndarray]:
    """
    Recover one instance's dot layout and the pen start that was recorded for it.

    The layout is constant within an instance, so the first episode's row is representative; the
    pen start is not, and this deliberately returns the *recorded* one (what `fix_initial_state`
    pins to) rather than resampling.
    """
    names = np.asarray(replay_buffer.task_names[:])
    idxs = np.where(names == task_name)[0]
    assert len(idxs) > 0, f'no episodes for instance "{task_name}"'
    i = int(idxs[0])
    data_start = int(replay_buffer.task_data_ends[i]) - int(replay_buffer.task_lengths[i])
    label_start = int(replay_buffer.task_labels_ends[i]) - int(replay_buffer.task_lengths[i])
    dots = np.asarray(replay_buffer.data['dots'][data_start]).reshape(-1, 2)
    pen_start = np.asarray(replay_buffer.labels['pen_start'][label_start]).reshape(3)
    return DotLayout(dots=dots.astype(np.float64)), pen_start.astype(np.float64)
