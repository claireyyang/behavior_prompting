"""
Dataset for `DrawingDotEnv`.

Deliberately much smaller than `DrawImageDataset`. That class carries goal-image conditioning,
prompting, receding-obs augmentation and goal-rotation augmentation, and it hardcodes the
`image`/`agent_pos`/`pen_down` schema throughout -- none of which applies here. This task has fixed
conditioning (the dot set is part of the observation), no prompting, and no augmentation: rotating
or cropping would change which manner a trajectory represents, which is the one thing that must stay
labelled correctly.

What is unusual here, and why:

  * **Actions are deltas, stored natively.** `pose_repr_util`'s abs<->relative machinery is
    SE(3)-specific and does not fit a 2D + z pen, so there is no conversion layer at all: the policy
    predicts deltas and the env integrates them.
  * **The action normalizer is symmetric with zero offset.** See
    `get_symmetric_range_normalizer_from_stat` -- for deltas, "hold still" must normalize to exactly
    0 or a diffusion policy's N(0, I) prior is fighting a constant bias.
  * **Action padding is `zeros`, not `repeat_last`** (set in `config/task/draw_dot.yaml`). Repeating
    a final delta commands the pen to keep drifting past the end of the episode.
"""

import copy
from typing import Dict, List, Optional

import numpy as np
import torch

from behavior_prompting.common.pytorch_util import dict_apply
from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.common.sampler import (
    SequenceSampler,
    get_train_mask,
    get_training_split_info_from_train_mask,
)
from behavior_prompting.train_network.dataset.base_dataset import BaseDataset
from behavior_prompting.train_network.model.common.normalize_util import (
    array_to_stats,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
    get_symmetric_range_normalizer_from_stat,
)
from behavior_prompting.train_network.model.common.normalizer import Normalizer
from behavior_prompting.train_network.utils.dataset_util import prepare_only_task_names


class DotDataset(BaseDataset):
    def __init__(self,
                 shape_meta: dict,
                 dataset_path: Optional[str] = None,
                 replay_buffer: Optional[ReplayBuffer] = None,
                 seed: int = 42,
                 val_ratio: float = 0.0,
                 sample_type: str = 'task',
                 max_segments: int = -1,
                 action_padding: bool = True,
                 training_split_info: Optional[Dict[str, bool]] = None,
                 only_task_names: Optional[List[str]] = None,
                 max_tasks: Optional[int] = None):
        assert dataset_path is not None or replay_buffer is not None, \
            'either dataset_path or replay_buffer must be provided'

        if replay_buffer is not None:
            assert dataset_path is None, 'dataset_path and replay_buffer cannot both be provided'
            self.replay_buffer = replay_buffer
        else:
            assert dataset_path.endswith('.zarr'), \
                'dataset_path must be a .zarr folder, not a .zarr.zip file'
            self.replay_buffer = ReplayBuffer.create_from_path(dataset_path)

        assert not shape_meta.get('use_prompting', False), \
            'prompting is not supported for draw_dot; conditioning is the dot set itself'

        only_task_names = prepare_only_task_names(self.replay_buffer, only_task_names,
                                                  max_tasks=max_tasks)
        # `sample_type='task'` splits per INSTANCE: each dot layout occupies the replay buffer's
        # task_names field, so every instance contributes demos to both train and val. Conceptually
        # this is still one task -- see docs/drawanything_dots.md.
        train_mask = get_train_mask(self.replay_buffer, sample_type, val_ratio,
                                    training_split_info, seed)
        buffer_task_names = self.replay_buffer.task_names[:]
        train_mask = train_mask & np.isin(buffer_task_names, only_task_names)

        rgb_keys, lowdim_keys = [], []
        for key, attr in shape_meta['obs'].items():
            if attr.get('type', 'low_dim') == 'rgb' and attr.get('in_replay_buffer', True):
                rgb_keys.append(key)
            elif attr.get('type', 'low_dim') == 'low_dim':
                lowdim_keys.append(key)

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.sample_type = sample_type
        self.only_task_names = only_task_names
        self.ignore_prompt = False

        self.sampler_kwargs = {
            'shape_meta': shape_meta,
            'replay_buffer': self.replay_buffer,
            'action_padding': action_padding,
            'sample_type': sample_type,
            'max_segments': max_segments,
            'seed': seed,
        }
        self.sampler = SequenceSampler(mask=self.train_mask, **self.sampler_kwargs)

    # -- BaseDataset contract -------------------------------------------------------------

    def get_validation_dataset(self) -> 'DotDataset':
        val_set = copy.copy(self)
        new_mask = (~self.train_mask) & np.isin(self.replay_buffer.task_names[:],
                                                self.only_task_names)
        val_set.sampler = SequenceSampler(mask=new_mask, **self.sampler_kwargs)
        val_set.train_mask = new_mask
        return val_set

    def get_normalizer(self, **kwargs) -> Normalizer:
        normalizer = Normalizer()

        # Symmetric and zero-offset, NOT the usual range normalizer: these are deltas, and a
        # zero delta ("hold still") has to land on exactly 0.
        normalizer['action'] = get_symmetric_range_normalizer_from_stat(
            array_to_stats(self.replay_buffer.data['action']))

        # Absolute scene-frame quantities keep the ordinary asymmetric range normalizer.
        normalizer['pen_pose'] = get_range_normalizer_from_stat(
            array_to_stats(self.replay_buffer.data['pen_pose']))
        normalizer['dots'] = get_range_normalizer_from_stat(
            array_to_stats(self.replay_buffer.data['dots']))

        # Images arrive already scaled to [0, 1] by __getitem__.
        normalizer['canvas'] = get_image_identity_normalizer()

        prompt_normalizer = copy.deepcopy(normalizer)
        prompt_normalizer.set_prompt_normalizer(None)
        normalizer.set_prompt_normalizer(prompt_normalizer)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        if 'canvas' in data:
            # (T, H, W, C) uint8 -> (T, C, H, W) float in [0, 1]
            canvas = data['canvas'].astype(np.float32) / 255.0
            data['canvas'] = np.moveaxis(canvas, -1, 1)
        if 'pen_pose' in data:
            data['pen_pose'] = data['pen_pose'].astype(np.float32)

        # The dot set is the task conditioning. It is constant within an episode, so the sampler's
        # per-step label slice is taken at the current step and reshaped to the declared (N, 2).
        if 'dots' in data:
            dots = np.asarray(data['dots'], dtype=np.float32)
            data['dots'] = dots.reshape(dots.shape[0], -1, 2)

        action = data.pop('action', None)
        metadata = data.pop('metadata', {})
        torch_data = {'obs': dict_apply(data, torch.from_numpy), 'metadata': metadata}
        if action is not None:
            torch_data['action'] = torch.from_numpy(action.astype(np.float32))
        return torch_data

    # -- passthroughs ----------------------------------------------------------------------

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.replay_buffer.data['action']))

    def shuffle_data_ordering(self, seed: int):
        pass                      # ordering only matters for prompting, which this task does not use

    def requires_epoch_shuffle(self) -> bool:
        return False

    def is_multi_task(self) -> bool:
        # One task, many instances -- but each instance occupies a task_names slot, so the workspace
        # groups and reports per instance.
        return True

    def get_unique_task_name_to_dataset_indices(self) -> Dict[str, list]:
        return self.sampler.get_unique_task_name_to_dataset_indices()

    def get_training_split_info(self) -> Dict[str, bool]:
        return get_training_split_info_from_train_mask(
            self.replay_buffer, self.sample_type, self.train_mask)

    def set_ignore_prompt(self, ignore_prompt: bool):
        self.ignore_prompt = ignore_prompt

    def get_ignore_prompt(self) -> bool:
        return self.ignore_prompt
