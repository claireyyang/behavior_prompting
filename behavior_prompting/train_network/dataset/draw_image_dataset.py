import os
from typing import Dict, List, Optional, Tuple
import cv2
import torch
import numpy as np
import copy

from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.common.sampler import SequenceSampler, get_train_mask, get_training_split_info_from_train_mask
from behavior_prompting.train_network.model.common.normalizer import Normalizer
from behavior_prompting.train_network.model.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_identity_normalizer,
    array_to_stats,
)
from behavior_prompting.common.pytorch_util import dict_apply
from behavior_prompting.train_network.dataset.base_dataset import BaseDataset
from behavior_prompting.train_network.utils.draw_util import get_target_drawing_image_for_task_idx
from behavior_prompting.train_network.utils.dataset_util import prepare_only_task_names

# Positions (`agent_pos`, `action`) in the draw datasets are in the 512px canvas frame of
# DrawEnv/SimpleDrawEnv (`window_size`), independent of the rendered image resolution.
DRAW_CANVAS_SIZE = 512
DRAW_CANVAS_CENTER = DRAW_CANVAS_SIZE / 2.0
# The range DrawEnv/SimpleDrawEnv sample `boundary_angle` from; the augmentation resamples within
# it so augmented board angles stay on-distribution rather than merely near it.
DRAW_BOUNDARY_ANGLE_LOW = -np.pi / 4
DRAW_BOUNDARY_ANGLE_HIGH = np.pi / 4


class DrawImageDataset(BaseDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: Optional[str]=None,
            replay_buffer: Optional[ReplayBuffer]=None,
            seed=42,
            val_ratio=0.0,
            sample_type='task',
            max_segments=-1,
            action_padding=False,
            only_prompt:bool=False,
            only_goal_image:bool=False,
            training_split_info: Optional[Dict[str, bool]]=None,
            only_task_names: Optional[List[str]]=None,
            max_tasks: Optional[int]=None,
            num_training_demos_per_task: Optional[int]=None,
            receding_obs_augmentation_enabled: Optional[bool]=False,
            receding_obs_augmentation_probability: Optional[float]=0.2,
            receding_obs_augmentation_min_shapes: Optional[int]=1,
            receding_obs_augmentation_max_shapes: Optional[int]=3,
            receding_obs_augmentation_debug: Optional[bool]=False,
            goal_rotation_augmentation_enabled: Optional[bool]=False,
            goal_rotation_augmentation_probability: Optional[float]=0.8,
            goal_rotation_augmentation_low: Optional[float]=DRAW_BOUNDARY_ANGLE_LOW,
            goal_rotation_augmentation_high: Optional[float]=DRAW_BOUNDARY_ANGLE_HIGH
            ):
        
        assert dataset_path is not None or replay_buffer is not None, 'either dataset_path or replay_buffer must be provided'

        if replay_buffer is not None:
            assert dataset_path is None, 'dataset_path and replay_buffer cannot both be provided'
            self.replay_buffer = replay_buffer
        else:
            assert dataset_path.endswith('.zarr'), 'dataset_path must be a .zarr folder, not a .zarr.zip file'
            self.replay_buffer = ReplayBuffer.create_from_path(dataset_path)

        only_task_names = prepare_only_task_names(self.replay_buffer, only_task_names, max_tasks=max_tasks)
        
        train_mask = get_train_mask(self.replay_buffer, sample_type, val_ratio, training_split_info, seed)

        # select only the task names in the replay buffer that are in the only_task_names list if provided
        buffer_task_names = self.replay_buffer.task_names[:]
        buffer_task_names_mask = np.isin(buffer_task_names, only_task_names)
        train_mask = train_mask & buffer_task_names_mask

        # Limit train mask to num_training_demos_per_task if specified
        if num_training_demos_per_task is not None and training_split_info is None:
            for task_name in only_task_names:
                task_indices = np.where((self.replay_buffer.task_names[:] == task_name) & train_mask)[0]
                assert len(task_indices) >= num_training_demos_per_task, f"Not enough training demos for task {task_name}"
                if len(task_indices) > num_training_demos_per_task:
                    train_mask[task_indices] = False # remove all train demos for this task temporarily
                    rng = np.random.RandomState(seed)
                    selected_indices = rng.choice(task_indices, size=num_training_demos_per_task, replace=False)
                    train_mask[selected_indices] = True

        # Extract keys from shape_meta
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb" and attr.get('in_replay_buffer', True):
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)
        
        # Check if goal image key is present
        self.use_prompting = shape_meta['use_prompting']
        self.use_goal_image = 'goal_image' in obs_shape_meta and not self.use_prompting

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.sample_type = sample_type
        self.max_segments = max_segments
        self.action_padding = action_padding
        self.only_prompt = only_prompt
        self.only_task_names = only_task_names
        self.only_goal_image = only_goal_image
        self.receding_obs_augmentation_enabled = receding_obs_augmentation_enabled
        self.receding_obs_augmentation_probability = receding_obs_augmentation_probability
        self.receding_obs_augmentation_min_shapes = receding_obs_augmentation_min_shapes
        self.receding_obs_augmentation_max_shapes = receding_obs_augmentation_max_shapes
        self.receding_obs_augmentation_debug = receding_obs_augmentation_debug
        self.goal_rotation_augmentation_enabled = goal_rotation_augmentation_enabled
        self.goal_rotation_augmentation_probability = goal_rotation_augmentation_probability
        self.goal_rotation_augmentation_low = goal_rotation_augmentation_low
        self.goal_rotation_augmentation_high = goal_rotation_augmentation_high
        if goal_rotation_augmentation_enabled:
            assert not self.use_prompting, (
                'goal_rotation_augmentation is not implemented for prompting: a prompt block is '
                'drawn from other episodes with their own board angles, so one rotation per sample '
                'is not well defined. Disable one of the two.')
            assert 'boundary_angle' in self.replay_buffer.labels, (
                'goal_rotation_augmentation needs the `boundary_angle` label to know each demo\'s '
                'recorded board angle; this dataset has none.')

        self.sampler_kwargs = {
            'shape_meta': self.shape_meta,
            'replay_buffer': self.replay_buffer,
            'action_padding': self.action_padding,
            'sample_type': self.sample_type,
            'only_prompt': self.only_prompt,
            'only_goal_image': self.only_goal_image,
            'max_segments': self.max_segments,
            'seed': seed
        }

        sampler = SequenceSampler(
            mask=self.train_mask,
            **self.sampler_kwargs
        )
        self.sampler = sampler

    def get_validation_dataset(self):
        val_set = copy.copy(self)

        new_train_mask = ~self.train_mask
        # select only the task names in the replay buffer that are in the only_task_names list if provided
        buffer_task_names = self.replay_buffer.task_names[:]
        buffer_task_names_mask = np.isin(buffer_task_names, self.only_task_names)
        new_train_mask = new_train_mask & buffer_task_names_mask

        val_set.sampler = SequenceSampler(
            mask=new_train_mask,
            **self.sampler_kwargs
        )

        val_set.train_mask = new_train_mask
        val_set.receding_obs_augmentation_enabled = False
        val_set.goal_rotation_augmentation_enabled = False
        return val_set

    def get_normalizer(self, **kwargs) -> Normalizer:
        normalizer = Normalizer()

        # Action normalizer - use range normalization to scale to [-1, 1] like PushT
        stat = array_to_stats(self.replay_buffer.data['action'])
        normalizer['action'] = get_range_normalizer_from_stat(stat)

        # Agent position normalizer - use range normalization
        stat = array_to_stats(self.replay_buffer.data['agent_pos'])
        normalizer['agent_pos'] = get_range_normalizer_from_stat(stat)
        
        # Pen down normalizer - use range normalization to scale to [-1, 1] (will be previously 0 to 1)
        stat = array_to_stats(self.replay_buffer.data['pen_down'])
        normalizer['pen_down'] = get_range_normalizer_from_stat(stat)

        # Image normalizer (0-1 range)
        normalizer['image'] = get_image_identity_normalizer()

        # Goal image normalizer (0-1 range)
        if self.use_goal_image:
            normalizer['goal_image'] = get_image_identity_normalizer()

        prompt_normalizer = copy.deepcopy(normalizer)
        prompt_normalizer.set_prompt_normalizer(None)

        normalizer.set_prompt_normalizer(prompt_normalizer) # use the same normalizer for prompt and receding obs

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        # Must run before prepare_obs_dict: it operates on raw uint8 (T, H, W, C) frames and
        # canvas-frame positions, not on the normalized (T, C, H, W) float tensors.
        if (self.goal_rotation_augmentation_enabled
                and np.random.random() < self.goal_rotation_augmentation_probability):
            self._apply_goal_rotation_augmentation(data)
        
        def prepare_obs_dict(data):
            # Handle image data
            if 'image' in data:
                # Convert from (T, H, W, C) to (T, C, H, W) and normalize to [0,1]
                image = data['image'].astype(np.float32) / 255.0
                image = np.moveaxis(image, -1, 1)  # Move channel dimension
                data['image'] = image

            # Handle agent position
            if 'agent_pos' in data:
                data['agent_pos'] = data['agent_pos'].astype(np.float32)
            
            # Handle pen down state
            if 'pen_down' in data:
                data['pen_down'] = data['pen_down'].astype(np.float32)

        current_obs_present = any(key in data for key in self.rgb_keys + self.lowdim_keys)

        if current_obs_present:
            if self.receding_obs_augmentation_enabled:
                data['image'] = self._overlay_lines_on_image(data['image'])

            prepare_obs_dict(data)
        if 'prompt' in data:
            prepare_obs_dict(data['prompt']['obs'])

        if 'goal_image' in data:
            # data['goal_image'] is (1, H, W, C)
            # Resize while still uint8 HWC. The goal image key may resolve to a label rather than
            # the observation stream (see SequenceSampler._get_goal_image), and `drawing_image` is
            # stored at the full 512px canvas resolution while shape_meta asks for the encoder's
            # input size -- TimmObsEncoder asserts the exact shape, so a mismatch is a hard error.
            # This runs after _apply_goal_rotation_augmentation, so any rotation happens at full
            # resolution and the downsample comes last.
            data['goal_image'] = self._resize_goal_image(np.asarray(data['goal_image']))
            data['goal_image'] = data['goal_image'].astype(np.float32) / 255.0
            data['goal_image'] = np.moveaxis(data['goal_image'], -1, 1) # (1, H, W, C) -> (1, C, H, W)

        # Convert to torch tensors
        # action and metadata
        action = data.pop('action', None)
        metadata = data.pop('metadata', {})

        # convert to torch
        torch_data = {
            "obs": dict_apply(data, torch.from_numpy),
            "metadata": metadata
        }
        if action is not None:
            torch_data['action'] = torch.from_numpy(action.astype(np.float32))
        return torch_data

    def _resize_goal_image(self, goal_image: np.ndarray) -> np.ndarray:
        """
        Downsample a (T, H, W, C) uint8 goal image stack to the resolution shape_meta declares.

        No-op when it already matches, which is the case whenever the goal image comes from the
        observation stream. INTER_AREA because this is always a downsample (512 -> 224 for
        `drawing_image`) and it is the right filter for that direction.
        """
        target_c, target_h, target_w = self.shape_meta['obs']['goal_image']['shape']
        h, w = goal_image.shape[1], goal_image.shape[2]
        if (h, w) == (target_h, target_w):
            return goal_image
        assert goal_image.shape[3] == target_c, (
            f'goal image has {goal_image.shape[3]} channels but shape_meta declares {target_c}')
        return np.stack([
            cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for frame in goal_image
        ])

    # ---------------------------------------------------------------------------------------
    # goal-image rotation augmentation
    # ---------------------------------------------------------------------------------------

    def _apply_goal_rotation_augmentation(self, data: dict) -> None:
        """
        Rotate an entire sample -- observation frames, goal image, `agent_pos` and `action` -- by
        one shared angle about the canvas centre. Modifies `data` in place.

        Why this is valid: the draw envs are rotation-equivariant about the canvas centre. The
        board is a square of side 350 in a 512 canvas, so its corners sit at radius 247 < 256 and
        no rotation can push ink out of frame. Rotating the rendered pixels and the commanded
        positions together therefore yields exactly the episode that would have been recorded with
        a different `boundary_angle` -- so this augments the conditioning image without inventing
        an action label.

        Why it matters here: with a per-task board angle, every demo of a task shares a
        BIT-IDENTICAL goal image, so a goal encoder sees only `num_tasks` distinct inputs, each
        repeated `demos_per_task` times. Repeated identical inputs add no coverage of image space
        and mildly increase memorisation pressure. The older per-demo board angle gave this
        augmentation for free; this restores it while keeping goal images identical on disk.

        The rotation targets an ABSOLUTE angle drawn uniformly from the env's own
        `boundary_angle` range, using the recorded angle to work out the delta. Applying an
        unconstrained delta instead would widen the board-angle distribution past what the env
        ever produces, spending capacity on angles never seen at rollout.

        ⚠️ One residual train/test gap: a rotated goal image is bilinearly resampled, so its
        12px strokes come out slightly soft, whereas at rollout the env renders them crisply
        (measured ink IoU ~0.93 against a natively-rendered image at the same angle -- all of it
        anti-aliasing at the stroke boundary, no offset). That is why the augmentation is applied
        with probability `goal_rotation_augmentation_probability` rather than always: the
        remaining fraction keeps crisply-rendered goal images in the training distribution. To
        remove the gap entirely you would re-render from the task's `_geometry.npz` sidecar at the
        new angle instead of resampling pixels, which is exact but needs the sidecar plumbed in.
        """
        metadata = data.get('metadata', {})
        task_idx = metadata.get('task_idx', None)
        assert task_idx is not None, 'goal_rotation_augmentation needs metadata["task_idx"]'

        # One shared TARGET angle, but a separate delta per source. The sampler draws
        # `goal_image` from a randomly chosen *other episode of the same task*
        # (`metadata['goal_image_task_idx']`), which carries its own recorded board angle -- so
        # rotating the goal image by the rollout episode's delta would leave the two in different
        # frames. Rotating each to the same absolute target is what keeps them consistent.
        target = float(np.random.uniform(self.goal_rotation_augmentation_low,
                                         self.goal_rotation_augmentation_high))

        rollout_delta = target - self._recorded_boundary_angle(int(task_idx))
        if rollout_delta != 0.0:
            if 'image' in data:
                data['image'] = self._rotate_frames(np.asarray(data['image']), rollout_delta)
            for key in ('agent_pos', 'action'):
                if key in data:
                    data[key] = self._rotate_positions(np.asarray(data[key]), rollout_delta)

        if 'goal_image' in data:
            goal_task_idx = metadata.get('goal_image_task_idx', task_idx)
            goal_delta = target - self._recorded_boundary_angle(int(goal_task_idx))
            if goal_delta != 0.0:
                data['goal_image'] = self._rotate_frames(np.asarray(data['goal_image']), goal_delta)

    def _recorded_boundary_angle(self, task_idx: int) -> float:
        """The board angle a task segment was recorded at. Constant within a segment."""
        label_end = int(np.asarray(self.replay_buffer.task_labels_ends)[task_idx])
        return float(np.asarray(
            self.replay_buffer.labels['boundary_angle'][label_end - 1]).ravel()[0])

    @staticmethod
    def _rotate_positions(positions: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate the leading two components of `positions` (..., >=2) about the canvas centre.

        Trailing components are carried through untouched, which is what makes this correct for
        both `agent_pos` (x, y) and `action` (x, y, pen) -- the pen state is not a coordinate.
        """
        out = positions.astype(np.float32, copy=True)
        x = out[..., 0] - DRAW_CANVAS_CENTER
        y = out[..., 1] - DRAW_CANVAS_CENTER
        c, s = np.cos(angle), np.sin(angle)
        out[..., 0] = x * c - y * s + DRAW_CANVAS_CENTER
        out[..., 1] = x * s + y * c + DRAW_CANVAS_CENTER
        return out

    @staticmethod
    def _rotate_frames(frames: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate a (T, H, W, C) uint8 stack by `angle` about the image centre, filling exposed
        corners with white so they match the canvas background.

        The centre is (W/2 - 0.5, H/2 - 0.5) -- canvas coordinate 256.0 lies on the boundary
        between pixels 255 and 256, which after the resize to W maps to index W/2 - 0.5, so this
        is the pixel-index image of the centre the position rotation uses. Verified: pi rotates
        to `img[::-1, ::-1]` bit-for-bit and pi/2 to `np.rot90(img, k=3)` bit-for-bit.
        """
        h, w = frames.shape[1], frames.shape[2]
        matrix = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), -np.degrees(angle), 1.0)
        return np.stack([
            cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            for frame in frames
        ])

    def _overlay_lines_on_image(self, image: np.ndarray) -> np.ndarray:
        """
        Takes image of shape (T, H, W, C) 0-255 uint8 and returns a new image with same shape/dtype but with random lines overlayed on top of the image.
        This is to simulate the case where the agent has made a mistake on the drawing, but we still want to keep drawing correctly despite a mistake in the view.
        The same random lines are overlayed across all timesteps to make the error show up consistently in the history of the observation.
        """
        if np.random.random() < self.receding_obs_augmentation_probability:
            assert type(image) == np.ndarray, f'image must be a numpy array, got {type(image)}' # make sure it's numpy array instead of being directly linked to the dataset so we don't accidentally modify the dataset
            num_shapes = np.random.randint(self.receding_obs_augmentation_min_shapes, self.receding_obs_augmentation_max_shapes + 1)
            for _ in range(num_shapes):
                shape_type = np.random.choice(['line', 'oval'])
                if shape_type == 'line':
                    start_x = np.random.randint(0, image.shape[2])
                    start_y = np.random.randint(0, image.shape[1])
                    end_x = np.random.randint(0, image.shape[2])
                    end_y = np.random.randint(0, image.shape[1])
                    for t in range(image.shape[0]):
                        image[t] = cv2.line(image[t], (start_x, start_y), (end_x, end_y), (0, 0, 255), 4)
                elif shape_type == 'oval':
                    center_x = np.random.randint(0, image.shape[2])
                    center_y = np.random.randint(0, image.shape[1])
                    # Sample oval size as proportions of image dimensions
                    prop_x = np.random.uniform(0.1, 0.8)
                    prop_y = np.random.uniform(0.1, 0.8)
                    axes_x = int(image.shape[2] * prop_x / 2)  # Convert to semi-axis
                    axes_y = int(image.shape[1] * prop_y / 2)  # Convert to semi-axis
                    angle = np.random.randint(0, 180)
                    # Sample a portion of the ellipse (arc)
                    start_angle = np.random.randint(0, 360)
                    arc_length = np.random.randint(30, 270)  # Arc length between 30 and 270 degrees
                    end_angle = (start_angle + arc_length) % 360
                    for t in range(image.shape[0]):
                        image[t] = cv2.ellipse(image[t], (center_x, center_y), (axes_x, axes_y), angle, start_angle, end_angle, (0, 0, 255), 4)

            if self.receding_obs_augmentation_debug:
                random_id = np.random.randint(0, 1000000)
                out_dir = f'tmp_augmentation'
                os.makedirs(out_dir, exist_ok=True)
                for t in range(image.shape[0]):
                    # Save image to disk with random identifier and timestep
                    path = f'{out_dir}/tmp_{random_id}_{t}.png'
                    cv2.imwrite(path, image[t, ::, ::, ::-1])
                    print(f'Saved augmented image to {path}')

        return image
    
    def get_target_drawing_image_for_task_idx(self, task_idx: int) -> Tuple[np.ndarray, float]:
        return get_target_drawing_image_for_task_idx(self.replay_buffer, task_idx)

    def shuffle_data_ordering(self, seed: int):
        self.sampler.shuffle_data_ordering(seed)

    def requires_epoch_shuffle(self) -> bool:
        return self.sampler.requires_epoch_shuffle()

    def is_multi_task(self) -> bool:
        return True

    def get_unique_task_name_to_dataset_indices(self) -> Dict[str, list[int]]:
        return self.sampler.get_unique_task_name_to_dataset_indices()

    def get_training_split_info(self) -> Dict[str, bool]:
        return get_training_split_info_from_train_mask(self.replay_buffer, self.sample_type, self.train_mask)

    def set_ignore_prompt(self, ignore_prompt: bool):
        self.sampler.set_ignore_prompt(ignore_prompt)

    def get_ignore_prompt(self) -> bool:
        return self.sampler.get_ignore_prompt()
