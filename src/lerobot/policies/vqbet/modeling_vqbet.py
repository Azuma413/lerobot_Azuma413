#!/usr/bin/env python

# Copyright 2024 Seungjae Lee and Yibin Wang and Haritheja Etukuru
# and H. Jin Kim and Nur Muhammad Mahi Shafiullah and Lerrel Pinto
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, get_output_shape, populate_queues
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.vqbet.vqbet_utils import GPT, ResidualVQ
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# ruff: noqa: N806


class VQBeTPolicy(PreTrainedPolicy):
    """
    VQ-BeT Policy as per "Behavior Generation with Latent Actions"
    """

    config_class = VQBeTConfig
    name = "vqbet"

    def __init__(
        self,
        config: VQBeTConfig | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.vqbet = VQBeTModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        vqvae_params = (
            list(self.vqbet.action_head.vqvae_model.encoder.parameters())
            + list(self.vqbet.action_head.vqvae_model.decoder.parameters())
            + list(self.vqbet.action_head.vqvae_model.vq_layer.parameters())
        )
        decay_params, no_decay_params = self.vqbet.policy.configure_parameters()
        decay_params = (
            decay_params
            + list(self.vqbet.rgb_encoder.parameters())
            + list(self.vqbet.state_projector.parameters())
            + list(self.vqbet.rgb_feature_projector.parameters())
            + [self.vqbet.action_token]
            + list(self.vqbet.action_head.map_to_cbet_preds_offset.parameters())
        )
        if hasattr(self.vqbet, "env_state_projector"):
            decay_params = decay_params + list(self.vqbet.env_state_projector.parameters())

        if self.config.sequentially_select:
            decay_params = (
                decay_params
                + list(self.vqbet.action_head.map_to_cbet_preds_primary_bin.parameters())
                + list(self.vqbet.action_head.map_to_cbet_preds_secondary_bin.parameters())
            )
        else:
            decay_params = decay_params + list(self.vqbet.action_head.map_to_cbet_preds_bin.parameters())

        return [
            {
                "params": decay_params,
            },
            {
                "params": vqvae_params,
                "weight_decay": self.config.optimizer_vqvae_weight_decay,
                "lr": self.config.optimizer_vqvae_lr,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

    def reset(self):
        """
        Clear observation and action queues. Should be called on `env.reset()`
        queues are populated during rollout of the policy, they contain the n latest observations and actions
        """
        self._queues = {
            OBS_IMAGES: deque(maxlen=self.config.n_obs_steps),
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.action_chunk_size),
        }
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        # Build batch from queues, handling OBS_IMAGES specially since it contains lists of tensors
        new_batch = {}
        for k in batch:
            if k not in self._queues:
                continue
            queue_list = list(self._queues[k])
            if k == OBS_IMAGES and isinstance(queue_list[0], list):
                # Each element in queue_list is a list of tensors (one per image type)
                # We need to stack each image type across time steps
                num_image_types = len(queue_list[0])
                stacked_images = []
                for img_idx in range(num_image_types):
                    # Collect all tensors for this image type across time steps
                    images_for_type = [queue_list[t][img_idx] for t in range(len(queue_list))]
                    stacked_images.append(torch.stack(images_for_type, dim=1))  # (B, S, C, H, W)
                new_batch[k] = stacked_images
            else:
                new_batch[k] = torch.stack(queue_list, dim=1)
        
        # Copy over non-queue items like image_keys
        if "image_keys" in batch:
            new_batch["image_keys"] = batch["image_keys"]
        
        actions = self.vqbet(new_batch, rollout=True)[:, : self.config.action_chunk_size]
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)
        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original

        # Store image keys for proper backbone routing
        image_keys = list(self.config.image_features.keys())
        batch["image_keys"] = image_keys

        # Stack images but also keep individual images for multi-backbone processing
        batch[OBS_IMAGES] = [batch[key] for key in image_keys]

        self._queues = populate_queues(self._queues, batch)

        if not self.vqbet.action_head.vqvae_model.discretized.item():
            warnings.warn(
                "To evaluate in the environment, your VQ-BeT model should contain a pretrained Residual VQ.",
                stacklevel=1,
            )

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            # since the data in the action queue's dimension is (action_chunk_size, batch_size, action_dim), we transpose the action and fill the queue
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original

        # Store image keys for proper backbone routing
        image_keys = list(self.config.image_features.keys())
        batch["image_keys"] = image_keys

        # Keep images as a list for multi-backbone processing
        batch[OBS_IMAGES] = [batch[key] for key in image_keys]

        # VQ-BeT discretizes action using VQ-VAE before training BeT (please refer to section 3.2 in the VQ-BeT paper https://huggingface.co/papers/2403.03181)
        if not self.vqbet.action_head.vqvae_model.discretized.item():
            # loss: total loss of training RVQ
            # n_different_codes: how many of the total possible VQ codes are being used in single batch (how many of them have at least one encoder embedding as a nearest neighbor). This can be at most `vqvae_n_embed * number of layers of RVQ (=2)`.
            # n_different_combinations: how many different code combinations are being used out of all possible combinations in single batch. This can be at most `vqvae_n_embed ^ number of layers of RVQ (=2)` (hint consider the RVQ as a decision tree).
            loss, n_different_codes, n_different_combinations, recon_l1_error = (
                self.vqbet.action_head.discretize(self.config.n_vqvae_training_steps, batch[ACTION])
            )
            return loss, {
                "n_different_codes": n_different_codes,
                "n_different_combinations": n_different_combinations,
                "recon_l1_error": recon_l1_error,
            }
        # if Residual VQ is already trained, VQ-BeT trains its GPT and bin prediction head / offset prediction head parts.
        _, loss_dict = self.vqbet(batch, rollout=False)
        loss = loss_dict.pop("loss")

        return loss, loss_dict


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://huggingface.co/papers/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class VQBeTModel(nn.Module):
    """VQ-BeT: The underlying neural network for VQ-BeT

    Note: In this code we use the terms `rgb_encoder`, 'policy', `action_head`. The meanings are as follows.
        - The `rgb_encoder` process rgb-style image observations to one-dimensional embedding vectors
        - A `policy` is a minGPT architecture, that takes observation sequences and action query tokens to generate `features`.
        - These `features` pass through the action head, which passes through the code prediction, offset prediction head,
        and finally generates a prediction for the action chunks.

        -------------------------------** legend **-------------------------------
        │   n = n_obs_steps, p = n_action_pred_token, c = action_chunk_size)   │
        │   o_{t} : visual observation at timestep {t}                           │
        │   s_{t} : state observation at timestep {t}                            │
        │   a_{t} : action at timestep {t}                                       │
        │   A_Q : action_query_token                                             │
        --------------------------------------------------------------------------


        Training Phase 1. Discretize action using Residual VQ (for config.n_vqvae_training_steps steps)


        ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
        │                 │            │                 │            │                 │
        │   RVQ encoder   │    ─►      │     Residual    │    ─►      │   RVQ Decoder   │
        │ (a_{t}~a_{t+p}) │            │  Code Quantizer │            │                 │
        │                 │            │                 │            │                 │
        └─────────────────┘            └─────────────────┘            └─────────────────┘

        Training Phase 2.

          timestep {t-n+1}   timestep {t-n+2}                timestep {t}
            ┌─────┴─────┐     ┌─────┴─────┐                 ┌─────┴─────┐

        o_{t-n+1}         o_{t-n+2}           ...         o_{t}
            │                 │                             │
            │ s_{t-n+1}       │ s_{t-n+2}         ...       │   s_{t}           p
            │     │           │     │                       │     │     ┌───────┴───────┐
            │     │    A_Q    │     │    A_Q          ...   │     │    A_Q     ...     A_Q
            │     │     │     │     │     │                 │     │     │               │
        ┌───▼─────▼─────▼─────▼─────▼─────▼─────────────────▼─────▼─────▼───────────────▼───┐
        │                                                                                   │
        │                                       GPT                                         │       =>    policy
        │                                                                                   │
        └───────────────▼─────────────────▼─────────────────────────────▼───────────────▼───┘
                        │                 │                             │               │
                    ┌───┴───┐         ┌───┴───┐                     ┌───┴───┐       ┌───┴───┐
                  code    offset    code    offset                code    offset  code    offset
                    ▼       │         ▼       │                     ▼       │       ▼       │       =>    action_head
               RVQ Decoder  │    RVQ Decoder  │                RVQ Decoder  │  RVQ Decoder  │
                    └── + ──┘         └── + ──┘                     └── + ──┘       └── + ──┘
                        ▼                 ▼                             ▼               ▼
                   action chunk      action chunk                  action chunk     action chunk
                    a_{t-n+1} ~       a_{t-n+2} ~                   a_{t} ~     ...  a_{t+p-1} ~
                     a_{t-n+c}         a_{t-n+c+1}                   a_{t+c-1}        a_{t+p+c-1}

                                                                        ▼
                                                      ONLY this chunk is used in rollout!
    """

    def __init__(self, config: VQBeTConfig):
        super().__init__()
        self.config = config

        self.rgb_encoder = VQBeTRgbEncoder(config)
        # Store image keys for consistent ordering
        self.image_keys = list(config.image_features.keys())
        # Number of output features from encoder (mic images are combined into one)
        self._compute_num_image_features()

        # This action query token is used as a prompt for querying action chunks. Please refer to "A_Q" in the image above.
        # Note: During the forward pass, this token is repeated as many times as needed. The authors also experimented with initializing the necessary number of tokens independently and observed inferior results.
        self.action_token = nn.Parameter(torch.randn(1, 1, self.config.gpt_input_dim))

        # To input state and observation features into GPT layers, we first project the features to fit the shape of input size of GPT.
        self.state_projector = MLP(
            config.robot_state_feature.shape[0], hidden_channels=[self.config.gpt_input_dim]
        )
        if config.env_state_feature:
            self.env_state_projector = MLP(
                config.env_state_feature.shape[0], hidden_channels=[self.config.gpt_input_dim]
            )
        self.rgb_feature_projector = MLP(
            self.rgb_encoder.feature_dim, hidden_channels=[self.config.gpt_input_dim]
        )

        # GPT part of VQ-BeT
        self.policy = GPT(config)
        # bin prediction head / offset prediction head part of VQ-BeT
        self.action_head = VQBeTHead(config)

        # Action tokens for: each observation step, the current action token, and all future action tokens.
        num_tokens = self.config.n_action_pred_token + self.config.n_obs_steps - 1
        self.register_buffer(
            "select_target_actions_indices",
            torch.row_stack([torch.arange(i, i + self.config.action_chunk_size) for i in range(num_tokens)]),
        )

    def _compute_num_image_features(self):
        """Compute the number of image features that will be output by the encoder."""
        mic_keys = self.rgb_encoder.mic_image_keys
        mic_count = sum(1 for k in self.image_keys if k in mic_keys)
        # Mic images are combined into one, so we only count them once
        if mic_count > 0:
            self.num_image_features = len(self.image_keys) - mic_count + 1
        else:
            self.num_image_features = len(self.image_keys)

    def forward(self, batch: dict[str, Tensor], rollout: bool) -> tuple[dict, dict]:
        # Input validation.
        assert set(batch).issuperset({OBS_STATE, OBS_IMAGES})
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Get image keys from batch or use stored keys
        image_keys = batch.get("image_keys", self.image_keys)

        # Handle both list format and stacked tensor format
        if isinstance(batch[OBS_IMAGES], list):
            # New format: list of tensors (B, S, C, H, W)
            images_list = batch[OBS_IMAGES]

            # Process images for each observation step
            all_features = []
            for s in range(n_obs_steps):
                # Extract images for this time step
                step_images = [img[:, s] for img in images_list]  # List of (B, C, H, W)
                # Process through multi-backbone encoder
                step_features = self.rgb_encoder.forward_with_keys(step_images, image_keys)
                all_features.append(step_features)

            # Reorganize: from [n_steps, n_features, (B, D)] to (B, n_steps, n_features, D)
            n_features = len(all_features[0])
            img_features = torch.stack([
                torch.stack([all_features[s][f] for s in range(n_obs_steps)], dim=1)
                for f in range(n_features)
            ], dim=2)  # (B, S, n_features, D)
        else:
            # Legacy format: stacked tensor (B, S, N, C, H, W)
            # Extract image feature (first combine batch and sequence dims).
            img_features = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
            # Separate batch and sequence dims.
            img_features = einops.rearrange(
                img_features, "(b s n) ... -> b s n ...", b=batch_size, s=n_obs_steps, n=len(self.image_keys)
            )

        # Arrange prior and current observation step tokens as shown in the class docstring.
        # First project features to token dimension.
        rgb_tokens = self.rgb_feature_projector(
            img_features
        )  # (batch, obs_step, number of features, projection dims)
        input_tokens = [rgb_tokens[:, :, i] for i in range(rgb_tokens.size(2))]
        input_tokens.append(self.state_projector(batch[OBS_STATE]))  # (batch, obs_step, projection dims)
        if self.config.env_state_feature:
            input_tokens.append(self.env_state_projector(batch[OBS_ENV_STATE]))
        input_tokens.append(einops.repeat(self.action_token, "1 1 d -> b n d", b=batch_size, n=n_obs_steps))
        # Interleave tokens by stacking and rearranging.
        input_tokens = torch.stack(input_tokens, dim=2)
        input_tokens = einops.rearrange(input_tokens, "b n t d -> b (n t) d")

        len_additional_action_token = self.config.n_action_pred_token - 1
        future_action_tokens = self.action_token.repeat(batch_size, len_additional_action_token, 1)

        # add additional action query tokens for predicting future action chunks
        input_tokens = torch.cat([input_tokens, future_action_tokens], dim=1)

        # get action features (pass through GPT)
        features = self.policy(input_tokens)

        # Compute number of input feature types for action index calculation
        # This includes: num_image_features + state + action_token (+ optionally env_state)
        num_feature_types = self.num_image_features + 1  # +1 for state (action token handled separately in index calc)
        if self.config.env_state_feature:
            num_feature_types += 1

        # len(self.config.input_features) is the number of different observation modes.
        # this line gets the index of action prompt tokens.
        historical_act_pred_index = np.arange(0, n_obs_steps) * (num_feature_types + 1) + num_feature_types

        # only extract the output tokens at the position of action query:
        # Behavior Transformer (BeT), and VQ-BeT are both sequence-to-sequence prediction models,
        # mapping sequential observation to sequential action (please refer to section 2.2 in BeT paper https://huggingface.co/papers/2206.11251).
        # Thus, it predicts a historical action sequence, in addition to current and future actions (predicting future actions : optional).
        if len_additional_action_token > 0:
            features = torch.cat(
                [features[:, historical_act_pred_index], features[:, -len_additional_action_token:]], dim=1
            )
        else:
            features = features[:, historical_act_pred_index]
        # pass through action head
        action_head_output = self.action_head(features)
        # if rollout, VQ-BeT don't calculate loss
        if rollout:
            return action_head_output["predicted_action"][:, n_obs_steps - 1, :].reshape(
                batch_size, self.config.action_chunk_size, -1
            )
        # else, it calculate overall loss (bin prediction loss, and offset loss)
        else:
            output = batch[ACTION][:, self.select_target_actions_indices]
            loss = self.action_head.loss_fn(action_head_output, output, reduction="mean")
            return action_head_output, loss


class VQBeTHead(nn.Module):
    def __init__(self, config: VQBeTConfig):
        """
        VQBeTHead takes output of GPT layers, and pass the feature through bin prediction head (`self.map_to_cbet_preds_bin`), and offset prediction head (`self.map_to_cbet_preds_offset`)

        self.map_to_cbet_preds_bin: outputs probability of each code (for each layer).
            The input dimension of `self.map_to_cbet_preds_bin` is same with the output of GPT,
            and the output dimension of `self.map_to_cbet_preds_bin` is `self.vqvae_model.vqvae_num_layers (=fixed as 2) * self.config.vqvae_n_embed`.
            if the agent select the code sequentially, we use self.map_to_cbet_preds_primary_bin and self.map_to_cbet_preds_secondary_bin instead of self._map_to_cbet_preds_bin.

        self.map_to_cbet_preds_offset: output the predicted offsets for all the codes in all the layers.
            The input dimension of ` self.map_to_cbet_preds_offset` is same with the output of GPT,
            and the output dimension of ` self.map_to_cbet_preds_offset` is `self.vqvae_model.vqvae_num_layers (=fixed as 2) * self.config.vqvae_n_embed * config.action_chunk_size * config.action_feature.shape[0]`.
        """

        super().__init__()
        self.config = config
        # init vqvae
        self.vqvae_model = VqVae(config)
        if config.sequentially_select:
            self.map_to_cbet_preds_primary_bin = MLP(
                in_channels=config.gpt_output_dim,
                hidden_channels=[self.config.vqvae_n_embed],
            )
            self.map_to_cbet_preds_secondary_bin = MLP(
                in_channels=config.gpt_output_dim + self.config.vqvae_n_embed,
                hidden_channels=[self.config.vqvae_n_embed],
            )
        else:
            self.map_to_cbet_preds_bin = MLP(
                in_channels=config.gpt_output_dim,
                hidden_channels=[self.vqvae_model.vqvae_num_layers * self.config.vqvae_n_embed],
            )
        self.map_to_cbet_preds_offset = MLP(
            in_channels=config.gpt_output_dim,
            hidden_channels=[
                self.vqvae_model.vqvae_num_layers
                * self.config.vqvae_n_embed
                * config.action_chunk_size
                * config.action_feature.shape[0],
            ],
        )
        # loss
        self._focal_loss_fn = FocalLoss(gamma=2.0)

    def discretize(self, n_vqvae_training_steps, actions):
        # Resize the action sequence data to fit the action chunk size using a sliding window approach.
        actions = torch.cat(
            [
                actions[:, j : j + self.config.action_chunk_size, :]
                for j in range(actions.shape[1] + 1 - self.config.action_chunk_size)
            ],
            dim=0,
        )
        # `actions` is a tensor of shape (new_batch, action_chunk_size, action_dim) where new_batch is the number of possible chunks created from the original sequences using the sliding window.

        loss, metric = self.vqvae_model.vqvae_forward(actions)
        n_different_codes = sum(
            [len(torch.unique(metric[2][:, i])) for i in range(self.vqvae_model.vqvae_num_layers)]
        )
        n_different_combinations = len(torch.unique(metric[2], dim=0))
        recon_l1_error = metric[0].detach().cpu().item()
        self.vqvae_model.optimized_steps += 1
        # if we updated RVQ more than `n_vqvae_training_steps` steps, we freeze the RVQ part.
        if self.vqvae_model.optimized_steps >= n_vqvae_training_steps:
            self.vqvae_model.discretized = torch.tensor(True)
            self.vqvae_model.vq_layer.freeze_codebook = torch.tensor(True)
            print("Finished discretizing action data!")
            self.vqvae_model.eval()
            for param in self.vqvae_model.vq_layer.parameters():
                param.requires_grad = False
        return loss, n_different_codes, n_different_combinations, recon_l1_error

    def forward(self, x, **kwargs) -> dict:
        # N is the batch size, and T is number of action query tokens, which are process through same GPT
        N, T, _ = x.shape
        # we calculate N and T side parallelly. Thus, the dimensions would be
        # (batch size * number of action query tokens, action chunk size, action dimension)
        x = einops.rearrange(x, "N T WA -> (N T) WA")

        # sample offsets
        cbet_offsets = self.map_to_cbet_preds_offset(x)
        cbet_offsets = einops.rearrange(
            cbet_offsets,
            "(NT) (G C WA) -> (NT) G C WA",
            G=self.vqvae_model.vqvae_num_layers,
            C=self.config.vqvae_n_embed,
        )
        # if self.config.sequentially_select is True, bin prediction head first sample the primary code, and then sample secondary code
        if self.config.sequentially_select:
            cbet_primary_logits = self.map_to_cbet_preds_primary_bin(x)

            # select primary bin first
            cbet_primary_probs = torch.softmax(
                cbet_primary_logits / self.config.bet_softmax_temperature, dim=-1
            )
            NT, choices = cbet_primary_probs.shape
            sampled_primary_centers = einops.rearrange(
                torch.multinomial(cbet_primary_probs.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )

            cbet_secondary_logits = self.map_to_cbet_preds_secondary_bin(
                torch.cat(
                    (x, F.one_hot(sampled_primary_centers, num_classes=self.config.vqvae_n_embed)),
                    axis=1,
                )
            )
            cbet_secondary_probs = torch.softmax(
                cbet_secondary_logits / self.config.bet_softmax_temperature, dim=-1
            )
            sampled_secondary_centers = einops.rearrange(
                torch.multinomial(cbet_secondary_probs.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )
            sampled_centers = torch.stack((sampled_primary_centers, sampled_secondary_centers), axis=1)
            cbet_logits = torch.stack([cbet_primary_logits, cbet_secondary_logits], dim=1)
        # if self.config.sequentially_select is False, bin prediction head samples primary and secondary code at once.
        else:
            cbet_logits = self.map_to_cbet_preds_bin(x)
            cbet_logits = einops.rearrange(
                cbet_logits, "(NT) (G C) -> (NT) G C", G=self.vqvae_model.vqvae_num_layers
            )
            cbet_probs = torch.softmax(cbet_logits / self.config.bet_softmax_temperature, dim=-1)
            NT, G, choices = cbet_probs.shape
            sampled_centers = einops.rearrange(
                torch.multinomial(cbet_probs.view(-1, choices), num_samples=1),
                "(NT G) 1 -> NT G",
                NT=NT,
            )

        device = get_device_from_parameters(self)
        indices = (
            torch.arange(NT, device=device).unsqueeze(1),
            torch.arange(self.vqvae_model.vqvae_num_layers, device=device).unsqueeze(0),
            sampled_centers,
        )
        # Use advanced indexing to sample the values (Extract the only offsets corresponding to the sampled codes.)
        sampled_offsets = cbet_offsets[indices]
        # Then, sum the offsets over the RVQ layers to get a net offset for the bin prediction
        sampled_offsets = sampled_offsets.sum(dim=1)
        with torch.no_grad():
            # Get the centroids (= vectors corresponding to the codes) of each layer to pass it through RVQ decoder
            return_decoder_input = self.vqvae_model.get_embeddings_from_code(sampled_centers).clone().detach()
            # pass the centroids through decoder to get actions.
            decoded_action = self.vqvae_model.get_action_from_latent(return_decoder_input).clone().detach()
        # reshaped extracted offset to match with decoded centroids
        sampled_offsets = einops.rearrange(
            sampled_offsets, "NT (W A) -> NT W A", W=self.config.action_chunk_size
        )
        # add offset and decoded centroids
        predicted_action = decoded_action + sampled_offsets
        predicted_action = einops.rearrange(
            predicted_action,
            "(N T) W A -> N T (W A)",
            N=N,
            T=T,
            W=self.config.action_chunk_size,
        )

        return {
            "cbet_logits": cbet_logits,
            "predicted_action": predicted_action,
            "sampled_centers": sampled_centers,
            "decoded_action": decoded_action,
        }

    def loss_fn(self, pred, target, **kwargs):
        """
        for given ground truth action values (target), and prediction (pred) this function calculates the overall loss.

        predicted_action: predicted action chunk (offset + decoded centroids)
        sampled_centers: sampled centroids (code of RVQ)
        decoded_action: decoded action, which is produced by passing sampled_centers through RVQ decoder
        NT: batch size * T
        T: number of action query tokens, which are process through same GPT
        cbet_logits: probability of all codes in each layer
        """
        action_seq = target
        predicted_action = pred["predicted_action"]
        sampled_centers = pred["sampled_centers"]
        decoded_action = pred["decoded_action"]
        NT = predicted_action.shape[0] * predicted_action.shape[1]

        cbet_logits = pred["cbet_logits"]

        predicted_action = einops.rearrange(
            predicted_action, "N T (W A) -> (N T) W A", W=self.config.action_chunk_size
        )

        action_seq = einops.rearrange(action_seq, "N T W A -> (N T) W A")
        # Figure out the loss for the actions.
        # First, we need to find the closest cluster center for each ground truth action.
        with torch.no_grad():
            state_vq, action_bins = self.vqvae_model.get_code(action_seq)  # action_bins: NT, G

        # Now we can compute the loss.

        # offset loss is L1 distance between the predicted action and ground truth action
        offset_loss = F.l1_loss(action_seq, predicted_action)

        # calculate primary code prediction loss
        cbet_loss1 = self._focal_loss_fn(
            cbet_logits[:, 0, :],
            action_bins[:, 0],
        )
        # calculate secondary code prediction loss
        cbet_loss2 = self._focal_loss_fn(
            cbet_logits[:, 1, :],
            action_bins[:, 1],
        )
        # add all the prediction loss
        cbet_loss = (
            cbet_loss1 * self.config.primary_code_loss_weight
            + cbet_loss2 * self.config.secondary_code_loss_weight
        )

        equal_primary_code_rate = torch.sum((action_bins[:, 0] == sampled_centers[:, 0]).int()) / (NT)
        equal_secondary_code_rate = torch.sum((action_bins[:, 1] == sampled_centers[:, 1]).int()) / (NT)

        action_mse_error = torch.mean((action_seq - predicted_action) ** 2)
        vq_action_error = torch.mean(torch.abs(action_seq - decoded_action))
        offset_action_error = torch.mean(torch.abs(action_seq - predicted_action))
        action_error_max = torch.max(torch.abs(action_seq - predicted_action))

        loss = cbet_loss + self.config.offset_loss_weight * offset_loss

        loss_dict = {
            "loss": loss,
            "classification_loss": cbet_loss.detach().cpu().item(),
            "offset_loss": offset_loss.detach().cpu().item(),
            "equal_primary_code_rate": equal_primary_code_rate.detach().cpu().item(),
            "equal_secondary_code_rate": equal_secondary_code_rate.detach().cpu().item(),
            "vq_action_error": vq_action_error.detach().cpu().item(),
            "offset_action_error": offset_action_error.detach().cpu().item(),
            "action_error_max": action_error_max.detach().cpu().item(),
            "action_mse_error": action_mse_error.detach().cpu().item(),
        }
        return loss_dict


class VQBeTRgbEncoder(nn.Module):
    """Encode RGB images, microphone audio, and spectrograms into 1D feature vectors.

    Supports three types of visual inputs with specialized backbones:
    - Standard RGB images (3 channels): front, side cameras
    - Microphone audio (mic_num channels): sound0, sound1 concatenated
    - Spectrogram (1 channel): spec input

    Similar to DiffusionRgbEncoder and ACT's multi-backbone approach.
    """

    def __init__(self, config: VQBeTConfig):
        super().__init__()
        self.config = config

        # Define image key categories
        self.std_image_keys = ["observation.images.front", "observation.images.side"]
        self.mic_image_keys = ["observation.images.sound0", "observation.images.sound1"]
        self.spec_image_key = "observation.images.spec"

        # Set up optional preprocessing (resizing)
        if config.crop_shape is not None:
            self.do_resize = True
            self.resize = torchvision.transforms.Resize(config.crop_shape)
        else:
            self.do_resize = False

        # Track which backbones are needed
        self.has_std_backbone = any(key in config.image_features for key in self.std_image_keys)
        self.has_mic_backbone = any(key in config.image_features for key in self.mic_image_keys)
        self.has_spec_backbone = self.spec_image_key in config.image_features

        # Get backbone model factory
        backbone_model_fn = getattr(torchvision.models, config.vision_backbone)

        # Standard 3-channel backbone (for front, side cameras)
        if self.has_std_backbone:
            backbone_model = backbone_model_fn(weights=config.pretrained_backbone_weights)
            self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
            if config.use_group_norm:
                if config.pretrained_backbone_weights:
                    raise ValueError(
                        "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                    )
                self.backbone = _replace_submodules(
                    root_module=self.backbone,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
                )

        # Microphone backbone (mic_num channels) - no pretrained weights
        if self.has_mic_backbone:
            mic_backbone_model = backbone_model_fn(weights=None)
            # Replace first conv layer for mic_num channels
            orig_conv = mic_backbone_model.conv1
            mic_backbone_model.conv1 = nn.Conv2d(
                config.mic_num,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )
            self.mic_backbone = nn.Sequential(*(list(mic_backbone_model.children())[:-2]))
            if config.use_group_norm:
                self.mic_backbone = _replace_submodules(
                    root_module=self.mic_backbone,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
                )

        # Spectrogram backbone (1 channel) - no pretrained weights
        if self.has_spec_backbone:
            spec_backbone_model = backbone_model_fn(weights=None)
            # Replace first conv layer for 1 channel
            orig_conv = spec_backbone_model.conv1
            spec_backbone_model.conv1 = nn.Conv2d(
                1,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )
            self.spec_backbone = nn.Sequential(*(list(spec_backbone_model.children())[:-2]))
            if config.use_group_norm:
                self.spec_backbone = _replace_submodules(
                    root_module=self.spec_backbone,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
                )

        # Get feature map shape using a dry run with the first available backbone
        # Use first image feature to determine dimensions
        first_key = next(iter(config.image_features.keys()))
        images_shape = config.image_features[first_key].shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]

        # Determine which backbone to use for getting feature map shape
        if self.has_std_backbone:
            dummy_input = torch.zeros(1, 3, *dummy_shape_h_w)
            feature_map_shape = get_output_shape(self.backbone, dummy_input.shape)[1:]
        elif self.has_mic_backbone:
            dummy_input = torch.zeros(1, config.mic_num, *dummy_shape_h_w)
            feature_map_shape = get_output_shape(self.mic_backbone, dummy_input.shape)[1:]
        elif self.has_spec_backbone:
            dummy_input = torch.zeros(1, 1, *dummy_shape_h_w)
            feature_map_shape = get_output_shape(self.spec_backbone, dummy_input.shape)[1:]
        else:
            raise ValueError("No backbone available. Check image_features configuration.")

        # Set up pooling and output layers (shared across all backbones)
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor, image_keys: list[str] | None = None) -> Tensor:
        """
        Process images through appropriate backbones based on image keys.

        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
               When processing multiple images, this should be called per-image.
            image_keys: Optional list of image keys to determine which backbone to use.
                       If None, uses standard backbone.
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe resize
        if self.do_resize:
            x = self.resize(x)

        # Select appropriate backbone based on channel count
        # This is a simple heuristic when image_keys is not provided
        in_channels = x.shape[1]

        if in_channels == 3 and self.has_std_backbone:
            features = self.backbone(x)
        elif in_channels == self.config.mic_num and self.has_mic_backbone:
            features = self.mic_backbone(x)
        elif in_channels == 1 and self.has_spec_backbone:
            features = self.spec_backbone(x)
        elif self.has_std_backbone:
            # Fallback to standard backbone
            features = self.backbone(x)
        elif self.has_mic_backbone:
            features = self.mic_backbone(x)
        elif self.has_spec_backbone:
            features = self.spec_backbone(x)
        else:
            raise ValueError(f"No suitable backbone for input with {in_channels} channels")

        # Pool and project
        x = torch.flatten(self.pool(features), start_dim=1)
        x = self.relu(self.out(x))
        return x

    def forward_with_keys(
        self, images: list[Tensor], image_keys: list[str]
    ) -> list[Tensor]:
        """
        Process multiple images through appropriate backbones based on their keys.

        Args:
            images: List of (B, C, H, W) image tensors
            image_keys: List of image key names corresponding to each image

        Returns:
            List of (B, D) feature tensors in consistent order
        """
        features_list = []
        mic_indices = []
        mic_processed = False

        # First pass: identify mic image indices
        for idx, key in enumerate(image_keys):
            if key in self.mic_image_keys:
                mic_indices.append(idx)

        # Process each image
        for idx, (img, key) in enumerate(zip(images, image_keys)):
            # Handle mic images: combine sound0 and sound1
            if key in self.mic_image_keys:
                if not mic_processed and len(mic_indices) > 0 and idx == min(mic_indices):
                    # Collect and concatenate all mic images
                    mic_imgs = [images[i] for i in sorted(mic_indices)]
                    combined_mic = torch.cat(mic_imgs, dim=1)
                    # Slice to mic_num channels
                    mic_input = combined_mic[:, : self.config.mic_num, :, :]

                    # Preprocess
                    if self.do_resize:
                        mic_input = self.resize(mic_input)

                    # Process through mic backbone
                    mic_features = self.mic_backbone(mic_input)
                    mic_features = torch.flatten(self.pool(mic_features), start_dim=1)
                    mic_features = self.relu(self.out(mic_features))
                    features_list.append(mic_features)
                    mic_processed = True
                # Skip subsequent mic images
                continue

            # Handle spectrogram
            elif key == self.spec_image_key:
                # Extract first channel only
                spec_input = img[:, 0:1, :, :]

                # Preprocess
                if self.do_resize:
                    spec_input = self.resize(spec_input)

                spec_features = self.spec_backbone(spec_input)
                spec_features = torch.flatten(self.pool(spec_features), start_dim=1)
                spec_features = self.relu(self.out(spec_features))
                features_list.append(spec_features)

            # Handle standard images (front, side, or any other)
            else:
                # Preprocess
                if self.do_resize:
                    img = self.resize(img)

                std_features = self.backbone(img)
                std_features = torch.flatten(self.pool(std_features), start_dim=1)
                std_features = self.relu(self.out(std_features))
                features_list.append(std_features)

        return features_list


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class VqVae(nn.Module):
    def __init__(
        self,
        config: VQBeTConfig,
    ):
        """
        VQ-VAE is composed of three parts: encoder, vq_layer, and decoder.
        Encoder and decoder are MLPs consisting of an input, output layer, and hidden layer, respectively.
        The vq_layer uses residual VQs.

        This class contains functions for training the encoder and decoder along with the residual VQ layer (for training phase 1),
        as well as functions to help BeT training part in training phase 2.
        """

        super().__init__()
        self.config = config
        # 'discretized' indicates whether the Residual VQ part is trained or not. (After finishing the training, we set discretized=True)
        self.register_buffer("discretized", torch.tensor(False))
        self.optimized_steps = 0
        # we use the fixed number of layers for Residual VQ across all environments.
        self.vqvae_num_layers = 2

        self.vq_layer = ResidualVQ(
            dim=config.vqvae_embedding_dim,
            num_quantizers=self.vqvae_num_layers,
            codebook_size=config.vqvae_n_embed,
        )

        self.encoder = MLP(
            in_channels=self.config.action_feature.shape[0] * self.config.action_chunk_size,
            hidden_channels=[
                config.vqvae_enc_hidden_dim,
                config.vqvae_enc_hidden_dim,
                config.vqvae_embedding_dim,
            ],
        )
        self.decoder = MLP(
            in_channels=config.vqvae_embedding_dim,
            hidden_channels=[
                config.vqvae_enc_hidden_dim,
                config.vqvae_enc_hidden_dim,
                self.config.action_feature.shape[0] * self.config.action_chunk_size,
            ],
        )

    def get_embeddings_from_code(self, encoding_indices):
        # This function gets code indices as inputs, and outputs embedding vectors corresponding to the code indices.
        with torch.no_grad():
            z_embed = self.vq_layer.get_codebook_vector_from_indices(encoding_indices)
            # since the RVQ has multiple layers, it adds the vectors in the axis of layers to provide a vector for that code combination.
            z_embed = z_embed.sum(dim=0)
        return z_embed

    def get_action_from_latent(self, latent):
        # given latent vector, this function outputs the decoded action.
        output = self.decoder(latent)
        if self.config.action_chunk_size == 1:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.config.action_feature.shape[0])
        else:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.config.action_feature.shape[0])

    def get_code(self, state):
        # in phase 2 of VQ-BeT training, we need a `ground truth labels of action data` to calculate the Focal loss for code prediction head. (please refer to section 3.3 in the paper https://huggingface.co/papers/2403.03181)
        # this function outputs the `GT code` of given action using frozen encoder and quantization layers. (please refer to Figure 2. in the paper https://huggingface.co/papers/2403.03181)
        state = einops.rearrange(state, "N T A -> N (T A)")
        with torch.no_grad():
            state_rep = self.encoder(state)
            state_rep_shape = state_rep.shape[:-1]
            state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
            state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
            state_vq = state_rep_flat.view(*state_rep_shape, -1)
            vq_code = vq_code.view(*state_rep_shape, -1)
            vq_loss_state = torch.sum(vq_loss_state)
            return state_vq, vq_code

    def vqvae_forward(self, state):
        # This function passes the given data through Residual VQ with Encoder and Decoder. Please refer to section 3.2 in the paper https://huggingface.co/papers/2403.03181).
        state = einops.rearrange(state, "N T A -> N (T A)")
        # We start with passing action (or action chunk) at:t+n through the encoder ϕ.
        state_rep = self.encoder(state)
        state_rep_shape = state_rep.shape[:-1]
        state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
        # The resulting latent embedding vector x = ϕ(at:t+n) is then mapped to an embedding vector in the codebook of the RVQ layers by the nearest neighbor look-up.
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        state_vq = state_rep_flat.view(*state_rep_shape, -1)
        vq_code = vq_code.view(*state_rep_shape, -1)
        # since the RVQ has multiple layers, it adds the vectors in the axis of layers to provide a vector for that code combination.
        vq_loss_state = torch.sum(vq_loss_state)
        # Then, the discretized vector zq(x) is reconstructed as ψ(zq(x)) by passing through the decoder ψ.
        dec_out = self.decoder(state_vq)
        # Calculate L1 reconstruction loss
        encoder_loss = (state - dec_out).abs().mean()
        # add encoder reconstruction loss and commitment loss
        rep_loss = encoder_loss + vq_loss_state * 5

        metric = (
            encoder_loss.clone().detach(),
            vq_loss_state.clone().detach(),
            vq_code,
            rep_loss.item(),
        )
        return rep_loss, metric


class FocalLoss(nn.Module):
    """
    From https://github.com/notmahi/miniBET/blob/main/behavior_transformer/bet.py
    """

    def __init__(self, gamma: float = 0, size_average: bool = True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if len(input.shape) == 3:
            N, T, _ = input.shape
            logpt = F.log_softmax(input, dim=-1)
            logpt = logpt.gather(-1, target.view(N, T, 1)).view(N, T)
        elif len(input.shape) == 2:
            logpt = F.log_softmax(input, dim=-1)
            logpt = logpt.gather(-1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class MLP(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
    ):
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1]))

        super().__init__(*layers)
