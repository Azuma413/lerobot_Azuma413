#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Shared raw-waveform encoder used by the policies that consume `observation.audio`.

The multi-microphone waveform `(num_mics, num_samples)` is encoded by a wav2vec-style
stack of strided 1D convolutions. Only the *encoder* differs from the spectrogram-image
setup: the resulting features are handed to each policy's existing fusion mechanism
(token concatenation for ACT, global-conditioning concatenation for Diffusion).
"""

from torch import Tensor, nn


class RawAudioConvStack(nn.Module):
    """wav2vec-style stack of strided 1D convolutions over a multichannel waveform.

    Args:
        in_channels: number of microphones.
        layers: sequence of `(out_channels, kernel_size, stride)`.
        num_groups: group count for the GroupNorm placed after every convolution.
    """

    def __init__(
        self,
        in_channels: int,
        layers: tuple[tuple[int, int, int], ...],
        num_groups: int = 16,
    ):
        super().__init__()
        blocks = []
        channels = in_channels
        total_stride = 1
        for out_channels, kernel_size, stride in layers:
            blocks.append(
                nn.Conv1d(channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
            )
            blocks.append(nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels))
            blocks.append(nn.GELU())
            channels = out_channels
            total_stride *= stride
        self.net = nn.Sequential(*blocks)
        self.out_channels = channels
        self.total_stride = total_stride

    def forward(self, x: Tensor) -> Tensor:
        """(B, num_mics, num_samples) -> (B, out_channels, num_frames)."""
        return self.net(x)


def normalize_conv_layers(layers) -> tuple[tuple[int, int, int], ...]:
    """Accept the config value whether it survived JSON round-tripping as lists or tuples."""
    return tuple((int(c), int(k), int(s)) for c, k, s in layers)
