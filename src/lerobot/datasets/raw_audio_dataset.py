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
"""LeRobotDataset variant that serves multichannel raw audio from per-episode WAV files.

A 1 second, 32 channel window at 16 kHz is 2 MB as float32. Storing that for every frame
would blow the parquet files up to ~130 GB for the four sound tasks, and 2/3 of the frames
would be exact duplicates (the acoustic simulation only updates at 10 Hz).

Instead the collection script writes one continuous 32 channel WAV per episode plus a
per-frame `audio_end_sample` index into that stream. This class slices the window that ends
at the frame's index and exposes it as `observation.audio`, so from the policy's point of
view it is an ordinary observation feature.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_AUDIO

AUDIO_INFO_PATH = "meta/audio_info.json"
AUDIO_END_SAMPLE_KEY = "audio_end_sample"


def has_raw_audio(root: str | Path | None) -> bool:
    """Whether the dataset directory carries per-episode raw audio."""
    return root is not None and (Path(root) / AUDIO_INFO_PATH).is_file()


class RawAudioLeRobotDataset(LeRobotDataset):
    """`LeRobotDataset` that adds an `observation.audio` window read from episode WAV files."""

    def __init__(self, *args, **kwargs):
        # The waveform lives outside the parquet files, so it must not be queried through the
        # regular delta-timestamp machinery; this class assembles the history itself.
        delta_timestamps = kwargs.get("delta_timestamps")
        if delta_timestamps and OBS_AUDIO in delta_timestamps:
            kwargs["delta_timestamps"] = {
                k: v for k, v in delta_timestamps.items() if k != OBS_AUDIO
            }
        super().__init__(*args, **kwargs)

        with open(Path(self.root) / AUDIO_INFO_PATH) as f:
            self.audio_info = json.load(f)

        self.audio_num_channels = int(self.audio_info["num_channels"])
        self.audio_window = int(self.audio_info["window"])
        self.audio_scale = float(self.audio_info["full_scale"]) / 32767.0
        self._audio_path_template = self.audio_info["path_template"]
        self._wav_cache: dict[int, np.ndarray] = {}

        # Expose the waveform as a regular dataset feature so that
        # `dataset_to_policy_features` turns it into a FeatureType.AUDIO policy input.
        self.meta.info["features"][OBS_AUDIO] = {
            "dtype": "float32",
            "shape": (self.audio_num_channels, self.audio_window),
            "names": None,
        }
        stats = self.audio_info["stats"]
        # Normalize every channel with a shared scale so that inter-microphone level
        # differences — which is where the localisation cue lives — are preserved.
        global_std = float(stats["global_std"]) or 1.0
        self.meta.stats[OBS_AUDIO] = {
            "mean": np.asarray(stats["mean"], dtype=np.float32).reshape(-1, 1),
            "std": np.full((self.audio_num_channels, 1), global_std, dtype=np.float32),
        }

        self._audio_end_sample = self._load_audio_end_samples()
        logging.info(
            f"Raw audio enabled: {self.audio_num_channels}ch x {self.audio_window} samples "
            f"@ {self.audio_info['sample_rate']}Hz"
        )

    def _load_audio_end_samples(self) -> np.ndarray:
        self._ensure_hf_dataset_loaded()
        column = self.hf_dataset[AUDIO_END_SAMPLE_KEY]
        if isinstance(column, torch.Tensor):
            column = column.numpy()
        return np.asarray(column, dtype=np.int64).reshape(-1)

    def _get_wav(self, episode_index: int) -> np.ndarray:
        """Memory-mapped (num_samples, num_channels) int16 array for one episode."""
        wav = self._wav_cache.get(episode_index)
        if wav is None:
            path = Path(self.root) / self._audio_path_template.format(episode_index=episode_index)
            _, wav = wavfile.read(path, mmap=True)
            if wav.ndim == 1:
                wav = wav[:, None]
            self._wav_cache[episode_index] = wav
        return wav

    def _read_window(self, episode_index: int, end_sample: int) -> np.ndarray:
        """Waveform window of `audio_window` samples ending at `end_sample`, zero-padded in front."""
        wav = self._get_wav(episode_index)
        end = int(min(end_sample, wav.shape[0]))
        start = max(0, end - self.audio_window)
        out = np.zeros((self.audio_num_channels, self.audio_window), dtype=np.float32)
        if end > start:
            chunk = np.asarray(wav[start:end], dtype=np.float32).T * self.audio_scale
            out[:, self.audio_window - chunk.shape[1] :] = chunk
        return out

    def _audio_query_indices(self, idx: int, ep_idx: int) -> list[int]:
        """Frame indices whose windows make up this item (mirrors `_get_query_indices`)."""
        deltas = None
        if self.delta_indices is not None:
            # Observation keys all share the same deltas; reuse whichever is available.
            for key, delta_idx in self.delta_indices.items():
                if key.startswith("observation"):
                    deltas = delta_idx
                    break
        if deltas is None:
            return [idx]

        ep = self.meta.episodes[ep_idx]
        ep_start, ep_end = ep["dataset_from_index"], ep["dataset_to_index"]
        return [max(ep_start, min(ep_end - 1, idx + delta)) for delta in deltas]

    def __getitem__(self, idx) -> dict:
        item = super().__getitem__(idx)
        ep_idx = int(item["episode_index"].item())
        indices = self._audio_query_indices(idx, ep_idx)
        windows = [self._read_window(ep_idx, self._audio_end_sample[i]) for i in indices]
        audio = windows[0] if len(windows) == 1 else np.stack(windows, axis=0)
        item[OBS_AUDIO] = torch.from_numpy(audio)
        return item
