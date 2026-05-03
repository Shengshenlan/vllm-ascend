#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""State definitions for Ascend weight residency management."""

from dataclasses import dataclass
from enum import Enum

import torch


class WeightState(str, Enum):
    CPU_ONLY = "cpu_only"
    COPY_QUEUED = "copy_queued"
    READY = "ready"
    IN_USE = "in_use"
    EVICTING = "evicting"


class BufferSlotState(str, Enum):
    FREE = "free"
    RESERVED = "reserved"
    LOADING = "loading"
    READY = "ready"
    IN_USE = "in_use"


@dataclass(frozen=True)
class ParamInfo:
    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype

    @property
    def key(self) -> tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype]:
        return (self.name, self.shape, self.stride, self.dtype)

    @property
    def num_bytes(self) -> int:
        numel = 1
        for dim in self.shape:
            numel *= dim
        return numel * torch.empty((), dtype=self.dtype).element_size()


@dataclass
class LayerResidency:
    layer_idx: int
    state: WeightState = WeightState.CPU_ONLY


@dataclass
class CopyHandle:
    layer_idx: int
    done_event: object | None = None
