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
"""Effective Ascend weight offload configuration."""

from dataclasses import dataclass, field
from typing import Literal

AscendOffloadBackend = Literal["noop", "prefetch"]


@dataclass(frozen=True)
class AscendOffloadConfig:
    """Normalized Ascend offload config.

    This is intentionally separate from vLLM's OffloadConfig. Ascend accepts
    the upstream CLI/config surface, but it must not mutate that source config
    while mapping compatibility fields such as cpu_offload_params.
    """

    backend: AscendOffloadBackend = "noop"
    group_size: int = 0
    num_in_group: int = 1
    prefetch_step: int = 1
    offload_params: frozenset[str] = field(default_factory=frozenset)
    device: str | None = None
    is_active: bool = False
    disabled_reason: str | None = None
    compatibility_mapping: bool = False

    @classmethod
    def noop(cls, reason: str | None = None) -> "AscendOffloadConfig":
        return cls(disabled_reason=reason)
