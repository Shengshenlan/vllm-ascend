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
"""Ascend model weight offloading infrastructure."""

from vllm_ascend.offloader.effective_config import (
    AscendOffloadBackend,
    AscendOffloadConfig,
)
from vllm_ascend.offloader.factory import (
    create_ascend_offloader,
    is_weight_offload_enabled,
)
from vllm_ascend.offloader.prefetch import AscendPrefetchOffloader

__all__ = [
    "AscendOffloadBackend",
    "AscendOffloadConfig",
    "AscendPrefetchOffloader",
    "create_ascend_offloader",
    "is_weight_offload_enabled",
]
