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
"""Pure selection helpers for Ascend weight offloading."""


def should_offload_layer(module_index: int, group_size: int, num_in_group: int) -> bool:
    """Return whether a decoder layer belongs to the offload group."""
    if group_size <= 0:
        return False
    if num_in_group <= 0 or num_in_group > group_size:
        raise ValueError(
            f"offload_num_in_group ({num_in_group}) must be in [1, {group_size}]"
        )
    return module_index % group_size >= group_size - num_in_group


def matches_param_segments(param_name: str, offload_params: set[str] | frozenset[str]) -> bool:
    """Match parameter name segments using vLLM's exact-segment convention."""
    if not offload_params:
        return True
    normalized_name = f".{param_name}."
    return any(f".{param}." in normalized_name for param in offload_params)


def select_param_names(
    param_names: list[str],
    offload_params: set[str] | frozenset[str],
) -> list[str]:
    """Return parameters selected by offload_params."""
    return [name for name in param_names if matches_param_segments(name, offload_params)]
