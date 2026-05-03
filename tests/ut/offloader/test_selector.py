#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import pytest

from vllm_ascend.offloader.selector import (
    matches_param_segments,
    select_param_names,
    should_offload_layer,
)


def test_should_offload_layer_uses_last_n_layers_in_group():
    selected = [
        idx for idx in range(8)
        if should_offload_layer(idx, group_size=4, num_in_group=2)
    ]

    assert selected == [2, 3, 6, 7]


def test_should_offload_layer_disabled_when_group_size_zero():
    assert not should_offload_layer(0, group_size=0, num_in_group=1)


def test_should_offload_layer_rejects_invalid_num_in_group():
    with pytest.raises(ValueError, match="offload_num_in_group"):
        should_offload_layer(0, group_size=2, num_in_group=3)


def test_matches_param_segments_exact_segments():
    assert matches_param_segments("mlp.down_proj.weight", frozenset({"mlp.down_proj"}))
    assert matches_param_segments("mlp.down_proj.weight", frozenset({"down_proj"}))
    assert not matches_param_segments("mlp.down_proj_scale.weight", frozenset({"down_proj"}))


def test_select_param_names_empty_params_selects_all():
    assert select_param_names(["a.weight", "b.weight"], frozenset()) == ["a.weight", "b.weight"]


def test_select_param_names_filters_by_segment():
    names = ["self_attn.o_proj.weight", "mlp.down_proj.weight"]
    assert select_param_names(names, frozenset({"mlp.down_proj"})) == ["mlp.down_proj.weight"]
