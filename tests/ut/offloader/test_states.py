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

import torch

from vllm_ascend.offloader.states import BufferSlotState, ParamInfo, WeightState


def test_weight_and_buffer_slot_states_are_stable_strings():
    assert WeightState.CPU_ONLY.value == "cpu_only"
    assert BufferSlotState.IN_USE.value == "in_use"


def test_param_info_key_and_num_bytes():
    info = ParamInfo(
        name="mlp.down_proj.weight",
        shape=(2, 3),
        stride=(3, 1),
        dtype=torch.float16,
    )

    assert info.key == ("mlp.down_proj.weight", (2, 3), (3, 1), torch.float16)
    assert info.num_bytes == 12
