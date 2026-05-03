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

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from vllm_ascend.offloader.effective_config import AscendOffloadConfig
from vllm_ascend.offloader.prefetch import AscendPrefetchOffloader, StaticNPUBufferPool
from vllm_ascend.offloader.states import BufferSlotState, ParamInfo, WeightState


def _offload_config(**kwargs):
    values = {
        "backend": "prefetch",
        "group_size": 1,
        "num_in_group": 1,
        "prefetch_step": 1,
        "is_active": True,
    }
    values.update(kwargs)
    return AscendOffloadConfig(**values)


def test_wrap_modules_moves_selected_params_to_cpu_storage():
    module = nn.Linear(4, 3)
    original_weight = module.weight.detach().clone()

    offloader = AscendPrefetchOffloader(_offload_config())
    wrapped = offloader.wrap_modules(iter([module]))

    assert wrapped == [module]
    assert module.weight.device.type == "cpu"
    param_offloader = offloader.module_offloaders[0].param_offloaders[0]
    assert torch.equal(param_offloader.cpu_storage, original_weight)
    assert module.weight.data.data_ptr() == param_offloader.cpu_storage.data_ptr()


def test_post_init_syncs_final_loaded_weights_and_uses_ring_pool():
    modules = [nn.Linear(4, 3), nn.Linear(4, 3), nn.Linear(4, 3)]
    offloader = AscendPrefetchOffloader(_offload_config(prefetch_step=1))
    offloader.wrap_modules(iter(modules))

    modules[0].weight.data.fill_(1)
    modules[1].weight.data.fill_(2)
    modules[2].weight.data.fill_(3)

    offloader.post_init()

    assert offloader.buffer_pool is not None
    assert offloader.buffer_pool.slot_capacity == 2
    assert offloader.total_cpu_storage_bytes > offloader.total_static_buffer_bytes
    assert offloader.net_npu_saving_bytes > 0
    assert torch.equal(
        offloader.module_offloaders[1].param_offloaders[0].cpu_storage,
        torch.full_like(modules[1].weight.data, 2),
    )


def test_forward_waits_current_layer_and_prefetches_next_layer_on_cpu_path():
    modules = [nn.Linear(4, 4), nn.Linear(4, 4)]
    offloader = AscendPrefetchOffloader(_offload_config(prefetch_step=1))
    wrapped = offloader.wrap_modules(iter(modules))
    offloader.post_init()

    assert offloader.module_offloaders[0].state == WeightState.READY
    assert offloader.module_offloaders[1].state == WeightState.CPU_ONLY

    wrapped[0](torch.ones(2, 4))

    assert offloader.module_offloaders[0].state == WeightState.CPU_ONLY
    assert offloader.module_offloaders[1].state == WeightState.READY


def test_post_init_prunes_deleted_parameters():
    module = nn.Linear(4, 3)
    offloader = AscendPrefetchOffloader(_offload_config())
    offloader.wrap_modules(iter([module]))

    del module.weight
    offloader.post_init()

    assert len(offloader.module_offloaders) == 1
    assert len(offloader.module_offloaders[0].param_offloaders) == 1
    assert offloader.module_offloaders[0].param_offloaders[0].param_name == "bias"


def test_static_buffer_pool_uses_name_shape_stride_dtype_key():
    infos = [
        ParamInfo("a.weight", (2, 3), (3, 1), torch.float16),
        ParamInfo("b.weight", (2, 3), (3, 1), torch.float16),
    ]

    pool = StaticNPUBufferPool(infos, slot_capacity=2, device=torch.device("cpu"))

    assert pool.get_buffer("a.weight", (2, 3), (3, 1), torch.float16, 0) is not (
        pool.get_buffer("b.weight", (2, 3), (3, 1), torch.float16, 0)
    )
    assert pool.get_buffer("a.weight", (2, 3), (3, 1), torch.float16, 0) is (
        pool.get_buffer("a.weight", (2, 3), (3, 1), torch.float16, 2)
    )
    assert pool.slot_states == [BufferSlotState.FREE, BufferSlotState.FREE]


def test_duplicate_prefetch_is_noop_on_ready_layer_cpu_path():
    module = nn.Linear(4, 4)
    offloader = AscendPrefetchOffloader(_offload_config())
    offloader.wrap_modules(iter([module]))
    offloader.post_init()
    first_ptr = module.weight.data.data_ptr()

    offloader.prefetch_layer(0)

    assert offloader.module_offloaders[0].state == WeightState.READY
    assert module.weight.data.data_ptr() == first_ptr


def test_prefetch_async_does_not_wait_on_current_stream_without_slot_event():
    module = nn.Linear(4, 4)
    offloader = AscendPrefetchOffloader(_offload_config())
    offloader.wrap_modules(iter([module]))
    offloader.post_init()

    module_offloader = offloader.module_offloaders[0]
    module_offloader.wait_ready()
    module_offloader.mark_done()
    copy_stream = MagicMock()
    event = MagicMock()

    with (
        patch("vllm_ascend.offloader.prefetch.npu_stream_switch"),
        patch("vllm_ascend.offloader.prefetch.torch.npu.Event", return_value=event),
    ):
        module_offloader.prefetch_async(copy_stream)

    copy_stream.wait_event.assert_not_called()
    event.record.assert_called_once_with(copy_stream)


def test_prefetch_async_reuses_copy_done_event():
    module = nn.Linear(4, 4)
    offloader = AscendPrefetchOffloader(_offload_config())
    offloader.wrap_modules(iter([module]))
    offloader.post_init()

    module_offloader = offloader.module_offloaders[0]
    copy_stream = MagicMock()
    compute_stream = MagicMock()
    copy_done_event = MagicMock()
    module_offloader.state = WeightState.CPU_ONLY

    with (
        patch("vllm_ascend.offloader.prefetch.current_stream", return_value=compute_stream),
        patch("vllm_ascend.offloader.prefetch.npu_stream_switch"),
        patch(
            "vllm_ascend.offloader.prefetch.torch.npu.Event",
            return_value=copy_done_event,
        ) as event_factory,
    ):
        module_offloader.prefetch_async(copy_stream)
        module_offloader.wait_ready()
        module_offloader.mark_done()
        module_offloader.prefetch_async(copy_stream)

    event_factory.assert_called_once_with()
    assert copy_done_event.record.call_count == 2
    compute_stream.wait_event.assert_called_once_with(copy_done_event)


def test_mark_done_reuses_compute_done_event():
    module = nn.Linear(4, 4)
    offloader = AscendPrefetchOffloader(_offload_config())
    offloader.wrap_modules(iter([module]))
    offloader.post_init()

    module_offloader = offloader.module_offloaders[0]
    module_offloader.device = torch.device("npu")
    module_offloader.state = WeightState.IN_USE
    compute_stream = MagicMock()
    compute_done_event = MagicMock()

    with (
        patch("vllm_ascend.offloader.prefetch.current_stream", return_value=compute_stream),
        patch(
            "vllm_ascend.offloader.prefetch.torch.npu.Event",
            return_value=compute_done_event,
        ) as event_factory,
    ):
        module_offloader.mark_done()
        module_offloader.load_sync()
        module_offloader.mark_done()

    event_factory.assert_called_once_with()
    assert compute_stream.record_event.call_count == 2
    compute_stream.record_event.assert_called_with(compute_done_event)


def test_slot_event_wait_clears_original_owner_pending_state():
    modules = [nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)]
    offloader = AscendPrefetchOffloader(_offload_config(prefetch_step=1))
    offloader.wrap_modules(iter(modules))
    offloader.post_init()

    first = offloader.module_offloaders[0]
    third = offloader.module_offloaders[2]
    first.device = torch.device("npu")
    third.device = torch.device("npu")
    first.state = WeightState.IN_USE
    compute_stream = MagicMock()
    compute_done_event = MagicMock()
    copy_done_event = MagicMock()
    copy_stream = MagicMock()

    with (
        patch("vllm_ascend.offloader.prefetch.current_stream", return_value=compute_stream),
        patch("vllm_ascend.offloader.prefetch.torch.npu.Event", return_value=compute_done_event),
    ):
        first.mark_done()

    assert first._compute_done_pending

    with (
        patch("vllm_ascend.offloader.prefetch.npu_stream_switch"),
        patch("vllm_ascend.offloader.prefetch.torch.npu.Event", return_value=copy_done_event),
    ):
        third.prefetch_async(copy_stream)

    copy_stream.wait_event.assert_called_once_with(compute_done_event)
    assert not first._compute_done_pending
    assert first.compute_done_event is compute_done_event
    assert offloader.buffer_pool is not None
    assert offloader.buffer_pool.slot_compute_done_events[0] is None
    assert offloader.buffer_pool.slot_compute_done_owners[0] is None
