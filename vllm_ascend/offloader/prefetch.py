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
"""Ascend prefetch weight offloader.

Selected layer parameters are moved to CPU storage during model construction
so weight loading writes the offloaded tensors directly on host memory.  After
weight processing, ``post_init`` allocates a small static NPU buffer pool and
uses a copy stream to prefetch CPU weights into the buffers before layer
forward.
"""

from collections.abc import Generator
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.offloader import BaseOffloader

from vllm_ascend.offloader.effective_config import AscendOffloadConfig
from vllm_ascend.offloader.selector import select_param_names, should_offload_layer
from vllm_ascend.offloader.states import BufferSlotState, ParamInfo, WeightState
from vllm_ascend.utils import current_stream, npu_stream_switch, prefetch_stream

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def _is_ascend_pin_memory_available() -> bool:
    try:
        probe = torch.empty(1, device="cpu", pin_memory=True)
    except RuntimeError as exc:
        logger.warning_once(
            "Pinned CPU storage is unavailable for Ascend weight offload; "
            "falling back to pageable CPU memory. Async copy overlap may be limited. "
            "Original error: %s",
            exc,
        )
        return False
    return bool(probe.is_pinned())


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _copy_to_cpu_storage(
    source: torch.Tensor,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    """Copy a tensor to CPU storage while preserving layout metadata."""
    if source.device.type == "cpu" and (not pin_memory or source.is_pinned()):
        return source

    cpu_storage = torch.empty_strided(
        size=source.size(),
        stride=source.stride(),
        dtype=source.dtype,
        layout=source.layout,
        device="cpu",
        pin_memory=pin_memory,
    )

    cpu_storage.copy_(source, non_blocking=False)
    return cpu_storage


class StaticNPUBufferPool:
    """Static ring buffer pool for offloaded layer parameters."""

    def __init__(
        self,
        param_infos: list[ParamInfo],
        slot_capacity: int,
        device: torch.device,
    ):
        if slot_capacity < 1:
            raise ValueError("slot_capacity must be >= 1")

        self.slot_capacity = slot_capacity
        self.device = device
        self.total_bytes = 0
        self.slot_states = [BufferSlotState.FREE for _ in range(slot_capacity)]
        self.slot_compute_done_events: list[Any | None] = [
            None for _ in range(slot_capacity)
        ]
        self.slot_compute_done_owners: list[Any | None] = [
            None for _ in range(slot_capacity)
        ]
        self._buffers: dict[tuple, list[torch.Tensor]] = {}

        unique_params: dict[tuple, ParamInfo] = {}
        for info in param_infos:
            unique_params.setdefault(info.key, info)

        for key, info in unique_params.items():
            buffers: list[torch.Tensor] = []
            for _ in range(slot_capacity):
                buffer = torch.empty_strided(
                    size=info.shape,
                    stride=info.stride,
                    dtype=info.dtype,
                    device=device,
                )
                buffers.append(buffer)
                self.total_bytes += info.num_bytes
            self._buffers[key] = buffers

        logger.debug(
            "Ascend weight offload buffer pool allocated: unique_params=%d, "
            "slot_capacity=%d, total=%.4f GiB",
            len(unique_params),
            self.slot_capacity,
            self.total_bytes / float(2**30),
        )

    def get_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        dtype: torch.dtype,
        slot_idx: int,
    ) -> torch.Tensor:
        key = (name, shape, stride, dtype)
        return self._buffers[key][slot_idx % self.slot_capacity]


class AscendPrefetchOffloader(BaseOffloader):
    """Ascend CPU/NPU weight offloader with async prefetch."""

    is_active = True

    def __init__(self, config: AscendOffloadConfig):
        self.config = config
        self.device = torch.device(config.device or "cpu")
        self.module_offloaders: list[AscendModuleOffloader] = []
        self.buffer_pool: StaticNPUBufferPool | None = None
        self.copy_stream: Any | None = None
        self.total_cpu_storage_bytes = 0
        self.total_static_buffer_bytes = 0
        self.total_original_offloaded_bytes = 0
        self.net_npu_saving_bytes = 0
        self._post_initialized = False

    @property
    def slot_capacity(self) -> int:
        return max(1, self.config.prefetch_step + 1)

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        if self.module_offloaders:
            raise RuntimeError("AscendPrefetchOffloader.wrap_modules() called more than once")

        modules: list[nn.Module] = []
        offloaded_modules: list[nn.Module] = []
        for module_index, module in enumerate(modules_generator):
            modules.append(module)
            if not should_offload_layer(
                module_index,
                self.config.group_size,
                self.config.num_in_group,
            ):
                continue

            param_names = [name for name, _ in module.named_parameters()]
            selected_param_names = select_param_names(param_names, self.config.offload_params)
            if not selected_param_names:
                continue

            module_offloader = AscendModuleOffloader(
                module=module,
                device=self.device,
                layer_idx=len(self.module_offloaders),
                param_names=selected_param_names,
            )
            self.module_offloaders.append(module_offloader)
            offloaded_modules.append(module)

        for layer_idx, module in enumerate(offloaded_modules):
            self._hook_module_forward(layer_idx, module)

        return modules

    def _hook_module_forward(self, layer_idx: int, module: nn.Module) -> None:
        original_forward = module.forward

        def forward(*args, **kwargs):
            self.wait_layer(layer_idx)
            output = original_forward(*args, **kwargs)
            self.mark_layer_done(layer_idx)
            self.prefetch_layer(layer_idx + self.config.prefetch_step)
            return output

        module.forward = forward  # type: ignore[method-assign]

    def post_init(self) -> None:
        if self._post_initialized:
            return
        self._post_initialized = True

        if not self.module_offloaders:
            logger.info("Ascend weight offload initialized with no matching layers.")
            return

        for offloader in self.module_offloaders:
            offloader.sync_cpu_storage()

        live_module_offloaders = [
            offloader for offloader in self.module_offloaders
            if offloader.param_offloaders
        ]
        if not live_module_offloaders:
            logger.info("Ascend weight offload initialized with no surviving parameters.")
            return

        param_infos: list[ParamInfo] = []
        device: torch.device | None = None
        for offloader in live_module_offloaders:
            param_infos.extend(offloader.get_param_infos())
            if device is None:
                device = offloader.device

        if device is None:
            return

        if device.type != "cpu":
            self.copy_stream = prefetch_stream()

        self.buffer_pool = StaticNPUBufferPool(
            param_infos=param_infos,
            slot_capacity=self.slot_capacity,
            device=device,
        )

        for idx, offloader in enumerate(live_module_offloaders):
            slot_idx = idx % self.slot_capacity
            offloader.assign_buffer_slot(self.buffer_pool, slot_idx)
            offloader.post_init()
            self.total_cpu_storage_bytes += offloader.cpu_storage_bytes
            self.total_original_offloaded_bytes += offloader.cpu_storage_bytes

        self.total_static_buffer_bytes = self.buffer_pool.total_bytes
        self.net_npu_saving_bytes = (
            self.total_original_offloaded_bytes - self.total_static_buffer_bytes
        )

        logger.info(
            "Ascend weight offload initialized: layers=%d, params=%d, "
            "cpu_storage=%.4f GiB, static_buffers=%.4f GiB, "
            "estimated_npu_saving=%.4f GiB, group_size=%d, num_in_group=%d, "
            "prefetch_step=%d, slot_capacity=%d",
            len(live_module_offloaders),
            sum(len(offloader.param_offloaders) for offloader in live_module_offloaders),
            self.total_cpu_storage_bytes / float(2**30),
            self.total_static_buffer_bytes / float(2**30),
            self.net_npu_saving_bytes / float(2**30),
            self.config.group_size,
            self.config.num_in_group,
            self.config.prefetch_step,
            self.slot_capacity,
        )

        started = 0
        for idx, offloader in enumerate(self.module_offloaders):
            if started >= min(self.config.prefetch_step, len(live_module_offloaders)):
                break
            if not offloader.param_offloaders:
                continue
            self.prefetch_layer(idx)
            started += 1

    def prefetch_layer(self, layer_idx: int) -> None:
        if not self.module_offloaders:
            return
        offloader = self.module_offloaders[layer_idx % len(self.module_offloaders)]
        if not offloader.param_offloaders:
            return
        if self.copy_stream is None:
            offloader.load_sync(mark_in_use=False)
            return
        offloader.prefetch_async(self.copy_stream)

    def wait_layer(self, layer_idx: int) -> None:
        if not self.module_offloaders:
            return
        self.module_offloaders[layer_idx].wait_ready()

    def mark_layer_done(self, layer_idx: int) -> None:
        if not self.module_offloaders:
            return
        self.module_offloaders[layer_idx].mark_done()

    def _wait_for_layer(self, layer_idx: int) -> None:
        self.wait_layer(layer_idx)

    def _start_prefetch(self, layer_idx: int) -> None:
        self.prefetch_layer(layer_idx)

    def sync_prev_onload(self) -> None:
        if self.copy_stream is None:
            return
        current_stream().wait_stream(self.copy_stream)

    def join_after_forward(self) -> None:
        self.sync_prev_onload()


class AscendModuleOffloader:
    """Manages CPU storage, NPU buffers and state for one offloaded layer."""

    def __init__(
        self,
        module: nn.Module,
        device: torch.device,
        layer_idx: int,
        param_names: list[str],
    ):
        self.module = module
        self.layer_idx = layer_idx
        self.device = device
        self.param_offloaders = [
            AscendParamOffloader(module=module, param_name=param_name)
            for param_name in param_names
        ]
        self.state = WeightState.CPU_ONLY
        self.cpu_storage_bytes = 0
        self.copy_done_event: Any | None = None
        self._copy_done_pending = False
        self.compute_done_event: Any | None = None
        self._compute_done_pending = False
        self._buffer_pool: StaticNPUBufferPool | None = None
        self._buffer_slot_idx = 0

    def sync_cpu_storage(self) -> None:
        live_param_offloaders: list[AscendParamOffloader] = []
        for param_offloader in self.param_offloaders:
            if param_offloader.sync_cpu_storage():
                live_param_offloaders.append(param_offloader)
        self.param_offloaders = live_param_offloaders

    def get_param_infos(self) -> list[ParamInfo]:
        return [offloader.param_info for offloader in self.param_offloaders]

    def assign_buffer_slot(self, pool: StaticNPUBufferPool, slot_idx: int) -> None:
        self._buffer_pool = pool
        self._buffer_slot_idx = slot_idx
        pool.slot_states[slot_idx] = BufferSlotState.RESERVED

        for param_offloader in self.param_offloaders:
            info = param_offloader.param_info
            buffer = pool.get_buffer(
                name=info.name,
                shape=info.shape,
                stride=info.stride,
                dtype=info.dtype,
                slot_idx=slot_idx,
            )
            param_offloader.assign_static_buffer(buffer)

    def post_init(self) -> None:
        self.cpu_storage_bytes = sum(
            param_offloader.cpu_storage_bytes
            for param_offloader in self.param_offloaders
        )

    def _reserve_slot_for_copy(self, copy_stream: Any) -> None:
        if self._buffer_pool is not None:
            slot_event = self._buffer_pool.slot_compute_done_events[self._buffer_slot_idx]
            if slot_event is not None:
                copy_stream.wait_event(slot_event)
                self._buffer_pool.slot_compute_done_events[self._buffer_slot_idx] = None
                owner = self._buffer_pool.slot_compute_done_owners[
                    self._buffer_slot_idx
                ]
                self._buffer_pool.slot_compute_done_owners[self._buffer_slot_idx] = None
                if owner is not None:
                    owner._compute_done_pending = False
        if self._compute_done_pending:
            assert self.compute_done_event is not None
            copy_stream.wait_event(self.compute_done_event)
            self._compute_done_pending = False
        if self._buffer_pool is not None:
            self._buffer_pool.slot_states[self._buffer_slot_idx] = BufferSlotState.LOADING

    def prefetch_async(self, copy_stream: Any) -> None:
        if self.state in (WeightState.COPY_QUEUED, WeightState.READY, WeightState.IN_USE):
            return

        self._reserve_slot_for_copy(copy_stream)

        # Active Ascend weight offload is eager-only in this phase.  Slot reuse
        # is protected by compute-done events, so do not serialize every H2D
        # prefetch behind the current compute stream.
        with npu_stream_switch(copy_stream):
            for param_offloader in self.param_offloaders:
                param_offloader.copy_to_static(non_blocking=True)

        if self.copy_done_event is None:
            self.copy_done_event = torch.npu.Event()
        self.copy_done_event.record(copy_stream)
        self._copy_done_pending = True
        self.state = WeightState.COPY_QUEUED

    def wait_ready(self) -> None:
        if not self.param_offloaders:
            return
        if self.state == WeightState.CPU_ONLY:
            self.load_sync()
            return

        if self.state == WeightState.COPY_QUEUED:
            assert self.copy_done_event is not None
            assert self._copy_done_pending
            current_stream().wait_event(self.copy_done_event)
            self._copy_done_pending = False
            self.state = WeightState.READY
            if self._buffer_pool is not None:
                self._buffer_pool.slot_states[self._buffer_slot_idx] = BufferSlotState.READY

        if self.state == WeightState.READY:
            self.state = WeightState.IN_USE
            if self._buffer_pool is not None:
                self._buffer_pool.slot_states[self._buffer_slot_idx] = BufferSlotState.IN_USE

    def mark_done(self) -> None:
        if not self.param_offloaders:
            return
        if self.state != WeightState.IN_USE:
            return
        if self.device.type == "cpu":
            self._compute_done_pending = False
            self.state = WeightState.CPU_ONLY
            if self._buffer_pool is not None:
                self._buffer_pool.slot_compute_done_events[self._buffer_slot_idx] = None
                self._buffer_pool.slot_compute_done_owners[self._buffer_slot_idx] = None
                self._buffer_pool.slot_states[self._buffer_slot_idx] = BufferSlotState.FREE
            return

        if self.compute_done_event is None:
            self.compute_done_event = torch.npu.Event()
        current_stream().record_event(self.compute_done_event)
        self._compute_done_pending = True
        self.state = WeightState.CPU_ONLY
        if self._buffer_pool is not None:
            self._buffer_pool.slot_compute_done_events[self._buffer_slot_idx] = (
                self.compute_done_event
            )
            self._buffer_pool.slot_compute_done_owners[self._buffer_slot_idx] = self
            self._buffer_pool.slot_states[self._buffer_slot_idx] = BufferSlotState.FREE

    def load_sync(self, *, mark_in_use: bool = True) -> None:
        if self.device.type != "cpu" and self._buffer_pool is not None:
            slot_event = self._buffer_pool.slot_compute_done_events[self._buffer_slot_idx]
            if slot_event is not None:
                current_stream().wait_event(slot_event)
                self._buffer_pool.slot_compute_done_events[self._buffer_slot_idx] = None
                owner = self._buffer_pool.slot_compute_done_owners[
                    self._buffer_slot_idx
                ]
                self._buffer_pool.slot_compute_done_owners[self._buffer_slot_idx] = None
                if owner is not None:
                    owner._compute_done_pending = False
        if self._compute_done_pending and self.device.type != "cpu":
            assert self.compute_done_event is not None
            current_stream().wait_event(self.compute_done_event)
            self._compute_done_pending = False
        for param_offloader in self.param_offloaders:
            param_offloader.copy_to_static(non_blocking=False)
        self.state = WeightState.IN_USE if mark_in_use else WeightState.READY
        if self._buffer_pool is not None:
            self._buffer_pool.slot_states[self._buffer_slot_idx] = (
                BufferSlotState.IN_USE if mark_in_use else BufferSlotState.READY
            )


class AscendParamOffloader:
    """Stores one parameter in CPU storage and points it to a static buffer."""

    def __init__(self, module: nn.Module, param_name: str):
        self.module = module
        self.param_name = param_name
        self.cpu_storage: torch.Tensor | None = None
        self.static_buffer: torch.Tensor | None = None
        self.cpu_storage_bytes = 0
        self.static_buffer_bytes = 0
        self._param_deleted = False
        self._offload_to_cpu()

    @property
    def param(self) -> nn.Parameter:
        obj: Any = self.module
        for attr in self.param_name.split("."):
            obj = getattr(obj, attr)
        return obj

    @property
    def param_info(self) -> ParamInfo:
        assert self.cpu_storage is not None
        return ParamInfo(
            name=self.param_name,
            shape=tuple(self.cpu_storage.shape),
            stride=tuple(self.cpu_storage.stride()),
            dtype=self.cpu_storage.dtype,
        )

    def _offload_to_cpu(self) -> None:
        param = self.param
        pin_memory = self._should_pin_memory(param.data)
        self.cpu_storage = _copy_to_cpu_storage(
            param.data,
            pin_memory=pin_memory,
        )
        self.cpu_storage_bytes = _tensor_nbytes(self.cpu_storage)
        param.data = self.cpu_storage

    def _should_pin_memory(self, source: torch.Tensor) -> bool:
        return _is_ascend_pin_memory_available()

    def _update_cpu_storage_from_param(self) -> None:
        param_data = self.param.data
        self.cpu_storage = _copy_to_cpu_storage(
            param_data,
            pin_memory=self._should_pin_memory(param_data),
        )
        self.cpu_storage_bytes = _tensor_nbytes(self.cpu_storage)

    def sync_cpu_storage(self) -> bool:
        try:
            self._update_cpu_storage_from_param()
        except AttributeError:
            logger.debug(
                "Skipping deleted offloaded parameter after weight processing: %s",
                self.param_name,
            )
            self._param_deleted = True
            self.cpu_storage = None
            self.cpu_storage_bytes = 0
            return False
        return True

    def assign_static_buffer(self, static_buffer: torch.Tensor) -> None:
        assert self.cpu_storage is not None
        self.static_buffer = static_buffer
        self.static_buffer_bytes = _tensor_nbytes(static_buffer)
        self.param.data = static_buffer

    def copy_to_static(self, *, non_blocking: bool) -> None:
        assert self.cpu_storage is not None
        assert self.static_buffer is not None
        self.static_buffer.copy_(self.cpu_storage, non_blocking=non_blocking)
