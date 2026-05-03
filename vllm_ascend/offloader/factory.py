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
"""Ascend model weight offloader factory."""

from typing import Any

from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.offloader import BaseOffloader, NoopOffloader

from vllm_ascend.offloader.effective_config import AscendOffloadConfig
from vllm_ascend.offloader.prefetch import AscendPrefetchOffloader

logger = init_logger(__name__)


def is_weight_offload_enabled(offloader: BaseOffloader | None) -> bool:
    """Return whether an offloader actively manages model weights."""
    return bool(getattr(offloader, "is_active", False))


def create_ascend_offloader(
    vllm_config: VllmConfig,
    *,
    is_310p: bool = False,
) -> BaseOffloader:
    """Create the effective Ascend weight offloader for a VllmConfig."""
    effective_config = resolve_ascend_offload_config(vllm_config, is_310p=is_310p)
    if not effective_config.is_active:
        if effective_config.disabled_reason:
            logger.debug("Ascend weight offload disabled: %s", effective_config.disabled_reason)
        return NoopOffloader()

    logger.info(
        "Ascend weight offload enabled: backend=%s, group_size=%d, "
        "num_in_group=%d, prefetch_step=%d, params=%s, "
        "compatibility_mapping=%s",
        effective_config.backend,
        effective_config.group_size,
        effective_config.num_in_group,
        effective_config.prefetch_step,
        sorted(effective_config.offload_params),
        effective_config.compatibility_mapping,
    )
    return AscendPrefetchOffloader(effective_config)


def resolve_ascend_offload_config(
    vllm_config: VllmConfig,
    *,
    is_310p: bool = False,
) -> AscendOffloadConfig:
    """Normalize upstream OffloadConfig into an immutable Ascend config."""
    offload_config = vllm_config.offload_config
    backend = offload_config.offload_backend
    uva = offload_config.uva
    prefetch = offload_config.prefetch

    if backend == "uva":
        raise ValueError(
            "Ascend does not support CUDA UVA weight offload. "
            "Use --offload-backend=prefetch with --offload-group-size instead."
        )

    if uva.cpu_offload_gb > 0:
        raise ValueError(
            "Ascend does not support --cpu-offload-gb / CUDA UVA weight offload. "
            "Use --offload-backend=prefetch with --offload-group-size instead."
        )

    if backend == "auto":
        if prefetch.offload_group_size > 0:
            backend = "prefetch"
        else:
            return AscendOffloadConfig.noop("no Ascend weight offload fields are set")

    if backend != "prefetch":
        return AscendOffloadConfig.noop(f"offload backend {backend!r} is not active")

    if prefetch.offload_group_size == 0:
        if prefetch.offload_params or uva.cpu_offload_params:
            logger.warning(
                "Ascend prefetch weight offload params are set, but "
                "offload_group_size is 0. No layers will be offloaded."
            )
        return AscendOffloadConfig.noop("prefetch backend has offload_group_size=0")

    if prefetch.offload_num_in_group <= 0:
        raise ValueError(
            f"offload_num_in_group ({prefetch.offload_num_in_group}) must be >= 1"
        )

    if prefetch.offload_num_in_group > prefetch.offload_group_size:
        raise ValueError(
            f"offload_num_in_group ({prefetch.offload_num_in_group}) must be "
            f"<= offload_group_size ({prefetch.offload_group_size})"
        )

    if prefetch.offload_prefetch_step < 1:
        raise ValueError(
            f"offload_prefetch_step ({prefetch.offload_prefetch_step}) must be >= 1"
        )

    _validate_ascend_active_offload_support(vllm_config, is_310p=is_310p)

    offload_params = frozenset(prefetch.offload_params)
    compatibility_mapping = False
    if not offload_params and uva.cpu_offload_params:
        offload_params = frozenset(uva.cpu_offload_params)
        compatibility_mapping = True
        logger.info(
            "Mapping cpu_offload_params to Ascend prefetch offload_params: %s",
            sorted(offload_params),
        )

    return AscendOffloadConfig(
        backend="prefetch",
        group_size=prefetch.offload_group_size,
        num_in_group=prefetch.offload_num_in_group,
        prefetch_step=prefetch.offload_prefetch_step,
        offload_params=offload_params,
        device=str(getattr(vllm_config.device_config, "device", "npu")),
        is_active=True,
        compatibility_mapping=compatibility_mapping,
    )


def _validate_ascend_active_offload_support(
    vllm_config: VllmConfig,
    *,
    is_310p: bool,
) -> None:
    parallel_config = vllm_config.parallel_config
    compilation_config = vllm_config.compilation_config
    if is_310p:
        raise ValueError("Ascend weight offload is not supported on the 310P runner yet.")

    if parallel_config.tensor_parallel_size > 1:
        raise ValueError("Ascend weight offload currently requires tensor_parallel_size=1.")

    if parallel_config.pipeline_parallel_size > 1:
        raise ValueError("Ascend weight offload currently requires pipeline_parallel_size=1.")

    if vllm_config.speculative_config is not None:
        raise ValueError("Ascend weight offload currently does not support speculative decoding.")

    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is not None:
        raise ValueError("Ascend weight offload currently supports non-quantized models only.")

    _raise_if_compilation_enabled(compilation_config)


def _raise_if_compilation_enabled(compilation_config: Any) -> None:
    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", CUDAGraphMode.NONE)
    compilation_mode = getattr(compilation_config, "mode", CompilationMode.NONE)

    if cudagraph_mode not in (None, CUDAGraphMode.NONE):
        raise ValueError("Ascend weight offload currently requires cudagraph_mode=NONE.")

    if compilation_mode not in (None, CompilationMode.NONE):
        raise ValueError("Ascend weight offload currently requires compilation mode NONE/eager.")
