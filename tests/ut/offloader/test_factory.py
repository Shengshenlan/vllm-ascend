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

from types import SimpleNamespace

import pytest
from vllm.config import CompilationMode, CUDAGraphMode

from vllm_ascend.offloader.factory import (
    create_ascend_offloader,
    is_weight_offload_enabled,
    resolve_ascend_offload_config,
)
from vllm_ascend.offloader.prefetch import AscendPrefetchOffloader


def _config(
    *,
    backend="auto",
    cpu_offload_gb=0,
    cpu_offload_params=None,
    group_size=0,
    num_in_group=1,
    prefetch_step=1,
    offload_params=None,
    tp=1,
    pp=1,
    spec_config=None,
    cudagraph_mode=CUDAGraphMode.NONE,
    compilation_mode=CompilationMode.NONE,
    quant_config=None,
):
    return SimpleNamespace(
        offload_config=SimpleNamespace(
            offload_backend=backend,
            uva=SimpleNamespace(
                cpu_offload_gb=cpu_offload_gb,
                cpu_offload_params=set(cpu_offload_params or ()),
            ),
            prefetch=SimpleNamespace(
                offload_group_size=group_size,
                offload_num_in_group=num_in_group,
                offload_prefetch_step=prefetch_step,
                offload_params=set(offload_params or ()),
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
        ),
        speculative_config=spec_config,
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            mode=compilation_mode,
        ),
        model_config=SimpleNamespace(enforce_eager=True),
        quant_config=quant_config,
        device_config=SimpleNamespace(device="npu"),
    )


def test_auto_without_fields_returns_noop():
    effective = resolve_ascend_offload_config(_config())

    assert not effective.is_active
    assert effective.backend == "noop"


def test_auto_with_group_selects_prefetch():
    effective = resolve_ascend_offload_config(_config(group_size=2))

    assert effective.is_active
    assert effective.backend == "prefetch"
    assert effective.group_size == 2
    assert effective.device == "npu"


def test_create_prefetch_offloader():
    offloader = create_ascend_offloader(_config(group_size=2))

    assert isinstance(offloader, AscendPrefetchOffloader)
    assert is_weight_offload_enabled(offloader)


def test_uva_backend_fails_fast():
    with pytest.raises(ValueError, match="CUDA UVA"):
        resolve_ascend_offload_config(_config(backend="uva"))


def test_cpu_offload_gb_fails_fast():
    with pytest.raises(ValueError, match="cpu-offload-gb"):
        resolve_ascend_offload_config(_config(cpu_offload_gb=1))


def test_num_in_group_zero_fails_fast():
    with pytest.raises(ValueError, match="offload_num_in_group"):
        resolve_ascend_offload_config(_config(group_size=2, num_in_group=0))


def test_cpu_offload_params_maps_without_mutating_source():
    cfg = _config(
        backend="prefetch",
        group_size=2,
        cpu_offload_params={"mlp.down_proj"},
    )

    effective = resolve_ascend_offload_config(cfg)

    assert effective.offload_params == frozenset({"mlp.down_proj"})
    assert effective.compatibility_mapping
    assert cfg.offload_config.prefetch.offload_params == set()
    assert cfg.offload_config.uva.cpu_offload_params == {"mlp.down_proj"}


def test_prefetch_group_zero_returns_noop():
    effective = resolve_ascend_offload_config(_config(backend="prefetch", offload_params={"mlp"}))

    assert not effective.is_active
    assert effective.backend == "noop"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tp": 2}, "tensor_parallel_size=1"),
        ({"pp": 2}, "pipeline_parallel_size=1"),
        ({"spec_config": object()}, "speculative decoding"),
        ({"quant_config": object()}, "non-quantized"),
        ({"cudagraph_mode": CUDAGraphMode.FULL}, "cudagraph_mode=NONE"),
        ({"compilation_mode": CompilationMode.VLLM_COMPILE}, "compilation mode"),
    ],
)
def test_active_prefetch_rejects_unsupported_combinations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_ascend_offload_config(_config(group_size=2, **kwargs))


def test_310p_rejects_active_prefetch():
    with pytest.raises(ValueError, match="310P"):
        resolve_ascend_offload_config(_config(group_size=2), is_310p=True)
