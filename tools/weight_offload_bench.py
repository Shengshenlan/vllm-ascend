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
"""Small single-process benchmark for Ascend weight offload experiments."""

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.offloader import get_offloader


def _npu_metric(name: str, default: int | None = None) -> int | None:
    try:
        return int(getattr(torch.npu, name)())
    except Exception:
        return default


def _sync_npu() -> None:
    try:
        torch.npu.synchronize()
    except Exception:
        return


def _reset_peak_memory() -> None:
    try:
        torch.npu.reset_peak_memory_stats()
    except Exception:
        return


def _rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _offloader_metrics() -> dict[str, Any]:
    offloader = get_offloader()
    module_offloaders = getattr(offloader, "module_offloaders", [])
    return {
        "offloader": type(offloader).__name__,
        "offload_layers": len(
            [layer for layer in module_offloaders if getattr(layer, "param_offloaders", None)]
        ),
        "cpu_storage_bytes": getattr(offloader, "total_cpu_storage_bytes", 0),
        "static_buffer_bytes": getattr(offloader, "total_static_buffer_bytes", 0),
        "estimated_npu_saving_bytes": getattr(offloader, "net_npu_saving_bytes", 0),
        "slot_capacity": getattr(offloader, "slot_capacity", 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--offload-group-size", type=int, default=0)
    parser.add_argument("--offload-num-in-group", type=int, default=1)
    parser.add_argument("--offload-prefetch-step", type=int, default=1)
    parser.add_argument("--prompt", default="Hello, my name is")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _reset_peak_memory()

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "enforce_eager": True,
        "disable_log_stats": True,
    }
    if args.offload_group_size > 0:
        llm_kwargs.update(
            {
                "offload_group_size": args.offload_group_size,
                "offload_num_in_group": args.offload_num_in_group,
                "offload_prefetch_step": args.offload_prefetch_step,
            }
        )

    init_start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    _sync_npu()
    init_s = time.perf_counter() - init_start

    metrics: dict[str, Any] = {
        "label": args.label,
        "model": args.model,
        "env": {
            "SOC_VERSION": os.environ.get("SOC_VERSION"),
            "ASCEND_HOME_PATH": os.environ.get("ASCEND_HOME_PATH"),
            "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get(
                "VLLM_ENABLE_V1_MULTIPROCESSING"
            ),
        },
        "config": {
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_tokens": args.max_tokens,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "offload_group_size": args.offload_group_size,
            "offload_num_in_group": args.offload_num_in_group,
            "offload_prefetch_step": args.offload_prefetch_step,
        },
        "init_s": init_s,
        "npu_allocated_after_init": _npu_metric("memory_allocated"),
        "npu_peak_after_init": _npu_metric("max_memory_allocated"),
        "rss_kb_after_init": _rss_kb(),
        **_offloader_metrics(),
    }

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        ignore_eos=True,
    )

    _reset_peak_memory()
    gen_start = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling_params)
    _sync_npu()
    gen_s = time.perf_counter() - gen_start

    generated_token_count = 0
    generated_text = ""
    if outputs and outputs[0].outputs:
        output = outputs[0].outputs[0]
        generated_text = output.text
        generated_token_count = len(output.token_ids)

    metrics.update(
        {
            "gen_s": gen_s,
            "generated_token_count": generated_token_count,
            "generated_tokens_per_s": (
                generated_token_count / gen_s if gen_s > 0 else None
            ),
            "generated_text_sample": generated_text[:160],
            "npu_allocated_after_gen": _npu_metric("memory_allocated"),
            "npu_peak_after_gen": _npu_metric("max_memory_allocated"),
            "rss_kb_after_gen": _rss_kb(),
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
