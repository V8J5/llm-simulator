#!/usr/bin/env python3
"""Collect fixed-batch vLLM truth for one Prefill or Decode shape.

All requests in a repeat are released together. Prefill uses time-to-first-token
with one generated token; Decode uses per-request TPOT over several generated
tokens. This is an end-to-end iteration validation artifact, not an operator
profile and not an online-arrival workload.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profiling_common import (  # noqa: E402
    SCHEMA_VERSION,
    collect_runtime_metadata,
    describe_ms,
    sha256_file,
    write_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--stage", choices=("prefill", "decode"), required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--length", type=int, required=True,
                        help="Prompt length for Prefill; initial KV length for Decode")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16",
                        help="Server model dtype recorded for reproducibility")
    parser.add_argument("--warmup", type=int, default=1, help="Full-batch warmup repetitions")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-start-skew-ms", type=float, default=50.0)
    parser.add_argument("--deployment-config", type=Path,
                        help="JSON containing the exact vLLM launch/runtime configuration")
    parser.add_argument("--server-command-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def exact_length_prompt(tokenizer, target: int) -> List[int]:
    seed = tokenizer.encode(
        "The quick brown fox describes an inference system. ",
        add_special_tokens=False)
    if not seed or target <= 0:
        raise ValueError("tokenizer seed and target prompt length must be positive")
    return (seed * (target // len(seed) + 1))[:target]


def unique_prompt_ids(base: List[int], namespace: str) -> List[int]:
    """Give every request a distinct first token block without changing length."""
    if not base:
        raise ValueError("cannot uniquify an empty prompt")
    result = list(base)
    pool = sorted(set(base))
    if len(pool) < 2:
        raise ValueError("prompt seed needs at least two distinct token ids")
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    for index in range(min(16, len(result))):
        result[index] = pool[digest[index] % len(pool)]
    return result


def prompt_sha256(prompt_ids: List[int]) -> str:
    encoded = ",".join(str(item) for item in prompt_ids).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


async def one_request(client, request, prompt_length: int, output_tokens: int,
                      prompt_fingerprint: str, request_id: str,
                      release: asyncio.Event) -> Dict:
    """Send an already-built HTTP request after the batch release barrier.

    Building an OpenAI request serializes the (potentially long) token-id
    prompt.  Doing that after releasing the barrier creates artificial skew
    proportional to batch size.  The caller therefore prepares every request
    before scheduling this coroutine.
    """
    await release.wait()
    started = time.perf_counter()
    first_token = None
    last_token = None
    usage = None
    error = None
    try:
        response = await client.send(request, stream=True)
        if response.is_error:
            await response.aread()
            response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            chunk = json.loads(raw)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.perf_counter()
                first_token = first_token or now
                last_token = now
        await response.aclose()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    completion_tokens = int((usage or {}).get("completion_tokens", 0) or 0)
    ttft_ms = (first_token - started) * 1000 if first_token else None
    tpot_ms = ((last_token - first_token) * 1000 / (completion_tokens - 1)
               if first_token and last_token and completion_tokens > 1 else None)
    return {
        "request_id": request_id,
        "prompt_sha256": prompt_fingerprint,
        "status": "success" if error is None and first_token else "failed",
        "error": error or (None if first_token else "response contained no output token"),
        "request_start_monotonic_s": started,
        "actual_prompt_tokens": int((usage or {}).get("prompt_tokens", 0) or
                                    prompt_length),
        "actual_output_tokens": completion_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": (ended - started) * 1000,
    }


async def run_batch(client, args: argparse.Namespace, prompt_ids: List[int],
                    repeat_id: str) -> Dict:
    output_tokens = 1 if args.stage == "prefill" else args.decode_tokens
    release = asyncio.Event()
    prepared = []
    request_ids = []
    for index in range(args.batch_size):
        request_id = (
            f"fixed_{args.stage}_b{args.batch_size}_l{len(prompt_ids)}_"
            f"{repeat_id}_{index + 1}")
        request_prompt_ids = unique_prompt_ids(prompt_ids, request_id)
        request = client.build_request("POST", "completions", json={
            "model": args.model,
            "prompt": request_prompt_ids,
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "request_id": request_id,
            "ignore_eos": True,
        })
        prepared.append(request)
        request_ids.append((request_id, prompt_sha256(request_prompt_ids)))
    tasks = [asyncio.create_task(one_request(
        client, request, len(prompt_ids), output_tokens, fingerprint,
        request_id, release))
        for request, (request_id, fingerprint) in zip(prepared, request_ids)]
    await asyncio.sleep(0)
    released = time.perf_counter()
    release.set()
    requests = await asyncio.gather(*tasks)
    successful = [row for row in requests if row["status"] == "success"]
    starts = [row["request_start_monotonic_s"] for row in requests]
    if args.stage == "prefill":
        metrics = [float(row["ttft_ms"]) for row in successful
                   if row["ttft_ms"] is not None]
        observed = max(metrics) if len(metrics) == args.batch_size else None
        metric_name = "batch_time_to_all_first_tokens_ms"
    else:
        metrics = [float(row["tpot_ms"]) for row in successful
                   if row["tpot_ms"] is not None]
        observed = statistics.fmean(metrics) if len(metrics) == args.batch_size else None
        metric_name = "mean_request_tpot_ms"
    return {
        "repeat_id": repeat_id,
        "release_monotonic_s": released,
        "request_start_skew_ms": (max(starts) - min(starts)) * 1000 if starts else 0.0,
        "successful_requests": len(successful),
        "metric_name": metric_name,
        "observed_iteration_ms": observed,
        "requests": requests,
    }


def load_deployment(args: argparse.Namespace) -> Dict:
    result = {}
    if args.deployment_config:
        config = json.loads(args.deployment_config.read_text(encoding="utf-8"))
        configured_tp = config.get("tensor_parallel_size", config.get("tp_size"))
        if configured_tp is not None and int(configured_tp) != args.tp_size:
            raise ValueError("deployment TP does not match --tp-size")
        configured_model = config.get("model")
        if configured_model is not None and str(configured_model) != args.model:
            raise ValueError("deployment model does not match --model")
        if config.get("enable_prefix_caching") is True:
            raise ValueError("fixed-batch baseline requires prefix caching to be disabled")
        max_sequences = config.get("max_num_seqs")
        if max_sequences is not None and int(max_sequences) < args.batch_size:
            raise ValueError("deployment max_num_seqs is smaller than --batch-size")
        result["config_path"] = str(args.deployment_config.resolve())
        result["config_sha256"] = sha256_file(args.deployment_config)
        result["config"] = config
    if args.server_command_file:
        result["server_command_file"] = str(args.server_command_file.resolve())
        result["server_command"] = args.server_command_file.read_text(
            encoding="utf-8", errors="replace").strip()
    return result


async def main_async(args: argparse.Namespace) -> int:
    if min(args.tp_size, args.batch_size, args.length, args.repeats) <= 0:
        raise SystemExit("TP, batch, length and repeats must be positive")
    if args.stage == "decode" and args.decode_tokens < 2:
        raise SystemExit("Decode validation needs at least two output tokens")
    import httpx
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True)
    prompt_ids = exact_length_prompt(tokenizer, args.length)
    base_url = args.base_url.rstrip("/") + "/"
    async with httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {args.api_key}"},
            timeout=args.timeout,
            trust_env=False) as client:
        response = await client.get("models")
        response.raise_for_status()
        warmups = []
        for index in range(args.warmup):
            warmups.append(await run_batch(
                client, args, prompt_ids, f"warmup_{index + 1}"))
        samples = []
        for index in range(args.repeats):
            sample = await run_batch(client, args, prompt_ids, f"repeat_{index + 1}")
            samples.append(sample)
            print(f"repeat {index + 1}/{args.repeats}: "
                  f"{sample['observed_iteration_ms']} ms")

    values = [float(row["observed_iteration_ms"]) for row in samples
              if row["observed_iteration_ms"] is not None]
    max_start_skew = max((row["request_start_skew_ms"] for row in samples), default=0.0)
    benchmark = {
        "kind": "vllm_fixed_batch_iteration_validation",
        "stage": args.stage,
        "tp_size": args.tp_size,
        "batch_size": args.batch_size,
        "prompt_length" if args.stage == "prefill" else "initial_kv_length": args.length,
        "decode_tokens": 1 if args.stage == "prefill" else args.decode_tokens,
        "dtype": args.dtype,
        "warmup_batches": args.warmup,
        "measured_batches": args.repeats,
        "release_policy": "all requests released in the same asyncio turn",
        "prompt_policy": (
            "exact-length token-id prompts with a request-unique first 16-token "
            "block to prevent prefix-cache reuse"),
        "metric": ("time until every request receives its first token"
                   if args.stage == "prefill" else
                   "mean per-request streamed TPOT"),
        "max_allowed_request_start_skew_ms": args.max_start_skew_ms,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "collector": "vllm_fixed_batch_streaming",
        "experiment_type": "fixed_batch_iteration_validation",
        "model": args.model,
        "base_url": args.base_url,
        "configuration": benchmark,
        "deployment": load_deployment(args),
        "metadata": collect_runtime_metadata(
            model_path=args.tokenizer or args.model,
            benchmark=benchmark, project_root=PROJECT_ROOT),
        "warmup": warmups,
        "samples": samples,
        "summary": describe_ms(values) if values else None,
        "valid": (len(values) == args.repeats and
                  max_start_skew <= args.max_start_skew_ms),
        "limitations": [
            "Prefill truth includes first-token sampling and serving overhead.",
            "Decode TPOT spans a growing KV length; --length is its initial KV length.",
            "Concurrent client release cannot prove vLLM admitted every request into one batch.",
        ],
    }
    write_profile(payload, args.output)
    print(f"saved fixed-batch run to {args.output}; valid={payload['valid']}")
    return 0 if payload["valid"] else 2


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
