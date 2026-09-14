#!/usr/bin/env python3
"""Collect concurrent, tokenizer-aware vLLM serving ground truth.

This replaces serial wall-clock tests. Requests are launched according to the
trace's absolute arrival timestamps; streaming gives TTFT, while the server's
usage record gives token counts. It works with any OpenAI-compatible vLLM server.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from openai import AsyncOpenAI
from transformers import AutoTokenizer


PROMETHEUS_LINE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)$")

DEFAULT_SCHEDULER_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:iteration_tokens_total",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="Model name exposed by vLLM")
    parser.add_argument("--tokenizer", help="HF tokenizer path; defaults to --model")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--metrics-url",
                        help="optional vLLM Prometheus endpoint, e.g. .../metrics")
    parser.add_argument("--metrics-interval-ms", type=float, default=25.0)
    return parser.parse_args()


def load_trace(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = [{"request_id": f"req_{index + 1}",
                 "arrival_time_ms": float(row["arrival_time_ms"]),
                 "prompt_len": int(row["prompt_len"]),
                 "output_len": int(row["output_len"])}
                for index, row in enumerate(csv.DictReader(handle))]
    if any(rows[i]["arrival_time_ms"] > rows[i + 1]["arrival_time_ms"]
           for i in range(len(rows) - 1)):
        raise ValueError("trace arrival_time_ms must be non-decreasing")
    return rows


def exact_length_prompt(tokenizer, target: int):
    if target <= 0:
        raise ValueError("prompt_len must be positive")

    seed_ids = tokenizer.encode(
        "The quick brown fox describes an inference system. ",
        add_special_tokens=False,
    )
    return (seed_ids * (target // len(seed_ids) + 1))[:target]


def parse_scheduler_metrics(text: str) -> Dict[str, float]:
    """Retain only scheduler gauges and iteration-token histogram series."""
    metrics: Dict[str, float] = {}
    for line in text.splitlines():
        match = PROMETHEUS_LINE.match(line.strip())
        if not match:
            continue
        name = match.group(1)
        retained = (name in DEFAULT_SCHEDULER_METRICS or
                    name.startswith("vllm:iteration_tokens_total_"))
        if not retained:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        if math.isfinite(value):
            metrics[match.group(1) + (match.group(2) or "")] = value
    return metrics


async def collect_metrics(http_client: Any, url: str, interval_ms: float,
                          epoch: float, stop: asyncio.Event) -> List[Dict]:
    samples: List[Dict] = []
    while True:
        sample: Dict[str, Any] = {
            "time_ms": (time.perf_counter() - epoch) * 1000.0,
        }
        try:
            response = await http_client.get(url)
            response.raise_for_status()
            sample.update({
                "status": "success",
                "metrics": parse_scheduler_metrics(response.text),
            })
        except Exception as exc:
            sample.update({
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
        samples.append(sample)
        if stop.is_set():
            break
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_ms / 1000.0)
        except asyncio.TimeoutError:
            pass
    return samples


async def run_one(client: AsyncOpenAI, semaphore: asyncio.Semaphore,
                  tokenizer, model: str, row: Dict, epoch: float) -> Dict:
    await asyncio.sleep(max(epoch + row["arrival_time_ms"] / 1000 - time.perf_counter(), 0))
    prompt_token_ids = exact_length_prompt(
        tokenizer,
        row["prompt_len"],
    )
    actual_prompt_tokens = len(prompt_token_ids)
    async with semaphore:
        started = time.perf_counter()
        first_token_at = None
        last_token_at = None
        usage = None
        error = None
        try:
            stream = await client.completions.create(
                model=model, prompt=prompt_token_ids, max_tokens=row["output_len"],
                temperature=0.0, stream=True,
                stream_options={"include_usage": True},
                extra_body={"request_id": row["request_id"]})
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if chunk.choices and chunk.choices[0].text:
                    now = time.perf_counter()
                    first_token_at = first_token_at or now
                    last_token_at = now
        except Exception as exc:  # retain failures in the ground-truth artifact
            error = f"{type(exc).__name__}: {exc}"
        ended = time.perf_counter()
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or actual_prompt_tokens)
    ttft_ms = ((first_token_at - started) * 1000) if first_token_at else 0.0
    tpot_ms = (((last_token_at - first_token_at) * 1000 / (completion_tokens - 1))
               if first_token_at and last_token_at and completion_tokens > 1 else 0.0)
    return {**row, "actual_prompt_tokens": prompt_tokens,
            "actual_output_tokens": completion_tokens,
            "request_start_ms": (started - epoch) * 1000,
            "ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
            "e2e_ms": (ended - started) * 1000,
            "status": "failed" if error else "success", "error": error}


async def main_async(args: argparse.Namespace):
    if args.metrics_interval_ms <= 0:
        raise ValueError("metrics interval must be positive")
    rows = load_trace(args.trace)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                               trust_remote_code=True)
    
    import httpx

    http_client = httpx.AsyncClient(trust_env=False, timeout=args.timeout)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key,
                         timeout=args.timeout, http_client=http_client)

    # 在设置正式采集时间起点前预热 tokenizer
    max_prompt_len = max(row["prompt_len"] for row in rows)
    exact_length_prompt(tokenizer, max_prompt_len)

    # 客户端和服务端预热
    await client.models.list()

    warmup_prompt = exact_length_prompt(tokenizer, 32)
    await client.completions.create(
        model=args.model,
        prompt=warmup_prompt,
        max_tokens=1,
        temperature=0.0,
        stream=False,
    )

    await asyncio.sleep(1.0)

    semaphore = asyncio.Semaphore(args.max_concurrency)
    epoch = time.perf_counter() + 0.5
    stop = asyncio.Event()
    metrics_task = (asyncio.create_task(collect_metrics(
        http_client, args.metrics_url, args.metrics_interval_ms, epoch, stop
    )) if args.metrics_url else None)
    results = await asyncio.gather(*[
        run_one(client, semaphore, tokenizer, args.model, row, epoch) for row in rows
    ])
    stop.set()
    metric_samples = await metrics_task if metrics_task else []
    await http_client.aclose()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "collector": "vllm_openai_streaming",
        "model": args.model,
        "base_url": args.base_url,
        "trace": str(Path(args.trace).resolve()),
        "collector_config": {
            "max_concurrency": args.max_concurrency,
            "timeout_seconds": args.timeout,
            "metrics_url": args.metrics_url,
            "metrics_interval_ms": args.metrics_interval_ms,
        },
        "requests": results,
        "scheduler_metric_samples": metric_samples,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    successful = sum(item["status"] == "success" for item in results)
    print(f"saved {len(results)} requests ({successful} successful) to {output}")


if __name__ == "__main__":
    asyncio.run(main_async(arguments()))
