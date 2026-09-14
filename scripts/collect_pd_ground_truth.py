#!/usr/bin/env python3
"""Collect auditable ground truth through a vLLM PD proxy/router.

The collector never sends requests directly to a prefill or decode worker.
It replays absolute trace arrivals against the OpenAI-compatible proxy, uses
text prompts verified to round-trip to exact token lengths, and optionally
samples worker Prometheus endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


PROMETHEUS_LINE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True,
                        help="PD proxy OpenAI base URL, normally .../v1")
    parser.add_argument("--model", required=True,
                        help="model name exposed by the PD proxy")
    parser.add_argument("--tokenizer", help="HF tokenizer path; defaults to model")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deployment-config", type=Path, required=True,
                        help="JSON manifest describing P/D workers and connector")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--warmup-prompt-len", type=int, default=32)
    # A one-token end-to-end request is a pathological case for vLLM 0.19's
    # P2pNcclConnector: both proxy phases become max_tokens=1 and the TP decode
    # workers can stall in sample_tokens. Keep PD warmup representative.
    parser.add_argument("--warmup-output-len", type=int, default=8)
    parser.add_argument("--allow-early-stop", action="store_true",
                        help="do not require actual_output_tokens == output_len")
    parser.add_argument("--repeat-id", default="repeat_1")
    parser.add_argument("--component-metrics-url", action="append", default=[],
                        metavar="NAME=URL",
                        help="repeat for prefill/decode /metrics endpoints")
    parser.add_argument("--metrics-interval-ms", type=float, default=500.0)
    parser.add_argument("--metrics-prefix", action="append",
                        default=["vllm:", "vllm_"],
                        help="Prometheus metric prefix to retain")
    return parser.parse_args()


def load_trace(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [{
            "request_id": f"req_{index + 1}",
            "arrival_time_ms": float(row["arrival_time_ms"]),
            "prompt_len": int(row["prompt_len"]),
            "output_len": int(row["output_len"]),
        } for index, row in enumerate(csv.DictReader(handle))]
    if not rows:
        raise ValueError("trace must contain at least one request")
    if any(row["arrival_time_ms"] < 0 or row["prompt_len"] <= 0 or
           row["output_len"] <= 0 for row in rows):
        raise ValueError("trace lengths and arrival times are invalid")
    if any(rows[index]["arrival_time_ms"] > rows[index + 1]["arrival_time_ms"]
           for index in range(len(rows) - 1)):
        raise ValueError("trace arrival_time_ms must be non-decreasing")
    return rows


def load_deployment(path: Path) -> Tuple[Dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("deployment config must be a JSON object")
    required = {"vllm_version", "prefill", "decode", "router", "kv_connector"}
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"deployment config is missing fields: {missing}")
    return value, hashlib.sha256(raw).hexdigest()


def parse_metrics_endpoints(values: Sequence[str]) -> Dict[str, str]:
    endpoints: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("component metrics endpoint must use NAME=URL")
        name, url = (part.strip() for part in value.split("=", 1))
        if not name or not url or name in endpoints:
            raise ValueError(f"invalid or duplicate metrics endpoint: {value}")
        endpoints[name] = url
    return endpoints


def exact_length_prompt(tokenizer: Any, target: int) -> List[int]:
    if target <= 0:
        raise ValueError("prompt length must be positive")
    seed = tokenizer.encode(
        "The quick brown fox describes a distributed inference system. ",
        add_special_tokens=False)
    if not seed:
        raise ValueError("tokenizer returned an empty seed")
    return (seed * (target // len(seed) + 1))[:target]


def exact_length_prompt_text(tokenizer: Any, target: int) -> str:
    """Return text whose server-side tokenization is exactly ``target`` long.

    vLLM 0.19's P2pNcclConnector can stall with OpenAI ``prompt=list[int]``
    requests under tensor parallelism. Sending text follows the reference PD
    path; the round-trip check preserves the experiment's exact-length rule.
    """
    if target <= 0:
        raise ValueError("prompt length must be positive")

    # Leading-space words are single, stable tokens in common BPE/tokenizer
    # vocabularies and do not merge across repetitions. Test rather than assume:
    # model/tokenizer variants are allowed as long as the final count is exact.
    units = (
        " x", " a", " the", " hello", " test", " token", " 0",
        " z", "\n", ".", "!",
    )
    observed: Dict[str, int] = {}
    for unit in units:
        candidate = unit * target
        actual = len(tokenizer.encode(candidate, add_special_tokens=False))
        observed[repr(unit)] = actual
        if actual == target:
            return candidate

    raise ValueError(
        "could not construct an exact-length text prompt for this tokenizer: "
        f"expected {target}; candidate counts={observed}")


def parse_prometheus(text: str, prefixes: Sequence[str]) -> Dict[str, float]:
    """Retain numeric vLLM series without depending on metric name versions."""
    metrics: Dict[str, float] = {}
    for line in text.splitlines():
        match = PROMETHEUS_LINE.match(line.strip())
        if not match or not any(match.group(1).startswith(p) for p in prefixes):
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        if math.isfinite(value):
            metrics[match.group(1) + (match.group(2) or "")] = value
    return metrics


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


async def collect_metrics(http_client: Any, endpoints: Dict[str, str],
                          prefixes: Sequence[str], interval_ms: float,
                          epoch: float, stop: asyncio.Event) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    while True:
        sample: Dict[str, Any] = {
            "time_ms": (time.perf_counter() - epoch) * 1000.0,
            "components": {},
        }
        for name, url in endpoints.items():
            try:
                response = await http_client.get(url)
                response.raise_for_status()
                sample["components"][name] = {
                    "status": "success",
                    "metrics": parse_prometheus(response.text, prefixes),
                }
            except Exception as exc:
                sample["components"][name] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        samples.append(sample)
        if stop.is_set():
            break
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_ms / 1000.0)
        except asyncio.TimeoutError:
            pass
    return samples


async def run_one(client: Any, semaphore: asyncio.Semaphore, tokenizer: Any,
                  model: str, row: Dict[str, Any], epoch: float,
                  allow_early_stop: bool) -> Dict[str, Any]:
    scheduled = epoch + row["arrival_time_ms"] / 1000.0
    await asyncio.sleep(max(scheduled - time.perf_counter(), 0.0))
    released = time.perf_counter()
    prompt = exact_length_prompt_text(tokenizer, row["prompt_len"])
    async with semaphore:
        acquired = time.perf_counter()
        first_token_at = None
        last_token_at = None
        usage = None
        error = None
        try:
            stream = await client.completions.create(
                model=model, prompt=prompt, max_tokens=row["output_len"],
                temperature=0.0, stream=True,
                stream_options={"include_usage": True},
                extra_body={
                    "request_id": row["request_id"],
                    "ignore_eos": not allow_early_stop,
                })
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if chunk.choices and chunk.choices[0].text:
                    now = time.perf_counter()
                    first_token_at = first_token_at or now
                    last_token_at = now
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        ended = time.perf_counter()

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    if error is None and first_token_at is None and row["output_len"] > 0:
        error = "response contained no output token"
    if error is None and prompt_tokens != row["prompt_len"]:
        error = (f"prompt token mismatch: expected {row['prompt_len']}, "
                 f"received {prompt_tokens}")
    if (error is None and not allow_early_stop and
            output_tokens != row["output_len"]):
        error = (f"output token mismatch: expected {row['output_len']}, "
                 f"received {output_tokens}")
    ttft = ((first_token_at - acquired) * 1000.0
            if first_token_at is not None else None)
    tpot = (((last_token_at - first_token_at) * 1000.0 / (output_tokens - 1))
            if first_token_at is not None and last_token_at is not None and
            output_tokens > 1 else 0.0 if output_tokens == 1 else None)
    return {
        **row,
        "actual_prompt_tokens": prompt_tokens,
        "actual_output_tokens": output_tokens,
        "scheduled_time_ms": row["arrival_time_ms"],
        "request_start_ms": (acquired - epoch) * 1000.0,
        "client_backpressure_ms": (acquired - released) * 1000.0,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "e2e_ms": (ended - acquired) * 1000.0,
        "status": "failed" if error else "success",
        "error": error,
    }


async def warmup(client: Any, tokenizer: Any, args: argparse.Namespace) -> None:
    prompt = exact_length_prompt_text(tokenizer, args.warmup_prompt_len)
    for _ in range(args.warmup_requests):
        await client.completions.create(
            model=args.model, prompt=prompt,
            max_tokens=args.warmup_output_len, temperature=0.0,
            stream=False, extra_body={"ignore_eos": True})


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [row for row in rows if row["status"] == "success"]
    result: Dict[str, Any] = {
        "total_requests": len(rows),
        "successful_requests": len(successful),
        "failed_requests": len(rows) - len(successful),
        "success_ratio": len(successful) / len(rows) if rows else 0.0,
    }
    for name in ("ttft_ms", "tpot_ms", "e2e_ms", "client_backpressure_ms"):
        values = [float(row[name]) for row in successful if row[name] is not None]
        result[name] = {
            "mean": sum(values) / len(values) if values else None,
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p99": percentile(values, 99),
            "max": max(values) if values else None,
        }
    return result


async def main_async(args: argparse.Namespace) -> int:
    if args.max_concurrency <= 0 or args.timeout <= 0:
        raise ValueError("max concurrency and timeout must be positive")
    if args.warmup_requests < 0 or args.metrics_interval_ms <= 0:
        raise ValueError("warmup count must be non-negative and interval positive")
    if args.warmup_requests and args.warmup_output_len < 2:
        raise ValueError(
            "PD warmup output length must be at least 2 for vLLM 0.19")
    rows = load_trace(args.trace)
    deployment, deployment_sha256 = load_deployment(args.deployment_config)
    endpoints = parse_metrics_endpoints(args.component_metrics_url)

    import httpx
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True)
    exact_length_prompt(tokenizer, max(row["prompt_len"] for row in rows))
    http_client = httpx.AsyncClient(trust_env=False, timeout=args.timeout)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key,
                         timeout=args.timeout, http_client=http_client)
    # The reference vLLM PD proxy exposes /v1/completions but not /v1/models.
    # A successful warmup is therefore the end-to-end health check.
    await warmup(client, tokenizer, args)
    await asyncio.sleep(1.0)

    epoch = time.perf_counter() + 0.5
    stop = asyncio.Event()
    metrics_task = (asyncio.create_task(collect_metrics(
        http_client, endpoints, args.metrics_prefix,
        args.metrics_interval_ms, epoch, stop)) if endpoints else None)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    results = await asyncio.gather(*[
        run_one(client, semaphore, tokenizer, args.model, row, epoch,
                args.allow_early_stop) for row in rows])
    stop.set()
    metric_samples = await metrics_task if metrics_task else []
    await http_client.aclose()

    payload = {
        "schema_version": 2,
        "collector": "vllm_pd_proxy_streaming",
        "experiment_type": "pd_ground_truth",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repeat_id": args.repeat_id,
        "model": args.model,
        "base_url": args.base_url,
        "trace": str(args.trace.resolve()),
        "deployment_config": str(args.deployment_config.resolve()),
        "deployment_config_sha256": deployment_sha256,
        "deployment": deployment,
        "collector_config": {
            "max_concurrency": args.max_concurrency,
            "timeout_seconds": args.timeout,
            "warmup_requests": args.warmup_requests,
            "allow_early_stop": args.allow_early_stop,
            "metrics_interval_ms": args.metrics_interval_ms,
            "component_metrics_urls": endpoints,
        },
        "summary": summarize(results),
        "requests": results,
        "component_metric_samples": metric_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = payload["summary"]["failed_requests"]
    print(f"saved {len(results)} requests ({failed} failed) to {args.output}")
    if failed:
        print("collection is invalid: inspect request errors before retrying")
        return 2
    return 0


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
