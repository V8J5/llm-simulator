"""Pure helpers for repeated vLLM ground-truth aggregation and comparison."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


LATENCY_METRICS = ("ttft_ms", "tpot_ms", "e2e_ms")
IDENTITY_FIELDS = ("arrival_time_ms", "prompt_len", "output_len")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(values: Sequence[float]) -> Dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return {key: 0.0 for key in
                ("mean", "std", "min", "p50", "p90", "p99", "max")}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def load_vllm_run(path: Path) -> Dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("requests"), list):
        raise ValueError(f"invalid vLLM ground-truth document: {path}")
    supported = {"vllm_openai_streaming", "vllm_pd_proxy_streaming"}
    if payload.get("collector") not in supported:
        raise ValueError(f"unsupported collector in {path}: {payload.get('collector')}")
    if not payload["requests"]:
        raise ValueError(f"ground-truth run is empty: {path}")
    payload["_source_path"] = str(path.resolve())
    return payload


def _request_map(run: Dict) -> Dict[str, Dict]:
    result = {}
    for request in run["requests"]:
        request_id = request.get("request_id")
        if not request_id or request_id in result:
            raise ValueError(f"missing or duplicate request_id: {request_id!r}")
        if request.get("status") != "success":
            raise ValueError(f"request {request_id} is not successful")
        for key in (*IDENTITY_FIELDS, *LATENCY_METRICS):
            if request.get(key) is None:
                raise ValueError(f"request {request_id} is missing {key}")
        result[request_id] = request
    return result


def _run_throughput(requests: Sequence[Dict]) -> Dict[str, float]:
    origin = min(float(request["arrival_time_ms"]) for request in requests)
    completion = max(float(request.get("request_start_ms", request["arrival_time_ms"]))
                     + float(request["e2e_ms"]) for request in requests)
    duration_ms = max(completion - origin, 0.0)
    duration_s = duration_ms / 1000.0
    input_tokens = sum(int(request.get("actual_prompt_tokens", request["prompt_len"]))
                       for request in requests)
    output_tokens = sum(int(request.get("actual_output_tokens", request["output_len"]))
                        for request in requests)
    return {
        "duration_ms": duration_ms,
        "input_tokens_per_s": input_tokens / duration_s if duration_s else 0.0,
        "output_tokens_per_s": output_tokens / duration_s if duration_s else 0.0,
    }


def aggregate_vllm_runs(paths: Sequence[Path]) -> Dict:
    if len(paths) < 2:
        raise ValueError("at least two repeated runs are required")
    runs = [load_vllm_run(Path(path)) for path in paths]
    maps = [_request_map(run) for run in runs]
    ordered_ids = [request["request_id"] for request in runs[0]["requests"]]
    expected_ids = set(ordered_ids)
    for index, request_map in enumerate(maps[1:], start=2):
        if set(request_map) != expected_ids:
            missing = sorted(expected_ids - set(request_map))
            extra = sorted(set(request_map) - expected_ids)
            raise ValueError(f"run {index} request IDs differ; missing={missing}, extra={extra}")

    schema_versions = {run.get("schema_version") for run in runs}
    collectors = {run.get("collector") for run in runs}
    models = {run.get("model") for run in runs}
    traces = {Path(str(run.get("trace", ""))).name for run in runs}
    deployments = {run.get("deployment_config_sha256") for run in runs}
    if (len(schema_versions) != 1 or len(collectors) != 1 or
            len(models) != 1 or len(traces) != 1 or len(deployments) != 1):
        raise ValueError(
            "schema_version, collector, model, trace, and deployment must match")

    aggregated_requests = []
    for request_id in ordered_ids:
        samples = [request_map[request_id] for request_map in maps]
        reference = samples[0]
        for sample in samples[1:]:
            for field in IDENTITY_FIELDS:
                if not math.isclose(float(sample[field]), float(reference[field]),
                                    rel_tol=0.0, abs_tol=1e-6):
                    raise ValueError(f"{request_id} has inconsistent {field}")
        row = {
            "request_id": request_id,
            "arrival_time_ms": float(reference["arrival_time_ms"]),
            "prompt_len": int(reference["prompt_len"]),
            "output_len": int(reference["output_len"]),
            "actual_prompt_tokens": int(reference.get(
                "actual_prompt_tokens", reference["prompt_len"])),
            "actual_output_tokens": int(reference.get(
                "actual_output_tokens", reference["output_len"])),
            "repeat_count": len(samples),
        }
        start_values = [float(sample.get("request_start_ms", sample["arrival_time_ms"]))
                        for sample in samples]
        row["request_start_ms_mean"] = statistics.fmean(start_values)
        for metric in LATENCY_METRICS:
            stats = describe([sample[metric] for sample in samples])
            for statistic, value in stats.items():
                row[f"{metric}_{statistic}"] = value
        aggregated_requests.append(row)

    run_summaries = []
    throughputs = []
    for run, request_map in zip(runs, maps):
        requests = [request_map[request_id] for request_id in ordered_ids]
        throughput = _run_throughput(requests)
        throughputs.append(throughput)
        run_summaries.append({
            "source": run["_source_path"],
            "request_count": len(requests),
            "latency": {metric: describe([request[metric] for request in requests])
                        for metric in LATENCY_METRICS},
            "throughput": throughput,
        })

    aggregate_latency = {
        metric: describe([request[f"{metric}_mean"]
                          for request in aggregated_requests])
        for metric in LATENCY_METRICS
    }
    repeatability = {}
    for metric in LATENCY_METRICS:
        run_means = [summary["latency"][metric]["mean"] for summary in run_summaries]
        stats = describe(run_means)
        stats["cv_pct"] = (100.0 * stats["std"] / stats["mean"]
                           if stats["mean"] else 0.0)
        repeatability[metric] = stats

    throughput_summary = {
        key: describe([throughput[key] for throughput in throughputs])
        for key in ("duration_ms", "input_tokens_per_s", "output_tokens_per_s")
    }
    return {
        "schema_version": runs[0].get("schema_version", 1),
        "collector": ("vllm_pd_proxy_streaming_aggregate"
                      if runs[0].get("collector") == "vllm_pd_proxy_streaming"
                      else "vllm_openai_streaming_aggregate"),
        "model": runs[0].get("model"),
        "trace": runs[0].get("trace"),
        "deployment_config_sha256": runs[0].get("deployment_config_sha256"),
        "deployment": runs[0].get("deployment"),
        "repeat_count": len(runs),
        "source_files": [run["_source_path"] for run in runs],
        "request_count": len(aggregated_requests),
        "requests": aggregated_requests,
        "summary": {
            "latency": aggregate_latency,
            "throughput": throughput_summary,
            "repeatability": repeatability,
            "runs": run_summaries,
        },
    }


def write_json(payload: Dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_rows(rows: Sequence[Dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _error_stats(gt_values: Sequence[float], sim_values: Sequence[float]) -> Dict:
    absolute = [abs(sim - gt) for gt, sim in zip(gt_values, sim_values)]
    signed_pct = [100.0 * (sim - gt) / gt if gt else 0.0
                  for gt, sim in zip(gt_values, sim_values)]
    squared = [(sim - gt) ** 2 for gt, sim in zip(gt_values, sim_values)]
    smape = [200.0 * abs(sim - gt) / (abs(gt) + abs(sim))
             if gt or sim else 0.0 for gt, sim in zip(gt_values, sim_values)]
    return {
        "mae_ms": statistics.fmean(absolute),
        "rmse_ms": math.sqrt(statistics.fmean(squared)),
        "mape_pct": statistics.fmean(abs(value) for value in signed_pct),
        "smape_pct": statistics.fmean(smape),
        "mean_signed_error_pct": statistics.fmean(signed_pct),
        "max_absolute_error_pct": max(abs(value) for value in signed_pct),
        "ground_truth": describe(gt_values),
        "simulation": describe(sim_values),
    }


def compare_requests(aggregate: Dict, simulation_rows: Sequence[Dict]) -> Tuple[List[Dict], Dict]:
    gt_map = {request["request_id"]: request for request in aggregate["requests"]}
    sim_map = {request["request_id"]: request for request in simulation_rows}
    if set(gt_map) != set(sim_map):
        raise ValueError("ground-truth and simulation request IDs do not match")
    detail = []
    for request in aggregate["requests"]:
        request_id = request["request_id"]
        sim = sim_map[request_id]
        row = {
            "request_id": request_id,
            "arrival_time_ms": request["arrival_time_ms"],
            "prompt_len": request["prompt_len"],
            "output_len": request["output_len"],
        }
        for metric in LATENCY_METRICS:
            gt_value = float(request[f"{metric}_mean"])
            sim_value = float(sim[metric])
            row[f"gt_{metric}"] = gt_value
            row[f"sim_{metric}"] = sim_value
            row[f"{metric}_absolute_error_ms"] = abs(sim_value - gt_value)
            row[f"{metric}_error_pct"] = (
                100.0 * (sim_value - gt_value) / gt_value if gt_value else 0.0)
        detail.append(row)
    summary = {}
    for metric in LATENCY_METRICS:
        summary[metric] = _error_stats(
            [row[f"gt_{metric}"] for row in detail],
            [row[f"sim_{metric}"] for row in detail],
        )
    return detail, summary
