"""Align real PD proxy timing logs with collector request records.

The proxy starts the prefill and decode HTTP requests concurrently.  Therefore
``decode_first_byte_ms - prefill_complete_ms`` is not raw network transfer
time: it is the observed post-prefill delay until the decode worker can emit a
token.  This tool keeps that quantity separate and groups nearly simultaneous
first-token events into remote-KV readiness/admission cohorts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple


FIELD_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")
TIMING_FIELDS = (
    "prefill_start_ms", "prefill_complete_ms", "decode_start_ms",
    "decode_headers_ms", "decode_first_byte_ms", "decode_complete_ms",
    "proxy_complete_ms",
)
CONNECTOR_KV_EVENTS = ("recv_start", "recv_complete")
LEGACY_KV_EVENTS = ("wait_start", "recv_finished", "promoted")
KV_EVENTS = CONNECTOR_KV_EVENTS + LEGACY_KV_EVENTS
VLLM_COMPLETION_REQUEST_RE = re.compile(r"^cmpl-(.*)-[0-9]+$")


def canonical_internal_request_id(request_id: str) -> str:
    """Remove vLLM's completion envelope from the proxy transfer id."""
    match = VLLM_COMPLETION_REQUEST_RE.fullmatch(request_id)
    return match.group(1) if match else request_id


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values: Iterable[float]) -> Dict[str, float | int | None]:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "mean": mean(data) if data else None,
        "p50": _percentile(data, 0.50),
        "p90": _percentile(data, 0.90),
        "p99": _percentile(data, 0.99),
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }


def load_timing_log(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if "PD_TIMING" not in line:
            continue
        fields = {match.group(1): match.group(2)
                  for match in FIELD_RE.finditer(line)}
        request_id = fields.get("client_request_id")
        if not request_id:
            raise ValueError(f"missing client_request_id in {path}:{line_number}")
        if request_id in records:
            raise ValueError(f"duplicate PD_TIMING record for {request_id} in {path}")
        missing = [field for field in TIMING_FIELDS if field not in fields]
        if missing:
            raise ValueError(
                f"missing timing fields for {request_id} in {path}: {missing}")
        records[request_id] = {
            "request_id": request_id,
            "internal_request_id": fields.get("internal_request_id", ""),
            **{field: float(fields[field]) for field in TIMING_FIELDS},
        }
        if "proxy_received_monotonic_s" in fields:
            records[request_id]["proxy_received_monotonic_s"] = float(
                fields["proxy_received_monotonic_s"])
    if not records:
        raise ValueError(f"no PD_TIMING records found in {path}")
    return records


def load_kv_event_log(path: Path) -> Dict[str, Dict[str, float]]:
    """Load decode-engine state transitions keyed by internal request id."""
    records: Dict[str, Dict[str, float]] = {}
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if "PD_KV_EVENT" not in line:
            continue
        fields = {match.group(1): match.group(2)
                  for match in FIELD_RE.finditer(line)}
        request_id = fields.get("request_id")
        event = fields.get("event")
        if not request_id or event not in KV_EVENTS or "monotonic_s" not in fields:
            raise ValueError(f"invalid PD_KV_EVENT in {path}:{line_number}")
        request_id = canonical_internal_request_id(request_id)
        events = records.setdefault(request_id, {})
        if event in events:
            raise ValueError(
                f"duplicate {event} for {request_id} in {path}:{line_number}")
        events[event] = float(fields["monotonic_s"])
    if not records:
        raise ValueError(f"no PD_KV_EVENT records found in {path}")
    return records


def _cohorts(rows: List[Dict[str, Any]], threshold_ms: float,
             scenario: str) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["absolute_decode_first_byte_ms"])
    groups: List[List[Dict[str, Any]]] = []
    for row in ordered:
        if (not groups or row["absolute_decode_first_byte_ms"] -
                groups[-1][-1]["absolute_decode_first_byte_ms"] > threshold_ms):
            groups.append([])
        groups[-1].append(row)

    output = []
    for index, group in enumerate(groups, start=1):
        cohort_id = f"{scenario}_cohort_{index}"
        for row in group:
            row["readiness_cohort_id"] = cohort_id
            row["readiness_cohort_size"] = len(group)
        first_times = [row["absolute_decode_first_byte_ms"] for row in group]
        prefill_times = [row["absolute_prefill_complete_ms"] for row in group]
        output.append({
            "scenario": scenario,
            "cohort_id": cohort_id,
            "size": len(group),
            "request_ids": ",".join(row["request_id"] for row in group),
            "first_token_min_ms": min(first_times),
            "first_token_max_ms": max(first_times),
            "first_token_spread_ms": max(first_times) - min(first_times),
            "earliest_prefill_complete_ms": min(prefill_times),
            "latest_prefill_complete_ms": max(prefill_times),
            "max_post_prefill_wait_ms": max(
                row["post_prefill_to_first_byte_ms"] for row in group),
        })
    return output


def analyze_run(scenario: str, run_path: Path, timing_path: Path,
                cohort_threshold_ms: float = 50.0,
                kv_event_path: Path | None = None,
                ) -> Tuple[Dict[str, Any], List[Dict[str, Any]],
                           List[Dict[str, Any]]]:
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    requests = payload.get("requests", [])
    successful = {row["request_id"]: row for row in requests
                  if row.get("status") == "success"}
    timings = load_timing_log(timing_path)
    kv_events = load_kv_event_log(kv_event_path) if kv_event_path else None
    missing = sorted(set(successful) - set(timings))
    extra = sorted(set(timings) - set(successful))
    if missing or extra:
        raise ValueError(
            f"request alignment failed for {scenario}: missing={missing}, extra={extra}")

    rows: List[Dict[str, Any]] = []
    for request_id, request in successful.items():
        timing = timings[request_id]
        arrival = float(request["arrival_time_ms"])
        prefill_service = (timing["prefill_complete_ms"] -
                           timing["prefill_start_ms"])
        post_prefill = (timing["decode_first_byte_ms"] -
                        timing["prefill_complete_ms"])
        row = {
            "scenario": scenario,
            "request_id": request_id,
            "arrival_time_ms": arrival,
            "prompt_len": int(request["prompt_len"]),
            "output_len": int(request["output_len"]),
            "client_ttft_ms": float(request["ttft_ms"]),
            "client_tpot_ms": float(request["tpot_ms"]),
            "client_e2e_ms": float(request["e2e_ms"]),
            "prefill_service_ms": prefill_service,
            "post_prefill_to_first_byte_ms": post_prefill,
            "decode_headers_ms": timing["decode_headers_ms"],
            "client_minus_proxy_first_byte_ms": (
                float(request["ttft_ms"]) - timing["decode_first_byte_ms"]),
            "absolute_prefill_complete_ms": (
                arrival + timing["prefill_complete_ms"]),
            "absolute_decode_first_byte_ms": (
                arrival + timing["decode_first_byte_ms"]),
            "proxy_decode_duration_ms": (
                timing["decode_complete_ms"] - timing["decode_first_byte_ms"]),
            "internal_request_id": timing["internal_request_id"],
        }
        if kv_events is not None:
            origin = timing.get("proxy_received_monotonic_s")
            if origin is None:
                raise ValueError(
                    f"{request_id} has no proxy_received_monotonic_s; restart "
                    "the proxy with the instrumented version")
            internal_id = canonical_internal_request_id(
                timing["internal_request_id"])
            events = kv_events.get(internal_id, {})
            if all(event in events for event in CONNECTOR_KV_EVENTS):
                relative = {
                    event: 1000.0 * (events[event] - origin)
                    for event in CONNECTOR_KV_EVENTS
                }
                row.update({
                    "kv_receive_start_ms": relative["recv_start"],
                    "kv_receive_complete_ms": relative["recv_complete"],
                    "prefill_complete_to_kv_receive_start_ms": (
                        relative["recv_start"] - timing["prefill_complete_ms"]),
                    "kv_receive_duration_ms": (
                        relative["recv_complete"] - relative["recv_start"]),
                    "kv_receive_complete_to_first_byte_ms": (
                        timing["decode_first_byte_ms"] -
                        relative["recv_complete"]),
                })
            elif all(event in events for event in LEGACY_KV_EVENTS):
                relative = {event: 1000.0 * (events[event] - origin)
                            for event in LEGACY_KV_EVENTS}
                row.update({
                    "kv_wait_start_ms": relative["wait_start"],
                    "kv_receive_finished_ms": relative["recv_finished"],
                    "kv_promoted_ms": relative["promoted"],
                    "prefill_complete_to_kv_receive_ms": (
                        relative["recv_finished"] -
                        timing["prefill_complete_ms"]),
                    "kv_receive_to_promotion_ms": (
                        relative["promoted"] - relative["recv_finished"]),
                    "promotion_to_first_byte_ms": (
                        timing["decode_first_byte_ms"] - relative["promoted"]),
                })
            else:
                raise ValueError(
                    f"missing KV events for {request_id} ({internal_id}): "
                    f"expected {CONNECTOR_KV_EVENTS}")
        rows.append(row)

    cohorts = _cohorts(rows, cohort_threshold_ms, scenario)
    post_prefill = [row["post_prefill_to_first_byte_ms"] for row in rows]
    summary = {
        "scenario": scenario,
        "run_file": str(run_path.resolve()),
        "timing_log": str(timing_path.resolve()),
        "deployment_config_sha256": payload.get("deployment_config_sha256"),
        "total_requests": len(requests),
        "successful_requests": len(successful),
        "aligned_timing_records": len(rows),
        "cohort_threshold_ms": cohort_threshold_ms,
        "readiness_cohort_count": len(cohorts),
        "multi_request_cohort_count": sum(row["size"] > 1 for row in cohorts),
        "largest_readiness_cohort": max((row["size"] for row in cohorts), default=0),
        "prefill_service_ms": describe(row["prefill_service_ms"] for row in rows),
        "post_prefill_to_first_byte_ms": describe(post_prefill),
        "decode_headers_ms": describe(row["decode_headers_ms"] for row in rows),
        "client_minus_proxy_first_byte_ms": describe(
            row["client_minus_proxy_first_byte_ms"] for row in rows),
        "post_prefill_wait_over_100ms_ratio": (
            sum(value > 100.0 for value in post_prefill) / len(post_prefill)
            if post_prefill else None),
    }
    if kv_events is not None:
        if "kv_receive_duration_ms" in rows[0]:
            names = (
                "prefill_complete_to_kv_receive_start_ms",
                "kv_receive_duration_ms",
                "kv_receive_complete_to_first_byte_ms",
            )
            summary["kv_event_source"] = "p2p_worker_blocking_receive"
        else:
            names = (
                "prefill_complete_to_kv_receive_ms",
                "kv_receive_to_promotion_ms",
                "promotion_to_first_byte_ms",
            )
            summary["kv_event_source"] = "legacy_scheduler_async_state"
        summary["kv_state_timing"] = {
            name: describe(row[name] for row in rows) for name in names
        }
    return summary, rows, cohorts


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", nargs=3, metavar=("NAME", "RUN_JSON", "LOG"),
        required=True, help="repeat for each named real-PD timing run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort-threshold-ms", type=float, default=50.0)
    parser.add_argument(
        "--kv-event-log", action="append", nargs=2, default=[],
        metavar=("NAME", "DECODE_LOG"),
        help="optional instrumented decode log for a named --input")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cohort_threshold_ms < 0:
        raise ValueError("cohort threshold must be non-negative")
    summaries: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    cohorts: List[Dict[str, Any]] = []
    kv_logs: Dict[str, Path] = {}
    for scenario, path in args.kv_event_log:
        if scenario in kv_logs:
            raise ValueError(f"duplicate --kv-event-log scenario: {scenario}")
        kv_logs[scenario] = Path(path)
    input_names = {item[0] for item in args.input}
    unknown_logs = sorted(set(kv_logs) - input_names)
    if unknown_logs:
        raise ValueError(f"KV event logs have no matching --input: {unknown_logs}")
    for scenario, run_file, timing_log in args.input:
        summary, run_rows, run_cohorts = analyze_run(
            scenario, Path(run_file), Path(timing_log), args.cohort_threshold_ms,
            kv_logs.get(scenario))
        summaries.append(summary)
        requests.extend(run_rows)
        cohorts.extend(run_cohorts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "experiment_type": "pd_remote_kv_readiness_analysis",
        "interpretation": (
            "post_prefill_to_first_byte_ms includes remote-KV readiness and "
            "decode admission; it is not raw interconnect transfer time"),
        "scenarios": summaries,
    }
    (args.output_dir / "pd_timing_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "pd_timing_requests.csv", requests)
    write_csv(args.output_dir / "pd_readiness_cohorts.csv", cohorts)
    for summary in summaries:
        wait = summary["post_prefill_to_first_byte_ms"]
        print(
            f"{summary['scenario']}: n={summary['aligned_timing_records']}, "
            f"wait_mean={wait['mean']:.2f} ms, wait_p90={wait['p90']:.2f} ms, "
            f"cohorts={summary['readiness_cohort_count']}, "
            f"largest={summary['largest_readiness_cohort']}")
    print(f"report: {args.output_dir / 'pd_timing_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
