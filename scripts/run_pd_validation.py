#!/usr/bin/env python3
"""Validate the PD simulator against aggregated real vLLM PD measurements."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pd_simulator import PDConfig, PDSimulator
from core.qwen3_cost_model import LLMCostModel
from core.simulator import RequestGenerator
from scripts.validation_utils import compare_requests, describe, write_json, write_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth", type=Path, required=True,
        help="aggregate.json produced from repeated real PD runs")
    parser.add_argument(
        "--trace", type=Path,
        default=PROJECT_ROOT / "data" / "trace" / "validation_trace.csv")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--operator-profile-dir", type=Path)
    parser.add_argument("--collective-profile-dir", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "validation" / "pd_tp4_1p1d")
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--enable-calibration", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument(
        "--kv-transfer-bw-gb-s", type=float,
        help="override measured bandwidth; defaults to manifest value or 12.5")
    parser.add_argument(
        "--kv-transfer-latency-ms", type=float,
        help="override measured latency; defaults to manifest value or 0.1")
    parser.add_argument("--kv-transfer-concurrency", type=int, default=1)
    parser.add_argument("--simulation-duration-ms", type=float, default=1_000_000.0)
    parser.add_argument("--ttft-sla-ms", type=float, default=500.0)
    parser.add_argument("--tpot-sla-ms", type=float, default=50.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_pd_aggregate(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("collector") != "vllm_pd_proxy_streaming_aggregate":
        raise ValueError(
            "--ground-truth must be a PD vllm_pd_proxy_streaming_aggregate document")
    if not payload.get("deployment_config_sha256"):
        raise ValueError("PD aggregate is missing deployment_config_sha256")
    if not isinstance(payload.get("deployment"), dict):
        raise ValueError("PD aggregate is missing its deployment manifest")
    if not payload.get("requests"):
        raise ValueError("PD aggregate contains no requests")
    return payload


def _required(mapping: Dict[str, Any], key: str, section: str) -> Any:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"deployment.{section} is missing {key}")
    return value


def config_from_deployment(
        deployment: Dict[str, Any], *, chunked_prefill: bool,
        bandwidth_gb_s: float | None = None,
        latency_ms: float | None = None,
        transfer_concurrency: int = 1) -> tuple[PDConfig, Dict[str, Any]]:
    """Build simulation configuration from the audited runtime manifest."""
    prefill = _required(deployment, "prefill", "root")
    decode = _required(deployment, "decode", "root")
    runtime = _required(deployment, "runtime", "root")
    connector = _required(deployment, "kv_connector", "root")
    p_tp = int(_required(prefill, "tensor_parallel_size", "prefill"))
    d_tp = int(_required(decode, "tensor_parallel_size", "decode"))
    p_replicas = int(_required(prefill, "replicas", "prefill"))
    d_replicas = int(_required(decode, "replicas", "decode"))

    measured_bw = connector.get("measured_bandwidth_gb_s")
    measured_latency = connector.get("measured_fixed_latency_ms")
    selected_bw = bandwidth_gb_s if bandwidth_gb_s is not None else measured_bw
    selected_latency = latency_ms if latency_ms is not None else measured_latency
    bandwidth_assumed = selected_bw is None
    latency_assumed = selected_latency is None
    if bandwidth_assumed:
        selected_bw = 12.5
    if latency_assumed:
        selected_latency = 0.1

    config = PDConfig(
        num_prefill_gpus=p_replicas * p_tp,
        prefill_tp=p_tp,
        prefill_num_gpu_blocks=int(_required(
            prefill, "num_gpu_blocks_from_log", "prefill")),
        prefill_max_num_batched_tokens=int(_required(
            prefill, "max_num_batched_tokens", "prefill")),
        prefill_max_num_seqs=int(_required(prefill, "max_num_seqs", "prefill")),
        num_decode_gpus=d_replicas * d_tp,
        decode_tp=d_tp,
        decode_num_gpu_blocks=int(_required(
            decode, "num_gpu_blocks_from_log", "decode")),
        decode_max_num_batched_tokens=int(_required(
            decode, "max_num_batched_tokens", "decode")),
        decode_max_num_seqs=int(_required(decode, "max_num_seqs", "decode")),
        block_size_tokens=int(_required(runtime, "block_size_tokens", "runtime")),
        enable_chunked_prefill=chunked_prefill,
        kv_transfer_mode=str(connector.get("transport", "custom")).lower(),
        kv_transfer_bw_gb_s=float(selected_bw),
        kv_transfer_latency_ms=float(selected_latency),
        kv_transfer_concurrency=transfer_concurrency,
        decode_kv_load_policy=(
            "blocking_receive"
            if connector.get("name") == "P2pNcclConnector"
            else "post_transfer"),
    )
    provenance = {
        "runtime_fields": "derived from ground-truth deployment manifest",
        "kv_transfer_bandwidth_assumed": bandwidth_assumed,
        "kv_transfer_latency_assumed": latency_assumed,
        "decode_kv_load_policy_source": (
            "derived from deployment.kv_connector.name"),
    }
    return config, provenance


def validate_trace_identity(aggregate: Dict[str, Any], requests: Sequence[Any]) -> None:
    expected = aggregate["requests"]
    if len(expected) != len(requests):
        raise ValueError(
            f"trace has {len(requests)} requests but aggregate has {len(expected)}")
    for reference, request in zip(expected, requests):
        values = (request.request_id, request.arrival_time,
                  request.prompt_len, request.output_len)
        wanted = (reference["request_id"], reference["arrival_time_ms"],
                  reference["prompt_len"], reference["output_len"])
        if values[0] != wanted[0] or any(
                abs(float(left) - float(right)) > 1e-6
                for left, right in zip(values[1:], wanted[1:])):
            raise ValueError(
                f"trace differs from aggregate at {reference['request_id']}: "
                f"trace={values}, aggregate={wanted}")


def simulation_rows(simulator: PDSimulator) -> list[dict]:
    return [{
        "request_id": request.request_id,
        "arrival_time_ms": request.arrival_time,
        "prompt_len": request.prompt_len,
        "output_len": request.output_len,
        "generated_tokens": request.generated_tokens,
        "prefill_start_time_ms": request.prefill_start_time,
        "prefill_end_time_ms": request.prefill_end_time,
        "kv_transfer_start_time_ms": request.kv_transfer_start_time,
        "kv_transfer_end_time_ms": request.kv_transfer_end_time,
        "decode_receive_start_time_ms": request.decode_receive_start_time,
        "decode_receive_end_time_ms": request.decode_receive_end_time,
        "first_token_time_ms": request.first_token_time,
        "completion_time_ms": request.completion_time,
        "ttft_ms": request.ttft,
        "tpot_ms": request.tpot,
        "e2e_ms": request.e2e_latency,
        "status": request.pd_status.value,
        "prefill_replica": request.assigned_prefill_replica,
        "decode_replica": request.assigned_decode_replica,
    } for request in sorted(simulator.requests, key=lambda item: item.arrival_time)]


def stage_diagnostics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def durations(end: str, start: str) -> list[float]:
        return [float(row[end]) - float(row[start]) for row in rows]

    return {
        "prefill_service_ms": describe(durations(
            "prefill_end_time_ms", "prefill_start_time_ms")),
        "kv_transfer_ms": describe(durations(
            "kv_transfer_end_time_ms", "kv_transfer_start_time_ms")),
        "post_transfer_to_first_token_ms": describe(durations(
            "first_token_time_ms", "kv_transfer_end_time_ms")),
        "decode_blocking_receive_ms": describe(durations(
            "decode_receive_end_time_ms", "decode_receive_start_time_ms")),
        "decode_after_first_token_ms": describe(durations(
            "completion_time_ms", "first_token_time_ms")),
    }


def main() -> int:
    args = parse_args()
    if args.enable_calibration and not args.calibration_file:
        raise SystemExit("--enable-calibration requires --calibration-file")
    aggregate = load_pd_aggregate(args.ground_truth)
    calibrated_transfer = {}
    if args.enable_calibration and args.calibration_file:
        with args.calibration_file.open("r", encoding="utf-8") as handle:
            calibration_payload = json.load(handle)
        value = calibration_payload.get("pd_transfer", {})
        if isinstance(value, dict):
            calibrated_transfer = value
    bandwidth = args.kv_transfer_bw_gb_s
    latency = args.kv_transfer_latency_ms
    transfer_concurrency = args.kv_transfer_concurrency
    bandwidth_source = "command_line" if bandwidth is not None else None
    latency_source = "command_line" if latency is not None else None
    if bandwidth is None and calibrated_transfer.get(
            "effective_bandwidth_gb_s") is not None:
        bandwidth = float(calibrated_transfer["effective_bandwidth_gb_s"])
        bandwidth_source = "calibration_file_effective_readiness_rate"
    if latency is None and calibrated_transfer.get("fixed_latency_ms") is not None:
        latency = float(calibrated_transfer["fixed_latency_ms"])
        latency_source = "calibration_file"
    if (args.kv_transfer_concurrency == 1 and
            calibrated_transfer.get("concurrency") is not None):
        transfer_concurrency = int(calibrated_transfer["concurrency"])
    config, provenance = config_from_deployment(
        aggregate["deployment"],
        chunked_prefill=not args.disable_chunked_prefill,
        bandwidth_gb_s=bandwidth,
        latency_ms=latency,
        transfer_concurrency=transfer_concurrency)
    if bandwidth_source:
        provenance["kv_transfer_bandwidth_source"] = bandwidth_source
        provenance["kv_transfer_bandwidth_assumed"] = False
    if latency_source:
        provenance["kv_transfer_latency_source"] = latency_source
        provenance["kv_transfer_latency_assumed"] = False

    generator = RequestGenerator(mode="trace", trace_file=str(args.trace))
    trace_requests = generator.generate_requests()
    validate_trace_identity(aggregate, trace_requests)
    model = LLMCostModel(
        str(args.data_dir),
        calibration_file=str(args.calibration_file) if args.calibration_file else None,
        enable_e2e_compensation=args.enable_calibration,
        operator_profile_dir=(str(args.operator_profile_dir)
                              if args.operator_profile_dir else None),
        collective_profile_dir=(str(args.collective_profile_dir)
                                if args.collective_profile_dir else None))
    simulator = PDSimulator(config, model, generator, verbose=args.verbose)
    simulation_summary = simulator.run(
        simulation_duration_ms=args.simulation_duration_ms,
        ttft_sla_ms=args.ttft_sla_ms,
        tpot_sla_ms=args.tpot_sla_ms,
        num_requests=aggregate["request_count"])
    rows = simulation_rows(simulator)
    completed = [row for row in rows if row["status"] == "completed"]
    if len(completed) != aggregate["request_count"]:
        raise RuntimeError(
            f"simulation completed {len(completed)}/{aggregate['request_count']} requests")

    detail, error_summary = compare_requests(aggregate, completed)
    gt_throughput = aggregate["summary"]["throughput"]
    throughput_comparison = {}
    for name, sim_key in (
            ("input_tokens_per_s", "input_throughput_tokens_per_s"),
            ("output_tokens_per_s", "output_throughput_tokens_per_s")):
        gt = float(gt_throughput[name]["mean"])
        simulated = float(simulation_summary[sim_key])
        throughput_comparison[name] = {
            "ground_truth_mean": gt,
            "simulation": simulated,
            "error_pct": 100.0 * (simulated - gt) / gt if gt else 0.0,
        }

    report = {
        "schema_version": 1,
        "experiment_type": "pd_ground_truth_validation",
        "ground_truth": {
            "file": str(args.ground_truth.resolve()),
            "deployment_config_sha256": aggregate["deployment_config_sha256"],
            "repeat_count": aggregate["repeat_count"],
            "request_count": aggregate["request_count"],
            "summary": aggregate["summary"],
        },
        "configuration": asdict(config),
        "configuration_provenance": provenance,
        "trace": str(args.trace.resolve()),
        "calibration": {
            "enabled": args.enable_calibration,
            "file": str(args.calibration_file.resolve())
            if args.calibration_file else None,
        },
        "profiling": {
            "operator_profile_dir": (
                str(args.operator_profile_dir.resolve())
                if args.operator_profile_dir else None),
            "collective_profile_dir": (
                str(args.collective_profile_dir.resolve())
                if args.collective_profile_dir else None),
        },
        "simulation_summary": simulation_summary,
        "error_summary": error_summary,
        "throughput_comparison": throughput_comparison,
        "simulation_stage_diagnostics": stage_diagnostics(completed),
        "limitations": [
            "Real per-request P/D stage timestamps are not exposed by the proxy; stage diagnostics are simulation-only.",
            ("KV-transfer bandwidth uses a fallback assumption because the deployment manifest has no measurement."
             if provenance["kv_transfer_bandwidth_assumed"] else
             "KV-transfer bandwidth comes from the deployment manifest."),
            "This validation trace was not used to fit the selected calibration parameters.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, args.output_dir / "pd_validation_report.json")
    write_rows(completed, args.output_dir / "pd_simulation_requests.csv")
    write_rows(detail, args.output_dir / "pd_request_errors.csv")
    if simulator.batch_history:
        write_rows(simulator.batch_history, args.output_dir / "pd_simulation_batches.csv")
    if simulator.transfer_history:
        write_rows(simulator.transfer_history, args.output_dir / "pd_simulation_transfers.csv")
    if simulator.decode_receive_history:
        write_rows(
            simulator.decode_receive_history,
            args.output_dir / "pd_simulation_decode_receives.csv")

    print(f"completed requests: {len(completed)}/{aggregate['request_count']}")
    for metric, stats in error_summary.items():
        print(f"{metric}: MAPE={stats['mape_pct']:.2f}%, "
              f"MAE={stats['mae_ms']:.2f} ms, "
              f"bias={stats['mean_signed_error_pct']:.2f}%")
    for metric, values in throughput_comparison.items():
        print(f"{metric}: error={values['error_pct']:.2f}%")
    print(f"report: {args.output_dir / 'pd_validation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
