#!/usr/bin/env python3
"""Fit PD-specific stage factors using independent PD calibration traces."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pd_simulator import PDSimulator
from core.qwen3_cost_model import LLMCostModel
from core.simulator import RequestGenerator
from scripts.fit_calibration import discover_scenarios
from scripts.fit_shape_calibration import coordinate_search
from scripts.run_pd_validation import (
    config_from_deployment,
    simulation_rows,
    validate_trace_identity,
)
from scripts.validation_utils import compare_requests, write_json


PARAMETER_RANGES = {
    "prefill_factor": (0.50, 2.00),
    "decode_base_factor": (0.50, 1.50),
    "batch_exponent": (-0.50, 0.50),
    "kv_exponent": (-0.35, 0.35),
    "iteration_overhead_ms": (0.0, 20.0),
    # Effective end-to-end connector readiness cost, not raw PCIe line rate.
    "transfer_ms_per_128mb": (5.0, 256.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-root", type=Path, default=(
        PROJECT_ROOT / "data" / "ground_truth_pd" / "calibration_tp4_1p1d"))
    parser.add_argument("--trace-dir", type=Path, default=(
        PROJECT_ROOT / "data" / "trace" / "calibration"))
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-config", type=Path, default=(
        PROJECT_ROOT / "configs" / "calibration_pd_tp4_1p1d.json"))
    parser.add_argument("--output-report", type=Path, default=(
        PROJECT_ROOT / "data" / "calibration" / "pd_tp4_1p1d" /
        "fit_report.json"))
    parser.add_argument("--kv-transfer-bw-gb-s", type=float)
    parser.add_argument("--kv-transfer-latency-ms", type=float)
    parser.add_argument("--kv-transfer-concurrency", type=int, default=1)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--simulation-duration-ms", type=float,
                        default=1_000_000.0)
    parser.add_argument("--reference-batch-size", type=float, default=8.0)
    parser.add_argument("--reference-kv-len", type=float, default=1024.0)
    parser.add_argument("--points", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def candidate_payload(tp_size: int, args: argparse.Namespace,
                      parameters: dict) -> dict:
    transfer_ms = parameters["transfer_ms_per_128mb"]
    return {
        "schema_version": 2,
        "prefill": {str(tp_size): parameters["prefill_factor"]},
        "decode": {str(tp_size): {
            "base_factor": parameters["decode_base_factor"],
            "reference_batch_size": args.reference_batch_size,
            "reference_kv_len": args.reference_kv_len,
            "batch_exponent": parameters["batch_exponent"],
            "kv_exponent": parameters["kv_exponent"],
            "min_factor": 0.20,
            "max_factor": 2.50,
        }},
        "iteration_overhead_ms": {
            str(tp_size): parameters["iteration_overhead_ms"]},
        "pd_transfer": {
            "effective_bandwidth_gb_s": 128.0 / transfer_ms,
            "fixed_latency_ms": (
                args.kv_transfer_latency_ms
                if args.kv_transfer_latency_ms is not None else 0.1),
            "concurrency": args.kv_transfer_concurrency,
            "interpretation": (
                "effective remote-KV readiness rate including connector and "
                "receiver scheduling; not raw interconnect bandwidth"),
        },
    }


def run_scenario(args: argparse.Namespace, scenario: dict,
                 candidate_file: Path,
                 parameters: dict) -> tuple[list[dict], dict, dict]:
    aggregate = scenario["aggregate"]
    deployment = aggregate.get("deployment")
    if not isinstance(deployment, dict):
        raise ValueError(f"{scenario['name']} has no deployment manifest")
    config, provenance = config_from_deployment(
        deployment,
        chunked_prefill=not args.disable_chunked_prefill,
        bandwidth_gb_s=(
            args.kv_transfer_bw_gb_s
            if args.kv_transfer_bw_gb_s is not None else
            128.0 / parameters["transfer_ms_per_128mb"]),
        latency_ms=args.kv_transfer_latency_ms,
        transfer_concurrency=args.kv_transfer_concurrency)
    model = LLMCostModel(
        str(args.data_dir), calibration_file=str(candidate_file),
        enable_e2e_compensation=True)
    generator = RequestGenerator(mode="trace", trace_file=str(scenario["trace"]))
    validate_trace_identity(aggregate, generator.generate_requests())
    simulator = PDSimulator(config, model, generator)
    summary = simulator.run(
        simulation_duration_ms=args.simulation_duration_ms,
        num_requests=aggregate["request_count"])
    rows = simulation_rows(simulator)
    completed = [row for row in rows if row["status"] == "completed"]
    if len(completed) != aggregate["request_count"]:
        raise RuntimeError(
            f"{scenario['name']}: completed {len(completed)}/"
            f"{aggregate['request_count']} requests")
    return completed, summary, provenance


def main() -> int:
    args = parse_args()
    scenarios = discover_scenarios(args.ground_truth_root, args.trace_dir)
    collectors = {item["aggregate"].get("collector") for item in scenarios}
    if collectors != {"vllm_pd_proxy_streaming_aggregate"}:
        raise ValueError(f"PD fitter received non-PD collectors: {collectors}")
    hashes = {item["aggregate"].get("deployment_config_sha256")
              for item in scenarios}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("all PD calibration scenarios must share one deployment hash")
    tp_sizes = {
        int(item["aggregate"]["deployment"]["decode"]["tensor_parallel_size"])
        for item in scenarios
    }
    if len(tp_sizes) != 1:
        raise ValueError(f"PD calibration scenarios have mixed TP sizes: {tp_sizes}")
    tp_size = next(iter(tp_sizes))

    cache: dict[tuple, dict] = {}
    with tempfile.TemporaryDirectory(
            prefix="llm-simulator-pd-calibration-") as temp_dir:
        candidate_file = Path(temp_dir) / "candidate.json"

        def evaluate(parameters: dict) -> float:
            key = tuple(round(parameters[name], 8) for name in PARAMETER_RANGES)
            if key in cache:
                return cache[key]["objective"]
            candidate_file.write_text(
                json.dumps(candidate_payload(tp_size, args, parameters)),
                encoding="utf-8")
            scenario_results = []
            provenance = None
            for scenario in scenarios:
                rows, _, provenance = run_scenario(
                    args, scenario, candidate_file, parameters)
                _, errors = compare_requests(scenario["aggregate"], rows)
                scenario_results.append({
                    "name": scenario["name"],
                    "request_count": scenario["aggregate"]["request_count"],
                    "ttft_mape_pct": errors["ttft_ms"]["mape_pct"],
                    "ttft_bias_pct": errors["ttft_ms"]["mean_signed_error_pct"],
                    "tpot_mape_pct": errors["tpot_ms"]["mape_pct"],
                    "tpot_bias_pct": errors["tpot_ms"]["mean_signed_error_pct"],
                    "e2e_mape_pct": errors["e2e_ms"]["mape_pct"],
                    "e2e_bias_pct": errors["e2e_ms"]["mean_signed_error_pct"],
                })
            macro_ttft = statistics.fmean(
                item["ttft_mape_pct"] for item in scenario_results)
            macro_tpot = statistics.fmean(
                item["tpot_mape_pct"] for item in scenario_results)
            macro_e2e = statistics.fmean(
                item["e2e_mape_pct"] for item in scenario_results)
            objective = 0.35 * macro_ttft + macro_tpot + 0.15 * macro_e2e
            cache[key] = {
                "objective": objective,
                "macro_ttft_mape_pct": macro_ttft,
                "macro_tpot_mape_pct": macro_tpot,
                "macro_e2e_mape_pct": macro_e2e,
                "scenarios": scenario_results,
                "parameters": dict(parameters),
                "transfer_provenance": provenance,
            }
            print(
                f"eval={len(cache):03d} objective={objective:.3f} "
                f"prefill={parameters['prefill_factor']:.4f} "
                f"decode={parameters['decode_base_factor']:.4f} "
                f"batch_exp={parameters['batch_exponent']:.4f} "
                f"kv_exp={parameters['kv_exponent']:.4f} "
                f"overhead={parameters['iteration_overhead_ms']:.3f} "
                f"xfer128={parameters['transfer_ms_per_128mb']:.3f}")
            return objective

        starts = [
            {"prefill_factor": 1.0, "decode_base_factor": 1.0,
             "batch_exponent": 0.0, "kv_exponent": 0.0,
             "iteration_overhead_ms": 0.0,
             "transfer_ms_per_128mb": 10.24},
            {"prefill_factor": 1.25, "decode_base_factor": 0.9,
             "batch_exponent": 0.0, "kv_exponent": 0.0,
             "iteration_overhead_ms": 5.0,
             "transfer_ms_per_128mb": 85.0},
            {"prefill_factor": 1.5, "decode_base_factor": 1.1,
             "batch_exponent": -0.15, "kv_exponent": 0.1,
             "iteration_overhead_ms": 3.0,
             "transfer_ms_per_128mb": 128.0},
        ]
        best, objective, history = coordinate_search(
            evaluate, starts, PARAMETER_RANGES,
            points=args.points, rounds=args.rounds)

    best_key = tuple(round(best[name], 8) for name in PARAMETER_RANGES)
    selected = cache[best_key]
    output_config = candidate_payload(tp_size, args, best)
    output_config["metadata"] = {
        "method": "multi-start coordinate search",
        "objective": "0.35 * macro TTFT MAPE + macro TPOT MAPE + 0.15 * macro E2E MAPE",
        "fit_dataset": "independent real PD calibration traces only",
        "validation_policy": "validation_trace.csv and holdout traces are excluded",
        "deployment_config_sha256": next(iter(hashes)),
        "transfer_fit_policy": (
            "effective remote-KV readiness rate fitted jointly; this is not "
            "reported as physical NCCL bandwidth"),
    }
    write_json(output_config, args.output_config)
    report = {
        "schema_version": 1,
        "experiment_type": "pd_stage_calibration",
        "selected": selected,
        "evaluation_count": len(cache),
        "parameter_ranges": PARAMETER_RANGES,
        "search_history": history,
        "sources": [{
            "name": scenario["name"],
            "trace": str(scenario["trace"]),
            "runs": [str(path) for path in scenario["runs"]],
        } for scenario in scenarios],
        "holdout_policy": (
            "Freeze this config before rerunning validation_trace.csv. "
            "No validation or holdout request is used during fitting."),
    }
    write_json(report, args.output_report)
    print(f"selected parameters: {best}")
    print(f"objective: {objective:.3f}")
    print(f"calibration config: {args.output_config}")
    print(f"fit report: {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
