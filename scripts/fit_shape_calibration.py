#!/usr/bin/env python3
"""Fit shape-aware TP4 Decode residuals on calibration workloads only."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fit_calibration import discover_scenarios, run_scenario
from scripts.validation_utils import compare_requests, write_json


PARAMETER_RANGES = {
    "base_factor": (0.35, 1.10),
    "batch_exponent": (-0.50, 0.50),
    "kv_exponent": (-0.35, 0.35),
    "iteration_overhead_ms": (0.0, 20.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-root", type=Path, default=(
        PROJECT_ROOT / "data" / "ground_truth_co_located" / "calibration_tp4"))
    parser.add_argument("--trace-dir", type=Path, default=(
        PROJECT_ROOT / "data" / "trace" / "calibration"))
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-config", type=Path, default=(
        PROJECT_ROOT / "configs" / "calibration_tp4_shape.json"))
    parser.add_argument("--output-report", type=Path, default=(
        PROJECT_ROOT / "data" / "calibration" / "tp4" /
        "shape_fit_report.json"))
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--num-gpu-blocks", type=int, default=22136)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--simulation-duration-ms", type=float,
                        default=1_000_000.0)
    parser.add_argument("--reference-batch-size", type=float, default=8.0)
    parser.add_argument("--reference-kv-len", type=float, default=1024.0)
    parser.add_argument("--points", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def coordinate_search(
        objective: Callable[[dict], float], starts: Sequence[dict],
        ranges: dict[str, tuple[float, float]], points: int = 7,
        rounds: int = 3) -> tuple[dict, float, list[dict]]:
    """Multi-start deterministic coordinate search for a discontinuous simulator."""
    if points < 3 or rounds < 1 or not starts:
        raise ValueError("invalid coordinate-search parameters")
    history = []
    global_best = None
    global_score = float("inf")
    for start_index, start in enumerate(starts):
        current = {
            name: min(max(float(start[name]), low), high)
            for name, (low, high) in ranges.items()
        }
        local_ranges = dict(ranges)
        score = objective(current)
        history.append({"start": start_index, "round": -1,
                        "parameters": dict(current), "objective": score})
        for round_index in range(rounds):
            for name in ranges:
                low, high = local_ranges[name]
                step = (high - low) / (points - 1)
                candidates = [low + index * step for index in range(points)]
                best_value, best_score = current[name], score
                for value in candidates:
                    candidate = dict(current)
                    candidate[name] = value
                    candidate_score = objective(candidate)
                    if candidate_score < best_score:
                        best_value, best_score = value, candidate_score
                current[name], score = best_value, best_score
                original_low, original_high = ranges[name]
                local_ranges[name] = (
                    max(original_low, best_value - step),
                    min(original_high, best_value + step),
                )
            history.append({"start": start_index, "round": round_index,
                            "parameters": dict(current), "objective": score})
        if score < global_score:
            global_best, global_score = dict(current), score
    return global_best, global_score, history


def candidate_payload(args: argparse.Namespace, parameters: dict) -> dict:
    return {
        "schema_version": 2,
        "prefill": {str(args.tp_size): 1.0},
        "decode": {str(args.tp_size): {
            "base_factor": parameters["base_factor"],
            "reference_batch_size": args.reference_batch_size,
            "reference_kv_len": args.reference_kv_len,
            "batch_exponent": parameters["batch_exponent"],
            "kv_exponent": parameters["kv_exponent"],
            "min_factor": 0.20,
            "max_factor": 2.00,
        }},
        "iteration_overhead_ms": {
            str(args.tp_size): parameters["iteration_overhead_ms"]},
    }


def main() -> int:
    args = parse_args()
    scenarios = discover_scenarios(args.ground_truth_root, args.trace_dir)
    cache: dict[tuple, dict] = {}
    with tempfile.TemporaryDirectory(
            prefix="llm-simulator-shape-calibration-") as temp_dir:
        candidate_file = Path(temp_dir) / "candidate.json"

        def evaluate(parameters: dict) -> float:
            key = tuple(round(parameters[name], 8) for name in PARAMETER_RANGES)
            if key in cache:
                return cache[key]["objective"]
            candidate_file.write_text(
                json.dumps(candidate_payload(args, parameters)), encoding="utf-8")
            scenario_results = []
            for scenario in scenarios:
                rows, _ = run_scenario(args, scenario, candidate_file)
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
            macro_tpot = statistics.fmean(
                item["tpot_mape_pct"] for item in scenario_results)
            macro_e2e = statistics.fmean(
                item["e2e_mape_pct"] for item in scenario_results)
            # Equal scenario weighting prevents the shorter isolated trace from
            # being hidden by the two 20-request traces. E2E is a secondary term.
            objective = macro_tpot + 0.20 * macro_e2e
            cache[key] = {
                "objective": objective,
                "macro_tpot_mape_pct": macro_tpot,
                "macro_e2e_mape_pct": macro_e2e,
                "scenarios": scenario_results,
                "parameters": dict(parameters),
            }
            print(
                f"eval={len(cache):03d} objective={objective:.3f} "
                f"base={parameters['base_factor']:.4f} "
                f"batch_exp={parameters['batch_exponent']:.4f} "
                f"kv_exp={parameters['kv_exponent']:.4f} "
                f"overhead={parameters['iteration_overhead_ms']:.3f}")
            return objective

        starts = [
            {"base_factor": 0.60, "batch_exponent": 0.0,
             "kv_exponent": 0.0, "iteration_overhead_ms": 0.0},
            {"base_factor": 0.75, "batch_exponent": 0.0,
             "kv_exponent": 0.0, "iteration_overhead_ms": 5.0},
            {"base_factor": 0.55, "batch_exponent": 0.15,
             "kv_exponent": 0.0, "iteration_overhead_ms": 10.0},
            {"base_factor": 0.70, "batch_exponent": -0.15,
             "kv_exponent": 0.10, "iteration_overhead_ms": 3.0},
        ]
        best, objective, history = coordinate_search(
            evaluate, starts, PARAMETER_RANGES,
            points=args.points, rounds=args.rounds)

    best_key = tuple(round(best[name], 8) for name in PARAMETER_RANGES)
    selected = cache[best_key]
    output_config = candidate_payload(args, best)
    output_config["metadata"] = {
        "calibration_role": "online_continuous_batch_colocated",
        "method": "multi-start coordinate search",
        "objective": "macro TPOT MAPE + 0.20 * macro E2E MAPE",
        "fit_dataset": "independent calibration traces only",
        "validation_policy": "no validation or holdout trace used",
    }
    write_json(output_config, args.output_config)
    report = {
        "schema_version": 2,
        "configuration": {
            "tp_size": args.tp_size,
            "num_gpu_blocks": args.num_gpu_blocks,
            "block_size_tokens": args.block_size_tokens,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "chunked_prefill_enabled": args.enable_chunked_prefill,
            "parameter_ranges": PARAMETER_RANGES,
            "reference_batch_size": args.reference_batch_size,
            "reference_kv_len": args.reference_kv_len,
        },
        "selected": selected,
        "evaluation_count": len(cache),
        "search_history": history,
        "sources": [{
            "name": scenario["name"],
            "trace": str(scenario["trace"]),
            "runs": [str(path) for path in scenario["runs"]],
        } for scenario in scenarios],
        "holdout_policy": (
            "Neither validation_trace.csv nor holdout_mixed_v2.csv is used in "
            "fitting. Freeze the generated config before running the new holdout."),
    }
    write_json(report, args.output_report)
    print(f"selected parameters: {best}")
    print(f"objective: {objective:.3f}")
    print(f"calibration config: {args.output_config}")
    print(f"fit report: {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
