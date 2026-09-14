#!/usr/bin/env python3
"""Fit a TP-specific decode factor using calibration traces only.

The validation trace is deliberately rejected so it remains an independent
holdout. Prefill is left at 1.0 because request-level TTFT errors are dominated
by scheduling and batch-boundary differences rather than a stable scale error.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.communication_model import CommunicationModel
from core.kv_block_pool import KVBlockPool
from core.qwen3_cost_model import LLMCostModel
from core.resource_checker import ResourceChecker
from core.simulator import RequestGenerator, Scheduler, Simulator
from scripts.run_validation import simulation_rows
from scripts.validation_utils import aggregate_vllm_runs, compare_requests, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-root", type=Path, default=(
        PROJECT_ROOT / "data" / "ground_truth_co_located" / "calibration_tp4"))
    parser.add_argument("--trace-dir", type=Path, default=(
        PROJECT_ROOT / "data" / "trace" / "calibration"))
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-config", type=Path,
                        default=PROJECT_ROOT / "configs" / "calibration_tp4.json")
    parser.add_argument("--output-report", type=Path, default=(
        PROJECT_ROOT / "data" / "calibration" / "tp4" / "fit_report.json"))
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--num-gpu-blocks", type=int, default=22136)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--simulation-duration-ms", type=float,
                        default=1_000_000.0)
    parser.add_argument("--factor-min", type=float, default=0.40)
    parser.add_argument("--factor-max", type=float, default=1.20)
    parser.add_argument("--grid-points", type=int, default=17)
    parser.add_argument("--refine-rounds", type=int, default=3)
    return parser.parse_args()


def grid_search(objective: Callable[[float], float], low: float, high: float,
                points: int = 17,
                rounds: int = 3) -> tuple[float, float, list[dict]]:
    """Deterministic coarse-to-fine search, robust to scheduling discontinuities."""
    if not 0 < low < high or points < 3 or rounds < 1:
        raise ValueError("invalid grid-search parameters")
    original_low, original_high = low, high
    evaluated: dict[float, float] = {}
    for _ in range(rounds):
        step = (high - low) / (points - 1)
        candidates = [low + index * step for index in range(points)]
        for factor in candidates:
            key = round(factor, 10)
            if key not in evaluated:
                evaluated[key] = float(objective(factor))
        best = min(candidates, key=lambda value: evaluated[round(value, 10)])
        low = max(original_low, best - step)
        high = min(original_high, best + step)
    best_key = min(evaluated, key=evaluated.get)
    curve = [{"decode_factor": factor, "objective_tpot_mape_pct": value}
             for factor, value in sorted(evaluated.items())]
    return best_key, evaluated[best_key], curve


def discover_scenarios(root: Path, trace_dir: Path) -> list[dict]:
    if not root.is_dir():
        raise FileNotFoundError(f"ground-truth root does not exist: {root}")
    scenarios = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        runs = sorted(directory.glob("repeat_*.json"))
        if len(runs) < 2:
            continue
        aggregate = aggregate_vllm_runs(runs)
        trace_name = Path(str(aggregate.get("trace", ""))).name
        if trace_name == "validation_trace.csv" or "validation" in directory.name.lower():
            raise ValueError("validation data must not be used for calibration")
        trace = trace_dir / trace_name
        if not trace.is_file():
            raise FileNotFoundError(
                f"cannot match {directory} to calibration trace {trace}")
        scenarios.append({
            "name": directory.name,
            "trace": trace.resolve(),
            "runs": [path.resolve() for path in runs],
            "aggregate": aggregate,
        })
    if not scenarios:
        raise ValueError(f"no repeated calibration scenarios found under {root}")
    return scenarios


def run_scenario(args: argparse.Namespace, scenario: dict,
                 calibration_file: Path) -> tuple[list[dict], dict]:
    model = LLMCostModel(
        str(args.data_dir), calibration_file=str(calibration_file),
        enable_e2e_compensation=True)
    kv_bytes = model.memory_estimator.kv_bytes_per_token_per_rank(args.tp_size)
    pool = KVBlockPool(
        args.num_gpu_blocks, args.block_size_tokens,
        kv_bytes_per_token_per_rank=kv_bytes)
    checker = ResourceChecker(pool, CommunicationModel())
    scheduler = Scheduler(
        args.max_num_batched_tokens, args.max_num_seqs,
        pool, model, checker,
        enable_chunked_prefill=getattr(
            args, "enable_chunked_prefill", False))
    generator = RequestGenerator(
        mode="trace", trace_file=str(scenario["trace"]))
    simulator = Simulator(
        model, pool, checker, scheduler, generator, tp_size=args.tp_size)
    summary = simulator.run(
        simulation_duration_ms=args.simulation_duration_ms,
        num_requests=scenario["aggregate"]["request_count"])
    rows = simulation_rows(simulator)
    if len(rows) != scenario["aggregate"]["request_count"]:
        raise RuntimeError(
            f"{scenario['name']}: simulation completed {len(rows)}/"
            f"{scenario['aggregate']['request_count']} requests")
    return rows, summary


def main() -> int:
    args = parse_args()
    scenarios = discover_scenarios(args.ground_truth_root, args.trace_dir)
    cache: dict[float, dict] = {}
    with tempfile.TemporaryDirectory(
            prefix="llm-simulator-calibration-") as temp_dir:
        candidate_file = Path(temp_dir) / "candidate.json"

        def evaluate(factor: float) -> float:
            key = round(factor, 10)
            if key in cache:
                return cache[key]["objective"]
            candidate_file.write_text(json.dumps({
                "prefill": {str(args.tp_size): 1.0},
                "decode": {str(args.tp_size): factor},
            }), encoding="utf-8")
            errors = []
            scenario_results = []
            for scenario in scenarios:
                rows, _ = run_scenario(args, scenario, candidate_file)
                _, summary = compare_requests(scenario["aggregate"], rows)
                request_count = scenario["aggregate"]["request_count"]
                errors.append((summary["tpot_ms"]["mape_pct"], request_count))
                scenario_results.append({
                    "name": scenario["name"],
                    "request_count": request_count,
                    "tpot_mape_pct": summary["tpot_ms"]["mape_pct"],
                    "e2e_mape_pct": summary["e2e_ms"]["mape_pct"],
                    "ttft_mape_pct": summary["ttft_ms"]["mape_pct"],
                })
            objective_value = sum(value * count for value, count in errors) / sum(
                count for _, count in errors)
            cache[key] = {
                "objective": objective_value,
                "scenarios": scenario_results,
            }
            print(f"decode_factor={factor:.6f}, "
                  f"TPOT MAPE={objective_value:.3f}%")
            return objective_value

        best_factor, best_objective, curve = grid_search(
            evaluate, args.factor_min, args.factor_max,
            args.grid_points, args.refine_rounds)

    config = {
        "schema_version": 1,
        "prefill": {str(args.tp_size): 1.0},
        "decode": {str(args.tp_size): round(best_factor, 6)},
        "metadata": {
            "method": "weighted request-level TPOT MAPE grid search",
            "fit_dataset": "independent calibration traces only",
            "prefill_policy": "kept at 1.0; not fitted on scheduler-sensitive TTFT",
        },
    }
    write_json(config, args.output_config)
    report = {
        "schema_version": 1,
        "configuration": {
            "tp_size": args.tp_size,
            "num_gpu_blocks": args.num_gpu_blocks,
            "block_size_tokens": args.block_size_tokens,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "factor_range": [args.factor_min, args.factor_max],
        },
        "selected": {
            "decode_factor": round(best_factor, 6),
            "objective_tpot_mape_pct": best_objective,
            "scenarios": cache[round(best_factor, 10)]["scenarios"],
        },
        "sources": [{
            "name": scenario["name"],
            "trace": str(scenario["trace"]),
            "runs": [str(path) for path in scenario["runs"]],
        } for scenario in scenarios],
        "search_curve": curve,
        "holdout_policy": (
            "validation_trace.csv is excluded from fitting and must be used only "
            "after this configuration is generated"),
    }
    write_json(report, args.output_report)
    print(f"selected decode factor: {best_factor:.6f}")
    print(f"calibration config: {args.output_config}")
    print(f"fit report: {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
