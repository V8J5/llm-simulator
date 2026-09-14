#!/usr/bin/env python3
"""Run the simulator on the vLLM trace and produce an error report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.communication_model import CommunicationModel
from core.kv_block_pool import KVBlockPool
from core.qwen3_cost_model import LLMCostModel
from core.resource_checker import ResourceChecker
from core.simulator import RequestGenerator, Scheduler, Simulator
from scripts.validation_utils import (
    aggregate_vllm_runs,
    compare_requests,
    write_json,
    write_rows,
)
from scripts.calibration_roles import (
    ONLINE_COLOCATED,
    require_calibration_role,
)


def parse_args() -> argparse.Namespace:
    default_run_dir = (PROJECT_ROOT / "data" / "ground_truth_co_located" /
                       "L20_Qwen3-32B_TP4")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path,
                        default=sorted(default_run_dir.glob("vllm_tp4_repeat*.json")))
    parser.add_argument("--trace", type=Path,
                        default=PROJECT_ROOT / "data" / "trace" / "validation_trace.csv")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--operator-profile-dir", type=Path)
    parser.add_argument("--collective-profile-dir", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "validation" / "tp4_baseline")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--num-gpu-blocks", type=int, default=22136)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--simulation-duration-ms", type=float, default=1_000_000.0)
    parser.add_argument("--ttft-sla-ms", type=float, default=500.0)
    parser.add_argument("--tpot-sla-ms", type=float, default=50.0)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--enable-calibration", action="store_true")
    return parser.parse_args()


def simulation_rows(simulator: Simulator) -> list[dict]:
    rows = []
    for request in sorted(simulator.completed_requests,
                          key=lambda item: item.arrival_time):
        rows.append({
            "request_id": request.request_id,
            "arrival_time_ms": request.arrival_time,
            "prompt_len": request.prompt_len,
            "output_len": request.output_len,
            "generated_tokens": request.generated_tokens,
            "prefill_start_time_ms": request.prefill_start_time,
            "first_token_time_ms": request.first_token_time,
            "completion_time_ms": request.completion_time,
            "ttft_ms": request.ttft,
            "tpot_ms": request.tpot,
            "e2e_ms": request.e2e_latency,
        })
    return rows


def main() -> int:
    args = parse_args()
    if len(args.runs) < 2:
        raise SystemExit("at least two --runs files are required")
    calibration_role = None
    if args.enable_calibration:
        if args.calibration_file is None:
            raise SystemExit(
                "--enable-calibration requires --calibration-file")
        calibration_role = require_calibration_role(
            args.calibration_file, ONLINE_COLOCATED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = aggregate_vllm_runs(args.runs)
    write_json(aggregate, args.output_dir / "ground_truth_aggregated.json")
    write_rows(aggregate["requests"],
               args.output_dir / "ground_truth_aggregated.csv")

    model = LLMCostModel(
        str(args.data_dir),
        calibration_file=str(args.calibration_file) if args.calibration_file else None,
        enable_e2e_compensation=args.enable_calibration,
        operator_profile_dir=(str(args.operator_profile_dir)
                              if args.operator_profile_dir else None),
        collective_profile_dir=(str(args.collective_profile_dir)
                                if args.collective_profile_dir else None),
    )
    kv_bytes = model.memory_estimator.kv_bytes_per_token_per_rank(args.tp_size)
    pool = KVBlockPool(
        total_blocks=args.num_gpu_blocks,
        block_size_tokens=args.block_size_tokens,
        kv_bytes_per_token_per_rank=kv_bytes,
    )
    checker = ResourceChecker(pool, CommunicationModel())
    scheduler = Scheduler(
        args.max_num_batched_tokens,
        args.max_num_seqs,
        pool,
        model,
        checker,
        enable_chunked_prefill=args.enable_chunked_prefill,
    )
    generator = RequestGenerator(mode="trace", trace_file=str(args.trace))
    simulator = Simulator(
        model,
        pool,
        checker,
        scheduler,
        generator,
        tp_size=args.tp_size,
    )
    simulation_summary = simulator.run(
        simulation_duration_ms=args.simulation_duration_ms,
        ttft_sla_ms=args.ttft_sla_ms,
        tpot_sla_ms=args.tpot_sla_ms,
        num_requests=aggregate["request_count"],
    )
    rows = simulation_rows(simulator)
    if len(rows) != aggregate["request_count"]:
        raise RuntimeError(
            f"simulation completed {len(rows)}/{aggregate['request_count']} requests")
    write_rows(rows, args.output_dir / "simulation_requests.csv")
    if simulator.batch_history:
        write_rows(simulator.batch_history,
                   args.output_dir / "simulation_batches.csv")

    detail, error_summary = compare_requests(aggregate, rows)
    write_rows(detail, args.output_dir / "request_errors.csv")
    gt_throughput = aggregate["summary"]["throughput"]
    throughput_comparison = {
        "input_tokens_per_s": {
            "ground_truth_mean": gt_throughput["input_tokens_per_s"]["mean"],
            "simulation": simulation_summary["input_throughput_tokens_per_s"],
        },
        "output_tokens_per_s": {
            "ground_truth_mean": gt_throughput["output_tokens_per_s"]["mean"],
            "simulation": simulation_summary["output_throughput_tokens_per_s"],
        },
    }
    for values in throughput_comparison.values():
        gt = values["ground_truth_mean"]
        values["error_pct"] = 100.0 * (values["simulation"] - gt) / gt if gt else 0.0

    report = {
        "schema_version": 1,
        "configuration": {
            "tp_size": args.tp_size,
            "num_gpu_blocks": args.num_gpu_blocks,
            "block_size_tokens": args.block_size_tokens,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "chunked_prefill_enabled": args.enable_chunked_prefill,
            "trace": str(args.trace.resolve()),
            "calibration_enabled": args.enable_calibration,
            "calibration_file": (str(args.calibration_file.resolve())
                                 if args.calibration_file else None),
            "calibration_role": calibration_role,
            "operator_profile_dir": (
                str(args.operator_profile_dir.resolve())
                if args.operator_profile_dir else None),
            "collective_profile_dir": (
                str(args.collective_profile_dir.resolve())
                if args.collective_profile_dir else None),
        },
        "ground_truth_summary": aggregate["summary"],
        "simulation_summary": simulation_summary,
        "error_summary": error_summary,
        "throughput_comparison": throughput_comparison,
        "limitations": [
            ("Chunked prefill is modeled with decode-first token budgeting."
             if args.enable_chunked_prefill else
             "Chunked prefill is disabled in this simulation."),
            ("End-to-end calibration is enabled."
             if args.enable_calibration else
             "This report is an uncalibrated baseline."),
        ],
    }
    write_json(report, args.output_dir / "validation_report.json")

    print(f"completed requests: {simulation_summary['completed_requests']}")
    for metric, stats in error_summary.items():
        print(f"{metric}: MAPE={stats['mape_pct']:.2f}%, "
              f"MAE={stats['mae_ms']:.2f} ms, "
              f"bias={stats['mean_signed_error_pct']:.2f}%")
    for metric, values in throughput_comparison.items():
        print(f"{metric}: error={values['error_pct']:.2f}%")
    print(f"report: {args.output_dir / 'validation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
