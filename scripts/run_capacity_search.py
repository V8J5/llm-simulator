#!/usr/bin/env python3
"""Scan co-located serving configurations and rank SLA capacity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_runner import (
    CapacityExperimentRunner,
    CapacityPolicy,
    WorkloadSpec,
    serving_grid,
)
from core.qwen3_cost_model import LLMCostModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--operator-profile-dir", type=Path)
    parser.add_argument("--collective-profile-dir", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "capacity_search")
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--enable-calibration", action="store_true")
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[4])
    parser.add_argument("--dp-sizes", nargs="+", type=int, default=[1],
                        help="number of independent inference replicas")
    parser.add_argument("--num-gpu-blocks", nargs="+", type=int, default=[22136])
    parser.add_argument("--block-size-tokens", nargs="+", type=int, default=[16])
    parser.add_argument("--max-num-batched-tokens", nargs="+", type=int,
                        default=[4096, 8192])
    parser.add_argument("--max-num-seqs", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--chunked-prefill", choices=["on", "off", "both"],
                        default="on")
    parser.add_argument("--arrival-rates-rps", nargs="+", type=float,
                        required=True)
    parser.add_argument("--refine-iterations", type=int, default=3,
                        help="binary refinements of the first SLA pass/fail interval")
    parser.add_argument("--mode", choices=["fixed", "poisson"], default="poisson")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--prompt-len-range", nargs=2, type=int,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--output-len-range", nargs=2, type=int,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--simulation-duration-ms", type=float, default=1_000_000.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--ttft-sla-ms", type=float, default=500.0)
    parser.add_argument("--tpot-sla-ms", type=float, default=50.0)
    parser.add_argument("--min-completion-ratio", type=float, default=1.0)
    parser.add_argument("--min-sla-success-ratio", type=float, default=0.9)
    parser.add_argument("--allow-unprofiled-tp", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [key for key in rows[0] if key != "runs"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> int:
    args = parse_args()
    if args.enable_calibration and not args.calibration_file:
        raise SystemExit("--enable-calibration requires --calibration-file")
    if args.calibration_file and not args.calibration_file.exists():
        raise SystemExit(f"calibration file not found: {args.calibration_file}")
    chunked = {"on": [True], "off": [False], "both": [False, True]}[
        args.chunked_prefill]
    model = LLMCostModel(
        str(args.data_dir),
        calibration_file=str(args.calibration_file) if args.calibration_file else None,
        enable_e2e_compensation=args.enable_calibration,
        operator_profile_dir=(str(args.operator_profile_dir)
                              if args.operator_profile_dir else None),
        collective_profile_dir=(str(args.collective_profile_dir)
                                if args.collective_profile_dir else None),
    )
    profiled_tps = {1, *model.comm_tables.keys()}
    missing_tps = sorted(set(args.tp_sizes) - profiled_tps)
    if missing_tps and not args.allow_unprofiled_tp:
        raise SystemExit(
            f"TP values lack communication profiles: {missing_tps}; "
            "collect profiles or pass --allow-unprofiled-tp explicitly")
    configs = serving_grid(
        args.tp_sizes, args.num_gpu_blocks, args.block_size_tokens,
        args.max_num_batched_tokens, args.max_num_seqs, chunked, args.dp_sizes)
    workload = WorkloadSpec(
        mode=args.mode, prompt_len=args.prompt_len, output_len=args.output_len,
        prompt_len_range=(tuple(args.prompt_len_range)
                          if args.prompt_len_range else None),
        output_len_range=(tuple(args.output_len_range)
                          if args.output_len_range else None),
        num_requests=args.num_requests,
        simulation_duration_ms=args.simulation_duration_ms, seed=args.seed)
    policy = CapacityPolicy(
        ttft_sla_ms=args.ttft_sla_ms, tpot_sla_ms=args.tpot_sla_ms,
        min_completion_ratio=args.min_completion_ratio,
        min_sla_success_ratio=args.min_sla_success_ratio)
    report = CapacityExperimentRunner(
        model, workload, policy, repeats=args.repeats).run(
            configs, args.arrival_rates_rps,
            refine_iterations=args.refine_iterations)
    report["provenance"] = {
        "data_dir": str(args.data_dir.resolve()),
        "calibration_enabled": args.enable_calibration,
        "calibration_file": (str(args.calibration_file.resolve())
                             if args.calibration_file else None),
        "operator_profile_dir": (
            str(args.operator_profile_dir.resolve())
            if args.operator_profile_dir else None),
        "collective_profile_dir": (
            str(args.collective_profile_dir.resolve())
            if args.collective_profile_dir else None),
        "profiled_tp_sizes": sorted(profiled_tps),
        "unprofiled_tp_sizes": missing_tps,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "capacity_report.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    write_csv(args.output_dir / "capacity_points.csv", report["points"])
    write_csv(args.output_dir / "capacity_ranking.csv", report["capacity_ranking"])
    write_csv(args.output_dir / "efficiency_ranking.csv", report["efficiency_ranking"])

    print(f"evaluated points: {len(report['points'])}")
    print("rank  lower_rps  upper_rps  config")
    for row in report["capacity_ranking"]:
        lower = row["capacity_lower_bound_rps"]
        upper = row["capacity_upper_bound_rps"]
        print(f"{row['rank']:>4}  {str(lower):>9}  {str(upper):>9}  "
              f"{row['config_id']}")
    print(f"report: {args.output_dir / 'capacity_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
