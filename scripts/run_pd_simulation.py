#!/usr/bin/env python3
"""Run one PD-disaggregated scenario and export inspectable event logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pd_simulator import PDConfig, PDSimulator
from core.qwen3_cost_model import LLMCostModel
from core.simulator import RequestGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--operator-profile-dir", type=Path)
    parser.add_argument("--collective-profile-dir", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "pd_simulation")
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--enable-calibration", action="store_true")
    parser.add_argument("--prefill-gpus", type=int, default=4)
    parser.add_argument("--prefill-tp", type=int, default=4)
    parser.add_argument("--prefill-num-gpu-blocks", type=int, default=22136)
    parser.add_argument("--prefill-max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--prefill-max-num-seqs", type=int, default=16)
    parser.add_argument("--decode-gpus", type=int, default=4)
    parser.add_argument("--decode-tp", type=int, default=4)
    parser.add_argument("--decode-num-gpu-blocks", type=int, default=22136)
    parser.add_argument("--decode-max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--decode-max-num-seqs", type=int, default=32)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--kv-transfer-mode", default="pcie")
    parser.add_argument("--kv-transfer-bw-gb-s", type=float, default=12.5)
    parser.add_argument("--kv-transfer-latency-ms", type=float, default=0.1)
    parser.add_argument("--kv-transfer-concurrency", type=int, default=1)
    parser.add_argument(
        "--decode-kv-load-policy",
        choices=["post_transfer", "blocking_receive"],
        default="blocking_receive",
        help=("Decode-side KV admission semantics. Use blocking_receive for "
              "the vLLM 0.19 P2pNcclConnector deployment (default)."))
    parser.add_argument("--mode", choices=["fixed", "poisson", "trace"],
                        default="poisson")
    parser.add_argument("--arrival-rate-rps", type=float, default=1.0)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--simulation-duration-ms", type=float, default=1_000_000)
    parser.add_argument("--ttft-sla-ms", type=float, default=500)
    parser.add_argument("--tpot-sla-ms", type=float, default=50)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.enable_calibration and not args.calibration_file:
        raise SystemExit("--enable-calibration requires --calibration-file")
    if args.calibration_file and not args.calibration_file.exists():
        raise SystemExit(f"calibration file not found: {args.calibration_file}")
    if args.mode == "trace" and not args.trace:
        raise SystemExit("--mode trace requires --trace")
    if args.arrival_rate_rps <= 0:
        raise SystemExit("--arrival-rate-rps must be positive")

    model = LLMCostModel(
        str(args.data_dir),
        calibration_file=str(args.calibration_file) if args.calibration_file else None,
        enable_e2e_compensation=args.enable_calibration,
        operator_profile_dir=(str(args.operator_profile_dir)
                              if args.operator_profile_dir else None),
        collective_profile_dir=(str(args.collective_profile_dir)
                                if args.collective_profile_dir else None))
    config = PDConfig(
        num_prefill_gpus=args.prefill_gpus,
        prefill_tp=args.prefill_tp,
        prefill_num_gpu_blocks=args.prefill_num_gpu_blocks,
        prefill_max_num_batched_tokens=args.prefill_max_num_batched_tokens,
        prefill_max_num_seqs=args.prefill_max_num_seqs,
        num_decode_gpus=args.decode_gpus,
        decode_tp=args.decode_tp,
        decode_num_gpu_blocks=args.decode_num_gpu_blocks,
        decode_max_num_batched_tokens=args.decode_max_num_batched_tokens,
        decode_max_num_seqs=args.decode_max_num_seqs,
        block_size_tokens=args.block_size_tokens,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        kv_transfer_mode=args.kv_transfer_mode,
        kv_transfer_bw_gb_s=args.kv_transfer_bw_gb_s,
        kv_transfer_latency_ms=args.kv_transfer_latency_ms,
        kv_transfer_concurrency=args.kv_transfer_concurrency,
        decode_kv_load_policy=args.decode_kv_load_policy)
    generator = RequestGenerator(
        mode=args.mode,
        arrival_interval_ms=1000.0 / args.arrival_rate_rps,
        prompt_len=args.prompt_len,
        output_len=args.output_len,
        trace_file=str(args.trace) if args.trace else None,
        seed=args.seed)
    simulator = PDSimulator(config, model, generator, verbose=args.verbose)
    metrics = simulator.run(
        simulation_duration_ms=args.simulation_duration_ms,
        ttft_sla_ms=args.ttft_sla_ms,
        tpot_sla_ms=args.tpot_sla_ms,
        num_requests=args.num_requests)

    request_rows = [{
        "request_id": request.request_id,
        "arrival_time_ms": request.arrival_time,
        "prompt_len": request.prompt_len,
        "output_len": request.output_len,
        "generated_tokens": request.generated_tokens,
        "prefill_start_time_ms": request.prefill_start_time,
        "prefill_end_time_ms": request.prefill_end_time,
        "kv_transfer_start_time_ms": request.kv_transfer_start_time,
        "kv_transfer_end_time_ms": request.kv_transfer_end_time,
        "first_token_time_ms": request.first_token_time,
        "completion_time_ms": request.completion_time,
        "ttft_ms": request.ttft,
        "tpot_ms": request.tpot,
        "e2e_ms": request.e2e_latency,
        "status": request.pd_status.value,
        "prefill_replica": request.assigned_prefill_replica,
        "decode_replica": request.assigned_decode_replica,
    } for request in simulator.requests]
    report = {
        "schema_version": 1,
        "experiment_type": "pd_disaggregated_simulation",
        "configuration": asdict(config),
        "workload": {
            "mode": args.mode,
            "arrival_rate_rps": args.arrival_rate_rps,
            "prompt_len": args.prompt_len,
            "output_len": args.output_len,
            "trace": str(args.trace.resolve()) if args.trace else None,
            "num_requests": len(simulator.requests),
            "seed": args.seed,
        },
        "calibration": {
            "enabled": args.enable_calibration,
            "file": (str(args.calibration_file.resolve())
                     if args.calibration_file else None),
        },
        "profiling": {
            "operator_profile_dir": (
                str(args.operator_profile_dir.resolve())
                if args.operator_profile_dir else None),
            "collective_profile_dir": (
                str(args.collective_profile_dir.resolve())
                if args.collective_profile_dir else None),
        },
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "pd_report.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    write_csv(args.output_dir / "pd_requests.csv", request_rows)
    write_csv(args.output_dir / "pd_batches.csv", simulator.batch_history)
    write_csv(args.output_dir / "pd_transfers.csv", simulator.transfer_history)
    write_csv(args.output_dir / "pd_blocking_receives.csv",
              simulator.decode_receive_history)
    print(f"completed requests: {metrics['completed_requests']}/{metrics['total_requests']}")
    print(f"TTFT p99: {metrics['ttft_ms']['p99']:.2f} ms")
    print(f"TPOT p99: {metrics['tpot_ms']['p99']:.2f} ms")
    print(f"SLA success: {metrics['goodput']['both_ratio']:.2%}")
    print(f"report: {args.output_dir / 'pd_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
