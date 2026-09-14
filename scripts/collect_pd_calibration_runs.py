#!/usr/bin/env python3
"""Collect repeated real PD runs for independent calibration traces."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = PROJECT_ROOT / "data" / "trace" / "calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--deployment-config", type=Path, required=True)
    parser.add_argument("--component-metrics-url", action="append", default=[])
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--traces", nargs="*", type=Path)
    parser.add_argument("--output-root", type=Path, default=(
        PROJECT_ROOT / "data" / "ground_truth_pd" / "calibration_tp4_1p1d"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--warmup-prompt-len", type=int, default=128)
    parser.add_argument("--warmup-output-len", type=int, default=8)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_traces(trace_dir: Path,
                    explicit: Sequence[Path] | None) -> list[Path]:
    traces = list(explicit or sorted(trace_dir.glob("calibration_*.csv")))
    if not traces:
        raise ValueError(f"no PD calibration traces found in {trace_dir}")
    missing = [str(path) for path in traces if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing trace files: {missing}")
    forbidden = [path.name for path in traces
                 if "validation" in path.name.lower() or "holdout" in path.name.lower()]
    if forbidden:
        raise ValueError(f"validation/holdout traces cannot be calibrated: {forbidden}")
    return [path.resolve() for path in traces]


def trace_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def build_command(args: argparse.Namespace, trace: Path, output: Path,
                  repeat_id: str) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "collect_pd_ground_truth.py"),
        "--base-url", args.base_url,
        "--model", args.model,
        "--tokenizer", args.tokenizer or args.model,
        "--trace", str(trace),
        "--deployment-config", str(args.deployment_config),
        "--output", str(output),
        "--repeat-id", repeat_id,
        "--api-key", args.api_key,
        "--max-concurrency", str(args.max_concurrency),
        "--timeout", str(args.timeout),
        "--warmup-requests", str(args.warmup_requests),
        "--warmup-prompt-len", str(args.warmup_prompt_len),
        "--warmup-output-len", str(args.warmup_output_len),
    ]
    for endpoint in args.component_metrics_url:
        command.extend(("--component-metrics-url", endpoint))
    return command


def validate_output(path: Path, expected_requests: int,
                    expected_hash: str | None = None) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("collector") != "vllm_pd_proxy_streaming":
        raise RuntimeError(f"not a PD ground-truth run: {path}")
    requests = payload.get("requests", [])
    failed = [item.get("request_id") for item in requests
              if item.get("status") != "success"]
    if len(requests) != expected_requests or failed:
        raise RuntimeError(
            f"invalid collection {path}: requests={len(requests)}/"
            f"{expected_requests}, failed={failed}")
    digest = payload.get("deployment_config_sha256")
    if not digest:
        raise RuntimeError(f"collection has no deployment hash: {path}")
    if expected_hash is not None and digest != expected_hash:
        raise RuntimeError(
            f"deployment changed during calibration: {digest} != {expected_hash}")
    return digest


def main() -> int:
    args = parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2 (3 is recommended)")
    traces = discover_traces(args.trace_dir, args.traces)
    total = len(traces) * args.repeats
    completed = 0
    deployment_hash = None
    for trace in traces:
        scenario_dir = args.output_root / trace.stem
        expected = trace_row_count(trace)
        for repeat in range(1, args.repeats + 1):
            repeat_id = f"{trace.stem}_repeat_{repeat}"
            output = scenario_dir / f"repeat_{repeat}.json"
            command = build_command(args, trace, output, repeat_id)
            if output.exists() and not args.overwrite:
                deployment_hash = validate_output(
                    output, expected, deployment_hash)
                print(f"[{completed + 1}/{total}] reuse {output}")
            elif args.dry_run:
                print(subprocess.list2cmdline(command))
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                print(f"[{completed + 1}/{total}] collect {trace.stem} repeat {repeat}")
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                deployment_hash = validate_output(
                    output, expected, deployment_hash)
            completed += 1
            if (not args.dry_run and completed < total and
                    args.cooldown_seconds > 0):
                time.sleep(args.cooldown_seconds)
    print(f"PD calibration collection completed: {completed} run(s), "
          f"root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
