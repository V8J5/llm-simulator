#!/usr/bin/env python3
"""Collect repeated vLLM runs for every independent calibration trace."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--model", required=True, help="Model name exposed by vLLM")
    parser.add_argument("--tokenizer", help="HF tokenizer path; defaults to --model")
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--traces", nargs="*", type=Path,
                        help="Explicit trace files; otherwise use calibration_*.csv")
    parser.add_argument("--output-root", type=Path, default=(
        PROJECT_ROOT / "data" / "ground_truth_co_located" / "calibration_tp4"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_traces(trace_dir: Path,
                    explicit: Sequence[Path] | None) -> list[Path]:
    traces = list(explicit or sorted(trace_dir.glob("calibration_*.csv")))
    if not traces:
        raise ValueError(f"no calibration traces found in {trace_dir}")
    missing = [str(path) for path in traces if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing trace files: {missing}")
    return [path.resolve() for path in traces]


def build_command(args: argparse.Namespace, trace: Path,
                  output: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "collect_vllm_ground_truth.py"),
        "--base-url", args.base_url,
        "--model", args.model,
        "--tokenizer", args.tokenizer or args.model,
        "--trace", str(trace),
        "--output", str(output),
        "--api-key", args.api_key,
        "--max-concurrency", str(args.max_concurrency),
        "--timeout", str(args.timeout),
    ]


def validate_output(path: Path, expected_requests: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests", [])
    failed = [item.get("request_id") for item in requests
              if item.get("status") != "success"]
    if len(requests) != expected_requests or failed:
        raise RuntimeError(
            f"invalid collection {path}: requests={len(requests)}/"
            f"{expected_requests}, failed={failed}")


def trace_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def main() -> int:
    args = parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2 (3 is recommended)")
    traces = discover_traces(args.trace_dir, args.traces)
    total = len(traces) * args.repeats
    completed = 0
    for trace in traces:
        scenario_dir = args.output_root / trace.stem
        expected = trace_row_count(trace)
        for repeat in range(1, args.repeats + 1):
            output = scenario_dir / f"repeat_{repeat}.json"
            command = build_command(args, trace, output)
            if output.exists() and not args.overwrite:
                validate_output(output, expected)
                print(f"[{completed + 1}/{total}] reuse {output}")
            elif args.dry_run:
                print(subprocess.list2cmdline(command))
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                print(f"[{completed + 1}/{total}] collect {trace.stem} repeat {repeat}")
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                validate_output(output, expected)
            completed += 1
            if (not args.dry_run and completed < total and
                    args.cooldown_seconds > 0):
                time.sleep(args.cooldown_seconds)
    print(f"collection plan completed: {completed} run(s), root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
