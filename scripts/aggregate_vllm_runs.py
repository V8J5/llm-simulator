#!/usr/bin/env python3
"""Aggregate repeated vLLM streaming runs into request-level ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validation_utils import aggregate_vllm_runs, write_json, write_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="two or more repeated vLLM JSON files")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aggregate = aggregate_vllm_runs(args.inputs)
    write_json(aggregate, args.output_json)
    write_rows(aggregate["requests"], args.output_csv)
    print(f"aggregated {aggregate['repeat_count']} runs and "
          f"{aggregate['request_count']} requests")
    for metric, stats in aggregate["summary"]["repeatability"].items():
        print(f"{metric}: mean={stats['mean']:.3f} ms, cv={stats['cv_pct']:.3f}%")
    print(f"JSON: {args.output_json}")
    print(f"CSV:  {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
