#!/usr/bin/env python3
"""Generate a paired iso-workload relative hardware comparison."""

import argparse
import csv
import json
from pathlib import Path

from core.benchmark_protocol import compare_reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--ground-truth-ratios", help="optional JSON mapping case name to real speedup")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    truth = None
    if args.ground_truth_ratios:
        truth = json.loads(Path(args.ground_truth_ratios).read_text(encoding="utf-8"))
    result = compare_reports(candidate, reference, truth)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "relative_benchmark.json"
    csv_path = output_dir / "relative_benchmark_cases.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = [
        "stage", "case", "batch_size", "representative_length",
        "candidate_throughput_tok_s", "reference_throughput_tok_s", "speedup",
        "ground_truth_speedup", "relative_error_pct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["points"])
    print(f"status={result['status']}")
    print(f"prefill speedup={result['scores']['prefill_speedup']}")
    print(f"decode speedup={result['scores']['decode_speedup']}")
    print(f"report: {json_path}")
    return 0 if result["status"] != "invalid" else 2


if __name__ == "__main__":
    raise SystemExit(main())

