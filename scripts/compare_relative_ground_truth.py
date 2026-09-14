#!/usr/bin/env python3
"""Build real A/B speedups and validate predicted relative hardware scores."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.benchmark_protocol import (  # noqa: E402
    adapt_legacy_report,
    compare_reports,
    geometric_mean,
    load_benchmark_spec,
)


RUNTIME_KEYS = (
    "tensor_parallel_size", "tp_size", "dtype", "max_model_len",
    "max_num_batched_tokens", "max_num_seqs", "enforce_eager",
    "enable_prefix_caching", "block_size_tokens",
)


def artifact_path(manifest_path: Path, stored: str) -> Path:
    path = Path(stored)
    if path.exists():
        return path
    local = manifest_path.parent / path.name
    if local.exists():
        return local
    raise FileNotFoundError(f"artifact not found: {stored} or {local}")


def load_manifest(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_type") != "relative_ground_truth_run":
        raise ValueError(f"not a relative ground-truth manifest: {path}")
    return payload


def load_entries(manifest_path: Path, manifest: Dict) -> Dict[Tuple[str, str], Dict]:
    result = {}
    for entry in manifest.get("entries", []):
        key = (entry["stage"], entry["case"])
        if key in result:
            raise ValueError(f"duplicate case in {manifest_path}: {key}")
        payload = json.loads(artifact_path(manifest_path, entry["file"]).read_text(
            encoding="utf-8"))
        if payload.get("valid") is not True:
            raise ValueError(f"invalid fixed-batch artifact: {entry['file']}")
        result[key] = {"entry": entry, "payload": payload}
    return result


def throughput(stage: str, entry: Dict, payload: Dict) -> Tuple[float, float]:
    summary = payload.get("summary") or {}
    mean_ms = float(summary.get("mean_ms", 0))
    std_ms = float(summary.get("std_ms", 0))
    if mean_ms <= 0:
        raise ValueError(f"case {entry['case']} has no positive mean_ms")
    batch = int(entry["batch_size"])
    tokens = batch * int(entry["prompt_length"]) if stage == "prefill" else batch
    return tokens * 1000.0 / mean_ms, std_ms / mean_ms


def runtime_identity(entries: Dict) -> Dict:
    if not entries:
        return {}
    payload = next(iter(entries.values()))["payload"]
    deployment = payload.get("deployment", {}).get("config", {})
    return {key: deployment.get(key) for key in RUNTIME_KEYS
            if deployment.get(key) is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-spec", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--relative-mape-threshold-pct", type=float, default=15.0)
    parser.add_argument("--max-cv-pct", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = load_benchmark_spec(args.benchmark_spec)
    reference_manifest = load_manifest(args.reference_manifest)
    candidate_manifest = load_manifest(args.candidate_manifest)
    identity_keys = ("protocol_version", "model_id", "tp_size", "dtype",
                     "benchmark_spec_sha256")
    identity = {key: reference_manifest.get(key) == candidate_manifest.get(key)
                for key in identity_keys}
    if not all(identity.values()):
        raise SystemExit(f"A/B manifest identity mismatch: {identity}")
    reference_entries = load_entries(args.reference_manifest, reference_manifest)
    candidate_entries = load_entries(args.candidate_manifest, candidate_manifest)
    if set(reference_entries) != set(candidate_entries):
        raise SystemExit("A/B manifests do not contain exactly the same cases")
    expected = {(stage, case["case"])
                for stage in ("prefill", "decode")
                for case in spec["workloads"][stage]}
    if set(reference_entries) != expected:
        raise SystemExit("manifests do not exactly cover the BenchmarkSpec cases")

    reference_runtime = runtime_identity(reference_entries)
    candidate_runtime = runtime_identity(candidate_entries)
    runtime_compatible = reference_runtime == candidate_runtime
    points = []
    ratios = {}
    repeatability_ok = True
    for stage, case in sorted(expected):
        ref = reference_entries[(stage, case)]
        cand = candidate_entries[(stage, case)]
        # Compare workload fields while allowing different file paths.
        ref_shape = {key: value for key, value in ref["entry"].items() if key != "file"}
        cand_shape = {key: value for key, value in cand["entry"].items() if key != "file"}
        if ref_shape != cand_shape:
            raise SystemExit(f"shape mismatch for {stage}/{case}")
        ref_rate, ref_cv = throughput(stage, ref["entry"], ref["payload"])
        cand_rate, cand_cv = throughput(stage, cand["entry"], cand["payload"])
        ratio = cand_rate / ref_rate
        ratios[case] = ratio
        cv_ok = max(ref_cv, cand_cv) * 100.0 <= args.max_cv_pct
        repeatability_ok = repeatability_ok and cv_ok
        points.append({
            "stage": stage,
            "case": case,
            "batch_size": ref["entry"]["batch_size"],
            "representative_length": ref["entry"].get(
                "prompt_length", ref["entry"].get("kv_length")),
            "reference_throughput_tok_s": ref_rate,
            "candidate_throughput_tok_s": cand_rate,
            "ground_truth_speedup": ratio,
            "reference_cv_pct": ref_cv * 100.0,
            "candidate_cv_pct": cand_cv * 100.0,
            "repeatability_ok": cv_ok,
        })

    verification = None
    acceptance = {
        "ground_truth_complete": True,
        "runtime_compatible": runtime_compatible,
        "repeatability_ok": repeatability_ok,
        "relative_mape_threshold_pct": args.relative_mape_threshold_pct,
        "rank_agreement_required": 1.0,
        "passed": None,
    }
    if bool(args.reference_report) != bool(args.candidate_report):
        raise SystemExit("provide both predicted reports or neither")
    if args.reference_report:
        reference_report = adapt_legacy_report(json.loads(
            args.reference_report.read_text(encoding="utf-8")), spec)
        candidate_report = adapt_legacy_report(json.loads(
            args.candidate_report.read_text(encoding="utf-8")), spec)
        verification = compare_reports(candidate_report, reference_report, ratios)
        validation = verification["validation"]
        acceptance["passed"] = bool(
            runtime_compatible and repeatability_ok
            and validation["rank_agreement_ratio"] == 1.0
            and validation["relative_mape_pct"] <= args.relative_mape_threshold_pct)

    output = {
        "schema_version": 1,
        "experiment_type": "relative_ground_truth_comparison",
        "protocol_version": spec["protocol_version"],
        "reference": {
            "hardware_id": reference_manifest["hardware_id"],
            "tp_size": reference_manifest["tp_size"],
            "manifest": str(args.reference_manifest.resolve()),
        },
        "candidate": {
            "hardware_id": candidate_manifest["hardware_id"],
            "tp_size": candidate_manifest["tp_size"],
            "manifest": str(args.candidate_manifest.resolve()),
        },
        "identity": identity,
        "runtime": {
            "compatible": runtime_compatible,
            "reference": reference_runtime,
            "candidate": candidate_runtime,
        },
        "ground_truth_ratios": ratios,
        "ground_truth_scores": {
            "prefill_speedup": geometric_mean(
                point["ground_truth_speedup"] for point in points
                if point["stage"] == "prefill"),
            "decode_speedup": geometric_mean(
                point["ground_truth_speedup"] for point in points
                if point["stage"] == "decode"),
        },
        "points": points,
        "verification": verification,
        "acceptance": acceptance,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "relative_ground_truth_report.json"
    ratios_path = args.output_dir / "ground_truth_ratios.json"
    csv_path = args.output_dir / "relative_ground_truth_cases.csv"
    report_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    ratios_path.write_text(json.dumps(ratios, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    print(f"real P-Score={output['ground_truth_scores']['prefill_speedup']:.4f}x")
    print(f"real D-Score={output['ground_truth_scores']['decode_speedup']:.4f}x")
    if verification:
        print(f"relative MAPE={verification['validation']['relative_mape_pct']:.2f}%")
        print(f"rank agreement={verification['validation']['rank_agreement_ratio']:.2%}")
        print(f"accepted={acceptance['passed']}")
    print(f"report: {report_path}")
    return 0 if acceptance["passed"] is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
