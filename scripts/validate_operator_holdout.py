#!/usr/bin/env python3
"""Validate operator-profile interpolation on independently measured holdout shapes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profiling_common import (  # noqa: E402
    describe_ms,
    dominant_component,
    validate_profile_document,
    write_profile,
)
from core.operator_interpolation import (  # noqa: E402
    group_operator_times,
    interpolate_operator_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-profile", type=Path, required=True)
    parser.add_argument("--holdout-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mape-threshold-pct", type=float, default=20.0)
    return parser.parse_args()


def load(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_profile_document(payload, "operator")
    return payload


def _bounds(values: Sequence[float], query: float) -> Tuple[float, float]:
    values = sorted(set(values))
    if len(values) < 2:
        raise ValueError("interpolation requires at least two points on each axis")
    if query <= values[0]:
        return values[0], values[1]
    if query >= values[-1]:
        return values[-2], values[-1]
    index = bisect_right(values, query)
    return values[index - 1], values[index]


def _fraction(low: float, high: float, value: float) -> float:
    return ((math.log(max(value, 1e-9)) - math.log(max(low, 1e-9))) /
            (math.log(max(high, 1e-9)) - math.log(max(low, 1e-9))))


def interpolate(rows: Sequence[Dict], stage: str, batch: int, length: int,
                component: str) -> Tuple[float, bool]:
    length_key = "prompt_length" if stage == "prefill" else "kv_length"
    stage_rows = [row for row in rows if row.get("status") == "success"
                  and row.get("stage") == stage]
    if stage_rows and stage_rows[0].get("operators"):
        table = []
        for row in stage_rows:
            converted = {
                "batch_size": int(row["shape"]["batch_size"]),
                length_key: int(row["shape"][length_key]),
            }
            converted.update({
                f"operator::{name}": float(timing["mean_ms"])
                for name, timing in row["operators"].items()
            })
            table.append(converted)
        detail, outside = interpolate_operator_table(
            table, stage, batch, length)
        grouped = group_operator_times(detail)
        if component == "attention_ms":
            return grouped["attention_ms"], outside
        if component == "ffn_ms":
            return grouped["ffn_ms"], outside
        if component == "total_compute_ms":
            return sum(grouped.values()), outside
        raise ValueError(f"unknown component: {component}")
    xs = sorted({float(row["shape"]["batch_size"]) for row in stage_rows})
    ys = sorted({float(row["shape"][length_key]) for row in stage_rows})
    x0, x1 = _bounds(xs, batch)
    y0, y1 = _bounds(ys, length)
    grid = {
        (float(row["shape"]["batch_size"]), float(row["shape"][length_key])):
        float(row["single_layer"][component])
        for row in stage_rows
    }

    def point(x: float, y: float) -> float:
        if (x, y) not in grid:
            raise ValueError(f"training grid is incomplete at ({x:g}, {y:g})")
        return grid[(x, y)]

    fx, fy = _fraction(x0, x1, batch), _fraction(y0, y1, length)
    lower = point(x0, y0) + (point(x1, y0) - point(x0, y0)) * fx
    upper = point(x0, y1) + (point(x1, y1) - point(x0, y1)) * fx
    value = lower + (upper - lower) * fy
    outside = batch < xs[0] or batch > xs[-1] or length < ys[0] or length > ys[-1]
    return max(value, 0.0), outside


def _identity(metadata: Mapping) -> Dict:
    benchmark = metadata.get("benchmark", {})
    model = metadata.get("model", {})
    return {
        "tp_size": benchmark.get("tp_size"),
        "dtype": benchmark.get("dtype"),
        "model_config_sha256": model.get("config_sha256"),
        "gpu_inventory": metadata.get("cuda", {}).get("nvidia_smi_query"),
    }


def _summary(errors: Sequence[float]) -> Dict:
    absolute = [abs(value) for value in errors]
    return {
        "count": len(errors),
        "mape_pct": statistics.fmean(absolute) if absolute else 0.0,
        "bias_pct": statistics.fmean(errors) if errors else 0.0,
        "p90_absolute_error_pct": (
            describe_ms(absolute)["p90_ms"] if absolute else 0.0),
        "max_absolute_error_pct": max(absolute, default=0.0),
    }


def csv_fieldnames(rows: Sequence[Mapping]) -> List[str]:
    """Return stable union fields for mixed Prefill/Decode point rows."""
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def main() -> int:
    args = parse_args()
    training, holdout = load(args.training_profile), load(args.holdout_profile)
    training_identity = _identity(training["metadata"])
    holdout_identity = _identity(holdout["metadata"])
    if training_identity != holdout_identity:
        raise SystemExit(
            "training and holdout metadata differ:\n" +
            json.dumps({"training": training_identity, "holdout": holdout_identity},
                       indent=2))

    detail: List[Dict] = []
    components = ("attention_ms", "ffn_ms", "total_compute_ms")
    for row in holdout["measurements"]:
        if row.get("status") != "success":
            continue
        stage = row["stage"]
        length_key = "prompt_length" if stage == "prefill" else "kv_length"
        batch = int(row["shape"]["batch_size"])
        length = int(row["shape"][length_key])
        output = {
            "stage": stage, "tp_size": row["shape"]["tp_size"],
            "batch_size": batch, length_key: length,
        }
        predicted_components = {}
        observed_components = {}
        extrapolated = False
        for component in components:
            predicted, outside = interpolate(
                training["measurements"], stage, batch, length, component)
            observed = float(row["single_layer"][component])
            error_pct = 100.0 * (predicted - observed) / observed if observed else 0.0
            label = component.removesuffix("_ms")
            output[f"observed_{label}_ms"] = observed
            output[f"predicted_{label}_ms"] = predicted
            output[f"{label}_error_pct"] = error_pct
            predicted_components[label] = predicted
            observed_components[label] = observed
            extrapolated = extrapolated or outside
        output["extrapolated"] = extrapolated
        output["predicted_bottleneck"] = dominant_component({
            "attention": predicted_components["attention"],
            "ffn": predicted_components["ffn"],
        })
        output["observed_bottleneck"] = dominant_component({
            "attention": observed_components["attention"],
            "ffn": observed_components["ffn"],
        })
        output["bottleneck_match"] = (
            output["predicted_bottleneck"] == output["observed_bottleneck"])
        detail.append(output)
    if not detail:
        raise SystemExit("holdout profile contains no successful points")

    summaries = {}
    for stage in ("prefill", "decode"):
        stage_rows = [row for row in detail if row["stage"] == stage]
        if not stage_rows:
            continue
        summaries[stage] = {
            component.removesuffix("_ms"): _summary([
                row[f"{component.removesuffix('_ms')}_error_pct"] for row in stage_rows])
            for component in components
        }
        summaries[stage]["bottleneck_match_ratio"] = statistics.fmean(
            float(row["bottleneck_match"]) for row in stage_rows)

    passed = all(
        values["total_compute"]["mape_pct"] <= args.mape_threshold_pct
        for values in summaries.values())
    report = {
        "schema_version": 1,
        "experiment_type": "operator_interpolation_holdout_validation",
        "training_profile": str(args.training_profile.resolve()),
        "holdout_profile": str(args.holdout_profile.resolve()),
        "metadata_identity": training_identity,
        "acceptance": {
            "mape_threshold_pct": args.mape_threshold_pct,
            "passed": passed,
        },
        "summary": summaries,
        "points": detail,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_profile(report, args.output_dir / "operator_holdout_report.json")
    with (args.output_dir / "operator_holdout_points.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames(detail))
        writer.writeheader()
        writer.writerows(detail)
    for stage, values in summaries.items():
        print(f"{stage}: total compute MAPE="
              f"{values['total_compute']['mape_pct']:.2f}%, "
              f"bottleneck match={100 * values['bottleneck_match_ratio']:.1f}%")
    print(f"passed={passed}; report={args.output_dir / 'operator_holdout_report.json'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())