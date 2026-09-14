"""Shape-aware interpolation for schema-v2 operator profiles."""

from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from typing import Dict, List, Mapping, Sequence, Tuple


def operator_work(stage: str, operator: str, batch_size: int, length: int) -> float:
    """Return the dominant shape coordinate for one operator.

    Dense token-wise operators see flattened tokens. Prefill attention sees
    query-by-key work, while Decode attention reads the existing KV sequence.
    做算子性能建模/基准测试时，每个算子不要用完整张量形状，而是提取一个“主导形状坐标”，也就是决定它计算量大小的那个主要维度。
    """
    if stage == "prefill":
        return float(batch_size * length * length if operator == "attention"
                     else batch_size * length)
    if stage == "decode":
        return float(batch_size * length if operator == "attention" else batch_size)
    raise ValueError(f"unsupported stage: {stage}")


def _interpolate_1d(points: Mapping[float, Sequence[float]], query: float) -> Tuple[float, bool]:
    """
    对数空间插值，避免大形状下的线性插值过度膨胀。
    points: {x: [y1, y2, ...]}，x 是主导形状坐标，y 是对应的算子耗时。
    query: 待插值的主导形状坐标。
    返回: (插值结果, 是否外推)
    """
    xs = sorted(points)
    if not xs:
        raise ValueError("operator profile has no interpolation points")
    values = {x: statistics.fmean(float(value) for value in points[x]) for x in xs}
    if query in values:
        return values[query], False
    if len(xs) == 1:
        return values[xs[0]], query != xs[0]
    outside = query < xs[0] or query > xs[-1]
    if query <= xs[0]:
        low, high = xs[0], xs[1]
    elif query >= xs[-1]:
        low, high = xs[-2], xs[-1]
    else:
        index = bisect_right(xs, query)
        low, high = xs[index - 1], xs[index]
    fraction = ((math.log(max(query, 1e-9)) - math.log(max(low, 1e-9))) /
                (math.log(max(high, 1e-9)) - math.log(max(low, 1e-9)))) # 对数
    fraction = min(max(fraction, -1.0), 2.0)
    return max(values[low] + (values[high] - values[low]) * fraction, 0.0), outside


def interpolate_operator_table(
    table: Sequence[Mapping], stage: str, batch_size: int, length: int,
) -> Tuple[Dict[str, float], bool]:
    """Interpolate every ``operator::<name>`` field using its work coordinate."""
    length_key = "prompt_length" if stage == "prefill" else "kv_length"
    operators = sorted({
        key.split("::", 1)[1]
        for row in table for key in row if key.startswith("operator::")
    })
    if not operators:
        raise ValueError("schema-v2 profile does not contain operator detail")
    result: Dict[str, float] = {}
    extrapolated = False
    for operator in operators:
        points: Dict[float, List[float]] = {}
        key = f"operator::{operator}"
        for row in table:
            if key not in row:
                continue
            work = operator_work(
                stage, operator, int(row["batch_size"]), int(row[length_key]))
            points.setdefault(work, []).append(float(row[key]))
        query = operator_work(stage, operator, batch_size, length)
        value, outside = _interpolate_1d(points, query)
        result[operator] = value
        extrapolated = extrapolated or outside
    return result, extrapolated


def group_operator_times(operators: Mapping[str, float]) -> Dict[str, float]:
    attention = sum(float(value) for name, value in operators.items()
                    if name == "attention" or name.startswith("attention_") or
                    name in {"qkv_projection", "output_projection"})
    ffn = sum(float(value) for name, value in operators.items()
              if name.startswith("ffn_"))
    other = sum(float(value) for name, value in operators.items()) - attention - ffn
    return {"attention_ms": attention, "ffn_ms": ffn, "other_ms": max(other, 0.0)}
