#!/usr/bin/env bash
# Collect stage-2 operator/collective profiles and validate interpolation holdouts.
"""采集训练网格和独立 Holdout 网格
"""

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to the local model directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/stage2_profiling}"
TP_SIZES="${TP_SIZES:-1 2 4 8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
DTYPE="${DTYPE:-bfloat16}"

devices_for_tp() {
  local count="$1"
  awk -F, -v n="$count" '{for(i=1;i<=n;i++){printf "%s%s", $i, (i<n ? "," : "")}}' \
    <<<"$GPU_LIST"
}

mkdir -p "$OUTPUT_ROOT/training" "$OUTPUT_ROOT/holdout" \
  "$OUTPUT_ROOT/collectives" "$OUTPUT_ROOT/holdout_reports"

for tp in $TP_SIZES; do
  echo "[TP${tp}] operator training grid"
  CUDA_VISIBLE_DEVICES="$(devices_for_tp "$tp")" python scripts/operator_micro_bench.py \
    --model "$MODEL_PATH" --tp-size "$tp" --stage both --dtype "$DTYPE" \
    --batch-sizes 1,2,4,8,16,32 \
    --prefill-lengths 128,512,1024,2048,4096 \
    --decode-kv-lengths 128,512,1024,2048,4096 \
    --output-dir "$OUTPUT_ROOT/training/tp${tp}"

  echo "[TP${tp}] independent operator holdout grid"
  CUDA_VISIBLE_DEVICES="$(devices_for_tp "$tp")" python scripts/operator_micro_bench.py \
    --model "$MODEL_PATH" --tp-size "$tp" --stage both --dtype "$DTYPE" \
    --batch-sizes 3,6,12,24 \
    --prefill-lengths 256,768,1536,3072 \
    --decode-kv-lengths 256,768,1536,3072 \
    --output-dir "$OUTPUT_ROOT/holdout/tp${tp}"

  echo "[TP${tp}] NCCL collective grid"
  CUDA_VISIBLE_DEVICES="$(devices_for_tp "$tp")" torchrun --standalone \
    --nproc-per-node "$tp" scripts/collective_micro_bench.py \
    --dtype "$DTYPE" --output-dir "$OUTPUT_ROOT/collectives/tp${tp}"

  python scripts/validate_operator_holdout.py \
    --training-profile "$OUTPUT_ROOT/training/tp${tp}/operator_profile_tp${tp}.json" \
    --holdout-profile "$OUTPUT_ROOT/holdout/tp${tp}/operator_profile_tp${tp}.json" \
    --output-dir "$OUTPUT_ROOT/holdout_reports/tp${tp}"
done

echo "Stage-2 profiling completed under $OUTPUT_ROOT"
