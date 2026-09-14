#!/usr/bin/env bash
# Launch the vLLM 0.19 reference 1P1D stack for Qwen3-32B on one 8-GPU host.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/home/public/weight/Qwen3-32B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_PATH}"
HOST_IP="${VLLM_HOST_IP:-}"
if [[ -z "$HOST_IP" ]]; then
  # P2pNcclEngine binds to vLLM's resolved host IP, not necessarily loopback.
  # Resolve the same routable address here so the proxy-encoded KV addresses
  # and the ZMQ listeners cannot silently diverge.
  HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
if [[ -z "$HOST_IP" || "$HOST_IP" == "127."* || "$HOST_IP" == "localhost" ]]; then
  echo "Could not resolve a non-loopback KV host. Set VLLM_HOST_IP to this server's reachable IP." >&2
  exit 2
fi
# Force vLLM's get_ip() and our proxy to use exactly the same address.
export VLLM_HOST_IP="$HOST_IP"
PREFILL_GPUS="${PREFILL_GPUS:-0,1,2,3}"
DECODE_GPUS="${DECODE_GPUS:-4,5,6,7}"
PREFILL_HTTP_PORT="${PREFILL_HTTP_PORT:-8100}"
DECODE_HTTP_PORT="${DECODE_HTTP_PORT:-8200}"
PROXY_HTTP_PORT="${PROXY_HTTP_PORT:-8000}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-14579}"
DECODE_KV_PORT="${DECODE_KV_PORT:-14589}"
PROXY_CONTROL_PORT="${PROXY_CONTROL_PORT:-30001}"
LOG_ROOT="${LOG_ROOT:-data/ground_truth_pd/runtime_logs/tp4_1p1d}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROXY_SCRIPT="$SCRIPT_DIR/pd_proxy_vllm019.py"
REQUEST_ID_PATCH="$SCRIPT_DIR/patch_vllm019_p2p_request_id.py"
KV_TIMING_PATCH="$SCRIPT_DIR/patch_vllm019_p2p_timing.py"

if [[ "$(vllm --version)" != "0.19.0" ]]; then
  echo "This launch profile requires vLLM 0.19.0; found $(vllm --version)" >&2
  exit 2
fi
if ! python "$REQUEST_ID_PATCH" --check; then
  echo "vLLM 0.19 P2P request-id fix is not installed." >&2
  echo "Run: python $REQUEST_ID_PATCH --apply" >&2
  exit 2
fi
if ! python "$KV_TIMING_PATCH" --check; then
  echo "vLLM 0.19 remote-KV timing instrumentation is not installed." >&2
  echo "Run: python $KV_TIMING_PATCH --apply" >&2
  exit 2
fi
python -c "import aiohttp, quart" >/dev/null

# P2pNcclConnector binds base_port + TP rank on every worker. With TP4 each
# side therefore needs four consecutive, non-overlapping TCP ports.
if (( PREFILL_KV_PORT <= DECODE_KV_PORT + 3 && \
      DECODE_KV_PORT <= PREFILL_KV_PORT + 3 )); then
  echo "P/D KV port ranges overlap: ${PREFILL_KV_PORT}-$((PREFILL_KV_PORT + 3)) and ${DECODE_KV_PORT}-$((DECODE_KV_PORT + 3))" >&2
  exit 2
fi
if command -v ss >/dev/null 2>&1; then
  PORTS=("$PREFILL_HTTP_PORT" "$DECODE_HTTP_PORT" "$PROXY_HTTP_PORT"
         "$PROXY_CONTROL_PORT")
  for rank in 0 1 2 3; do
    PORTS+=("$((PREFILL_KV_PORT + rank))" "$((DECODE_KV_PORT + rank))")
  done
  for port in "${PORTS[@]}"; do
    if ss -ltnH "sport = :$port" | grep -q .; then
      echo "TCP port $port is already occupied:" >&2
      ss -ltnp "sport = :$port" >&2 || true
      exit 2
    fi
  done
fi

mkdir -p "$LOG_ROOT"
PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_worker() {
  local name="$1"
  local port="$2"
  local pid="$3"
  local attempt
  for attempt in $(seq 1 1200); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited before becoming healthy; inspect $LOG_ROOT/${name}.log" >&2
      return 1
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/v1/models" \
        >/dev/null 2>&1; then
      echo "$name is ready on port $port"
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for $name" >&2
  return 1
}

PREFILL_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_rank\":0,\"kv_parallel_size\":2,\"kv_buffer_size\":\"1e9\",\"kv_port\":\"${PREFILL_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${HOST_IP}\",\"proxy_port\":\"${PROXY_CONTROL_PORT}\",\"http_ip\":\"${HOST_IP}\",\"http_port\":\"${PREFILL_HTTP_PORT}\",\"send_type\":\"PUT_ASYNC\"}}"
DECODE_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_rank\":1,\"kv_parallel_size\":2,\"kv_buffer_size\":\"1e10\",\"kv_port\":\"${DECODE_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${HOST_IP}\",\"proxy_port\":\"${PROXY_CONTROL_PORT}\",\"http_ip\":\"${HOST_IP}\",\"http_port\":\"${DECODE_HTTP_PORT}\",\"send_type\":\"PUT_ASYNC\"}}"

CUDA_VISIBLE_DEVICES="$PREFILL_GPUS" vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PREFILL_HTTP_PORT" \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --block-size 16 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --enforce-eager \
  --trust-remote-code \
  --kv-transfer-config "$PREFILL_KV_CONFIG" \
  >"$LOG_ROOT/prefill.log" 2>&1 &
PREFILL_PID=$!
PIDS+=("$PREFILL_PID")

CUDA_VISIBLE_DEVICES="$DECODE_GPUS" vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$DECODE_HTTP_PORT" \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --block-size 16 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --trust-remote-code \
  --kv-transfer-config "$DECODE_KV_CONFIG" \
  >"$LOG_ROOT/decode.log" 2>&1 &
DECODE_PID=$!
PIDS+=("$DECODE_PID")

wait_for_worker prefill "$PREFILL_HTTP_PORT" "$PREFILL_PID"
wait_for_worker decode "$DECODE_HTTP_PORT" "$DECODE_PID"

python "$PROXY_SCRIPT" \
  --port "$PROXY_HTTP_PORT" \
  --prefill-url "http://${HOST_IP}:${PREFILL_HTTP_PORT}" \
  --decode-url "http://${HOST_IP}:${DECODE_HTTP_PORT}" \
  --kv-host "$HOST_IP" \
  --prefill-kv-port "$PREFILL_KV_PORT" \
  --decode-kv-port "$DECODE_KV_PORT" \
  >"$LOG_ROOT/proxy.log" 2>&1 &
PROXY_PID=$!
PIDS+=("$PROXY_PID")

sleep 2
if ! kill -0 "$PROXY_PID" 2>/dev/null; then
  echo "PD proxy failed to start; inspect $LOG_ROOT/proxy.log" >&2
  exit 1
fi

echo "PD stack is ready: http://127.0.0.1:${PROXY_HTTP_PORT}/v1/completions"
echo "Logs: $LOG_ROOT"
echo "Keep this terminal open; press Ctrl+C to stop exactly these three processes."
wait "$PROXY_PID"
