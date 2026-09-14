#!/usr/bin/env python3
"""Create a reviewable BenchmarkSpec from a Hugging Face model config.

This is deliberately a deterministic recommender.  An LLM/agent may explain
or propose edits later, but it must not silently change the fair iso-workload
contract used to rank hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.benchmark_protocol import PROTOCOL_VERSION, attention_type


def _value(config, *names, default=None):
    for name in names:
        if config.get(name) is not None:
            return config[name]
    return default


def recommend(config: dict, model_id: str, dtype: str = "bfloat16") -> dict:
    heads = int(_value(config, "num_attention_heads", "n_head"))
    kv_heads = int(_value(config, "num_key_value_heads", default=heads))
    hidden = int(_value(config, "hidden_size", "n_embd"))
    layers = int(_value(config, "num_hidden_layers", "n_layer"))
    intermediate = int(_value(config, "intermediate_size", "n_inner", default=4 * hidden))
    context = int(_value(config, "max_position_embeddings", "n_positions", default=4096))
    architectures = " ".join(config.get("architectures", []))
    model_type = str(config.get("model_type", "unknown"))
    signature = f"{architectures} {model_type}".lower()
    special_attention = any(word in signature for word in ("mamba", "linear", "hybrid"))
    is_moe = bool(_value(config, "num_experts", "num_local_experts", default=0))
    family = "special_or_hybrid" if special_attention else attention_type(heads, kv_heads)

    short = min(512, context)
    medium = min(1024, context)
    long = min(4096, context)
    review_reasons = []
    if special_attention:
        review_reasons.append("linear/state-space/hybrid attention needs a model-family adapter")
    if is_moe:
        review_reasons.append("MoE routing distribution and expert parallelism need explicit cases")
    if context < 4096:
        review_reasons.append("model context limit shortened the standard long-context cases")

    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_name": f"{model_id} recommended P/D benchmark",
        "model": {
            "name": model_id.rsplit("/", 1)[-1],
            "model_id": model_id,
            "architecture": "moe_decoder_transformer" if is_moe else "dense_decoder_transformer",
            "attention_type": family,
            "num_hidden_layers": layers,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": int(_value(config, "head_dim", default=hidden // heads)),
            "max_position_embeddings": context,
            "dtype": dtype,
        },
        "fairness": {
            "primary_track": "iso_workload",
            "iso_workload_rule": "model, dtype, TP, batch size, length, kernel semantics and software protocol must match",
            "oom_policy": "mark infeasible; do not reduce batch in the primary score",
            "secondary_track": "best_feasible",
        },
        "workloads": {
            "prefill": [
                {"case": "P_S", "batch_size": 1, "prompt_length": short, "purpose": "latency"},
                {"case": "P_M", "batch_size": 4, "prompt_length": medium, "purpose": "balanced"},
                {"case": "P_L", "batch_size": 8, "prompt_length": long, "purpose": "long context"},
                {"case": "P_XL", "batch_size": 16, "prompt_length": long, "purpose": "memory pressure"},
            ],
            "decode": [
                {"case": "D_S", "batch_size": 1, "kv_length": short, "purpose": "latency"},
                {"case": "D_M", "batch_size": 8, "kv_length": medium, "purpose": "balanced"},
                {"case": "D_L", "batch_size": 16, "kv_length": long, "purpose": "KV pressure"},
                {"case": "D_XL", "batch_size": 32, "kv_length": long, "purpose": "concurrency"},
            ],
        },
        "score": {
            "per_case": "candidate_throughput / reference_throughput",
            "stage_aggregation": "geometric_mean",
            "combined_score": None,
        },
        "recommendation": {
            "method": "deterministic_config_rules_v0.1",
            "review_required": bool(review_reasons),
            "review_reasons": review_reasons,
            "agent_role": "explain and propose reviewed changes; never silently mutate scored workloads",
            "decision_trace": [
                f"attention family derived from q_heads={heads}, kv_heads={kv_heads}: {family}",
                f"lengths capped by max_position_embeddings={context}",
                "P and D are kept as separate score families",
                "iso-workload and best-feasible tracks are separated",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True, help="local Hugging Face config.json")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    result = recommend(config, args.model_id, args.dtype)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"attention_type={result['model']['attention_type']}")
    print(f"review_required={result['recommendation']['review_required']}")
    print(f"spec: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

