#!/usr/bin/env python3
"""Collect a Cartesian fixed-batch validation suite from one running vLLM TP."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def csv_ints(value: str) -> List[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--stages", nargs="+", choices=("prefill", "decode"),
                        default=["prefill", "decode"])
    parser.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("1,4,8"))
    parser.add_argument("--prefill-lengths", type=csv_ints,
                        default=csv_ints("128,512,1024"))
    parser.add_argument("--decode-kv-lengths", type=csv_ints,
                        default=csv_ints("128,512,1024"))
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--deployment-config", type=Path, required=True)
    parser.add_argument("--server-command-file", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Skip existing valid points whose shape and TP match")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def reusable_output(path: Path, args: argparse.Namespace, stage: str,
                    batch: int, length: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["configuration"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    length_key = "prompt_length" if stage == "prefill" else "initial_kv_length"
    return (
        payload.get("schema_version") == 2 and
        payload.get("collector") == "vllm_fixed_batch_streaming" and
        payload.get("valid") is True and
        payload.get("model") == args.model and
        config.get("stage") == stage and
        int(config.get("tp_size", -1)) == args.tp_size and
        int(config.get("batch_size", -1)) == batch and
        int(config.get(length_key, -1)) == length and
        "request-unique first 16-token block" in config.get("prompt_policy", "")
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    stage_lengths = {
        "prefill": args.prefill_lengths,
        "decode": args.decode_kv_lengths,
    }
    for stage in args.stages:
        lengths = stage_lengths[stage]
        for batch in args.batch_sizes:
            for length in lengths:
                output = args.output_dir / f"{stage}_b{batch}_l{length}.json"
                if args.resume and reusable_output(
                        output, args, stage, batch, length):
                    print(f"reuse valid {stage}: TP={args.tp_size}, "
                          f"batch={batch}, length={length}")
                    outputs.append(str(output.resolve()))
                    continue
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "collect_fixed_batch_vllm.py"),
                    "--base-url", args.base_url,
                    "--api-key", args.api_key,
                    "--model", args.model,
                    "--tokenizer", args.tokenizer or args.model,
                    "--stage", stage,
                    "--tp-size", str(args.tp_size),
                    "--batch-size", str(batch),
                    "--length", str(length),
                    "--decode-tokens", str(args.decode_tokens),
                    "--warmup", str(args.warmup),
                    "--repeats", str(args.repeats),
                    "--dtype", args.dtype,
                    "--deployment-config", str(args.deployment_config),
                    "--output", str(output),
                ]
                if args.server_command_file:
                    command.extend(["--server-command-file", str(args.server_command_file)])
                print(f"collect {stage}: TP={args.tp_size}, batch={batch}, length={length}")
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                outputs.append(str(output.resolve()))
    manifest = {
        "schema_version": 1,
        "experiment_type": "fixed_batch_suite_manifest",
        "tp_size": args.tp_size,
        "deployment_config": str(args.deployment_config.resolve()),
        "files": outputs,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(outputs)} fixed-batch points; manifest={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
