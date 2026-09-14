#!/usr/bin/env python3
"""Patch vLLM 0.19's PD request-id mismatch, with check/restore support.

P2pNcclConnector keys transferred tensors by the engine's internal request id.
vLLM 0.19 appends an independent random suffix in every P/D process, so the
producer and consumer use different tensor keys and wait forever.  The proxy
already supplies a globally unique id; retaining it is sufficient and makes
the two sides deterministic.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import shutil
from pathlib import Path


OLD = "request.request_id = f\"{request.external_req_id}-{random_uuid():.8}\""
NEW = "request.request_id = request.external_req_id  # llm-simulator PD fix"
BACKUP_SUFFIX = ".llm-simulator-pd-backup"


def target_file() -> Path:
    version = importlib.metadata.version("vllm")
    if version != "0.19.0":
        raise RuntimeError(
            f"this compatibility patch only supports vLLM 0.19.0, found {version}")
    spec = importlib.util.find_spec("vllm.v1.engine.input_processor")
    if spec is None or spec.origin is None:
        raise RuntimeError("could not locate vllm.v1.engine.input_processor")
    return Path(spec.origin).resolve()


def state(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        return "patched"
    if OLD in text:
        return "unpatched"
    return "unknown"


def apply(path: Path) -> None:
    current = state(path)
    if current == "patched":
        print(f"already patched: {path}")
        return
    if current != "unpatched":
        raise RuntimeError(
            f"refusing to modify an unrecognized input_processor.py: {path}")
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patched: {path}")
    print(f"backup:  {backup}")


def restore(path: Path) -> None:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        raise RuntimeError(f"backup does not exist: {backup}")
    shutil.copy2(backup, path)
    print(f"restored: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    path = target_file()
    if args.apply:
        apply(path)
        return 0
    if args.restore:
        restore(path)
        return 0
    current = state(path)
    print(f"{current}: {path}")
    return 0 if current == "patched" else 1


if __name__ == "__main__":
    raise SystemExit(main())
