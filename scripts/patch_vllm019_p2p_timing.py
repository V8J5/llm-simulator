#!/usr/bin/env python3
"""Instrument vLLM 0.19 P2pNcclConnector's blocking receive path.

P2pNcclConnector reports ``load_async=False`` to the scheduler, so the generic
WAITING_FOR_REMOTE_KVS transitions are not used.  The real wait happens in the
decode worker's ``start_load_kv`` method.  This patch logs request-level start
and completion timestamps around that method's per-layer recv_tensor loop.
Only tensor-parallel local rank zero logs, avoiding duplicate request events.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


BACKUP_SUFFIX = ".llm-simulator-p2p-timing-backup"
START_MARKER = "PD_KV_EVENT event=recv_start"
COMPLETE_MARKER = "PD_KV_EVENT event=recv_complete"

IMPORT_ANCHORS = {"import regex as re", "import re"}
START_ANCHOR = """        for request in metadata.requests:
            request_id = request.request_id"""
COMPLETE_ANCHOR = """                inject_kv_into_layer(
                    layer, kv_cache, request.block_ids, request.request_id
                )"""


def target_file() -> Path:
    version = importlib.metadata.version("vllm")
    if version != "0.19.0":
        raise RuntimeError(
            f"this instrumentation patch only supports vLLM 0.19.0, found {version}")
    spec = importlib.util.find_spec(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector")
    if spec is None or spec.origin is None:
        raise RuntimeError("could not locate vLLM P2pNcclConnector")
    return Path(spec.origin).resolve()


def state(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    markers = START_MARKER in text, COMPLETE_MARKER in text
    if all(markers):
        return "patched"
    if any(markers):
        return "partial"
    if text.count(START_ANCHOR) == 1 and text.count(COMPLETE_ANCHOR) == 1:
        return "unpatched"
    return "unknown"


def instrument_text(text: str) -> str:
    if START_MARKER in text and COMPLETE_MARKER in text:
        return text
    counts = {
        "start": text.count(START_ANCHOR),
        "complete": text.count(COMPLETE_ANCHOR),
    }
    if counts != {"start": 1, "complete": 1}:
        raise RuntimeError(
            "refusing to patch an unrecognized P2pNcclConnector; "
            f"expected each anchor once, found {counts}")
    if "import time" not in text:
        lines = text.splitlines(keepends=True)
        import_indices = [
            index for index, line in enumerate(lines)
            if line.strip() in IMPORT_ANCHORS
        ]
        if len(import_indices) != 1:
            raise RuntimeError("could not safely insert the time import")
        index = import_indices[0]
        newline = "\r\n" if lines[index].endswith("\r\n") else "\n"
        lines.insert(index + 1, f"import time{newline}")
        text = "".join(lines)

    start = START_ANCHOR + """
            if self._local_rank == 0:
                logger.info(
                    "PD_KV_EVENT event=recv_start request_id=%s "
                    "monotonic_s=%.9f",
                    request_id,
                    time.monotonic(),
                )"""
    complete = COMPLETE_ANCHOR + """
            if self._local_rank == 0:
                logger.info(
                    "PD_KV_EVENT event=recv_complete request_id=%s "
                    "monotonic_s=%.9f",
                    request_id,
                    time.monotonic(),
                )"""
    return text.replace(START_ANCHOR, start, 1).replace(
        COMPLETE_ANCHOR, complete, 1)


def apply(path: Path) -> None:
    current = state(path)
    if current == "patched":
        print(f"already patched: {path}")
        return
    if current != "unpatched":
        raise RuntimeError(
            f"refusing to modify connector in state {current}: {path}")
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    instrumented = instrument_text(path.read_text(encoding="utf-8"))
    compile(instrumented, str(path), "exec")
    path.write_text(instrumented, encoding="utf-8")
    validation = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm.distributed.kv_transfer.kv_connector.v1.p2p."
            "p2p_nccl_connector import P2pNcclConnector",
        ],
        capture_output=True,
        text=True,
    )
    if validation.returncode != 0:
        shutil.copy2(backup, path)
        detail = validation.stderr.strip() or validation.stdout.strip()
        raise RuntimeError(
            "patched connector failed a clean-process import and was "
            f"automatically restored: {detail}")
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
