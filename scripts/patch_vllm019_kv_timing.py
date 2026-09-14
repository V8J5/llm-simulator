#!/usr/bin/env python3
"""Instrument vLLM 0.19 remote-KV state transitions, with restore support.

The patch records monotonic timestamps when a decode request starts waiting for
remote KV, when the connector reports receive completion, and when the
scheduler promotes the request.  It changes logging only, not scheduling.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import re
import shutil
from pathlib import Path


BACKUP_SUFFIX = ".llm-simulator-kv-timing-backup"
WAIT_MARKER = "PD_KV_EVENT event=wait_start"
RECV_MARKER = "PD_KV_EVENT event=recv_finished"
PROMOTE_MARKER = "PD_KV_EVENT event=promoted"

WAIT_OLD = "request.status = RequestStatus.WAITING_FOR_REMOTE_KVS"
RECV_OLD = 'logger.debug("Finished recving KV transfer for request %s", req_id)'
PROMOTE_OLD = "self._update_waiting_for_remote_kv(request)"


def target_file() -> Path:
    version = importlib.metadata.version("vllm")
    if version != "0.19.0":
        raise RuntimeError(
            f"this instrumentation patch only supports vLLM 0.19.0, found {version}")
    spec = importlib.util.find_spec("vllm.v1.core.sched.scheduler")
    if spec is None or spec.origin is None:
        raise RuntimeError("could not locate vllm.v1.core.sched.scheduler")
    return Path(spec.origin).resolve()


def state(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    markers = (WAIT_MARKER in text, RECV_MARKER in text, PROMOTE_MARKER in text)
    if all(markers):
        return "patched"
    if any(markers):
        return "partial"
    if all(text.count(value) == 1 for value in (WAIT_OLD, RECV_OLD, PROMOTE_OLD)):
        return "unpatched"
    return "unknown"


def instrument_text(text: str) -> str:
    if all(marker in text for marker in (WAIT_MARKER, RECV_MARKER, PROMOTE_MARKER)):
        return text
    counts = {value: text.count(value)
              for value in (WAIT_OLD, RECV_OLD, PROMOTE_OLD)}
    if any(count != 1 for count in counts.values()):
        raise RuntimeError(
            "refusing to patch an unrecognized scheduler.py; expected each "
            f"anchor once, found {counts}")
    def insert_after(source: str, anchor: str, event: str,
                     request_expression: str) -> str:
        pattern = re.compile(
            rf"^(?P<indent>[ \t]*){re.escape(anchor)}[ \t]*$", re.MULTILINE)

        def replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            inner = indent + "    "
            return (
                match.group(0) + "\n" +
                indent + "logger.info(\n" +
                inner + f'"PD_KV_EVENT event={event} request_id=%s '
                'monotonic_s=%.9f",\n' +
                inner + request_expression + ",\n" +
                inner + "time.monotonic(),\n" +
                indent + ")")

        updated, count = pattern.subn(replacement, source, count=1)
        if count != 1:
            raise RuntimeError(f"could not insert instrumentation after: {anchor}")
        return updated

    text = insert_after(text, WAIT_OLD, "wait_start", "request_id")
    text = insert_after(text, RECV_OLD, "recv_finished", "req_id")
    return insert_after(
        text, PROMOTE_OLD, "promoted", "request.request_id")


def apply(path: Path) -> None:
    current = state(path)
    if current == "patched":
        print(f"already patched: {path}")
        return
    if current != "unpatched":
        raise RuntimeError(
            f"refusing to modify scheduler.py in state {current}: {path}")
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    instrumented = instrument_text(path.read_text(encoding="utf-8"))
    # Fail before touching site-packages if an upstream formatting difference
    # would make the generated module syntactically invalid.
    compile(instrumented, str(path), "exec")
    path.write_text(instrumented, encoding="utf-8")
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
