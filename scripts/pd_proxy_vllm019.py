#!/usr/bin/env python3
"""Minimal vLLM 0.19 P2pNcclConnector 1P1D completion proxy.

Protocol behavior follows vLLM v0.19.0's P2P xPyD design: issue a one-token
prefill request and immediately start the original decode request, using a
shared X-Request-Id and X-KV-Target header. Decode must be active while
PUT_ASYNC transfers multi-block KV tensors or the two phases can deadlock.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from urllib.parse import urlparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefill-url", default="http://127.0.0.1:8100")
    parser.add_argument("--decode-url", default="http://127.0.0.1:8200")
    parser.add_argument("--kv-host", default="127.0.0.1")
    parser.add_argument("--prefill-kv-port", type=int, default=14579)
    parser.add_argument("--decode-kv-port", type=int, default=14580)
    parser.add_argument("--timeout", type=float, default=6 * 60 * 60)
    return parser.parse_args()


def host_port(url: str) -> str:
    parsed = urlparse(url)
    port = parsed.port or (80 if parsed.scheme == "http" else 443)
    return f"{parsed.hostname or '127.0.0.1'}:{port}"


def build_request_id(prefill_kv_addr: str, decode_kv_addr: str) -> str:
    return (f"___prefill_addr_{prefill_kv_addr}___decode_addr_"
            f"{decode_kv_addr}_{uuid.uuid4().hex}")


def create_app(args):
    import aiohttp
    from quart import Quart, Response, make_response, request

    app = Quart(__name__)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    prefill_base = args.prefill_url.rstrip("/")
    decode_base = args.decode_url.rstrip("/")
    prefill_kv_addr = f"{args.kv_host}:{args.prefill_kv_port}"
    decode_kv_addr = f"{args.kv_host}:{args.decode_kv_port}"
    kv_target = host_port(args.decode_url)

    def headers(request_id: str):
        value = {"X-Request-Id": request_id, "X-KV-Target": kv_target}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            value["Authorization"] = f"Bearer {api_key}"
        return value

    async def run_prefill(path, payload, request_headers, request_id, timing):
        started = time.perf_counter()
        timing["prefill_start"] = started
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                    f"{prefill_base}{path}", json=payload,
                    headers=request_headers) as response:
                body = await response.read()
                if response.status != 200:
                    raise RuntimeError(
                        f"prefill HTTP {response.status}: "
                        f"{body.decode(errors='replace')}")
        timing["prefill_complete"] = time.perf_counter()
        app.logger.info("prefill %s completed in %.3f s", request_id,
                        timing["prefill_complete"] - started)

    async def stream_decode(path, payload, request_headers, request_id, timing):
        try:
            timing["decode_start"] = time.perf_counter()
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                        f"{decode_base}{path}", json=payload,
                        headers=request_headers) as response:
                    if response.status != 200:
                        body = await response.text()
                        app.logger.error("decode HTTP %s: %s",
                                         response.status, body)
                        yield (f'{{"error":"decode HTTP '
                               f'{response.status}"}}').encode()
                        return
                    timing["decode_headers"] = time.perf_counter()
                    async for chunk in response.content.iter_any():
                        timing.setdefault("decode_first_byte", time.perf_counter())
                        yield chunk
            timing["decode_complete"] = time.perf_counter()
            app.logger.info("decode %s completed", request_id)
        except asyncio.TimeoutError:
            app.logger.exception("decode timeout for %s", request_id)
            yield b'{"error":"decode timeout"}'
        except aiohttp.ClientError:
            app.logger.exception("decode unavailable for %s", request_id)
            yield b'{"error":"decode unavailable"}'

    async def stream_pd(path, prefill_payload, decode_payload,
                        request_headers, request_id, client_request_id,
                        timing):
        """Run P and D concurrently so PUT_ASYNC always has a receiver."""
        timing_emitted = False

        def emit_timing():
            """Emit exactly once, including when the client stops at [DONE]."""
            nonlocal timing_emitted
            if timing_emitted:
                return
            timing_emitted = True
            timing["proxy_complete"] = time.perf_counter()
            origin = timing["proxy_received"]

            def elapsed(name):
                value = timing.get(name)
                return None if value is None else round(
                    (value - origin) * 1000.0, 3)

            app.logger.info(
                "PD_TIMING client_request_id=%s internal_request_id=%s "
                "proxy_received_monotonic_s=%.9f "
                "prefill_start_ms=%s prefill_complete_ms=%s "
                "decode_start_ms=%s decode_headers_ms=%s "
                "decode_first_byte_ms=%s decode_complete_ms=%s "
                "proxy_complete_ms=%s",
                client_request_id, request_id, origin,
                elapsed("prefill_start"), elapsed("prefill_complete"),
                elapsed("decode_start"), elapsed("decode_headers"),
                elapsed("decode_first_byte"), elapsed("decode_complete"),
                elapsed("proxy_complete"))

        prefill_task = asyncio.create_task(run_prefill(
            path, prefill_payload, request_headers, request_id, timing))
        # Let the prefill POST reach its first I/O wait before opening decode.
        await asyncio.sleep(0)
        stream_tail = b""
        try:
            async for chunk in stream_decode(
                    path, decode_payload, request_headers, request_id, timing):
                # The OpenAI client is allowed to stop consuming as soon as it
                # sees this sentinel.  Log before forwarding it; otherwise the
                # async generator's cleanup is not guaranteed to run promptly.
                stream_tail = (stream_tail + chunk)[-64:]
                if b"data: [DONE]" in stream_tail:
                    timing.setdefault("decode_complete", time.perf_counter())
                    emit_timing()
                yield chunk
            await prefill_task
        except Exception:
            app.logger.exception("PD stream failed for %s", request_id)
            yield b'{"error":"PD stream failed"}'
        finally:
            if not prefill_task.done():
                prefill_task.cancel()
            await asyncio.gather(prefill_task, return_exceptions=True)
            emit_timing()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions():
        try:
            original = await request.get_json()
            if not isinstance(original, dict):
                return Response("invalid JSON body", status=400)
            prefill = dict(original)
            prefill["max_tokens"] = 1
            if "max_completion_tokens" in prefill:
                prefill["max_completion_tokens"] = 1
            client_request_id = original.get("request_id")
            request_id = build_request_id(prefill_kv_addr, decode_kv_addr)
            request_headers = headers(request_id)
            timing = {"proxy_received": time.perf_counter()}
            response = await make_response(stream_pd(
                request.path, prefill, original, request_headers, request_id,
                client_request_id, timing))
            response.timeout = None
            response.content_type = "text/event-stream"
            return response
        except asyncio.CancelledError:
            return Response('{"error":"request cancelled"}', status=503,
                            content_type="application/json")
        except Exception:
            app.logger.exception("PD request failed")
            return Response('{"error":"internal proxy error"}', status=500,
                            content_type="application/json")

    return app


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    create_app(args).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
