"""Optional bounded replica proxy; install with ``pip install 'bilgeval[proxy]'``.

Replicas must serve identical weights, tokenizer and settings. Model-name checks
cannot establish that equivalence. Retries can duplicate upstream computation.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
import logging
import math
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import quote

import httpx


_LOGGER = logging.getLogger(__name__)
_HOP_HEADERS = {
    b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
    b"te", b"trailer", b"transfer-encoding", b"upgrade", b"host", b"content-length",
}
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_ROUTES = {
    "/models": ["GET"],
    "/v1/models": ["GET"],
    "/v1/completions": ["POST"],
    "/v1/chat/completions": ["POST"],
    "/tokenize": ["POST"],
    "/get_model_info": ["GET"],
    "/generate": ["POST"],
}


def _normalize_endpoint(value: str) -> str:
    try:
        if not isinstance(value, str) or any(char.isspace() for char in value):
            raise ValueError
        url = httpx.URL(value)
        if (url.scheme not in {"http", "https"} or not url.host or url.userinfo
                or url.query or url.fragment):
            raise ValueError
        path = url.path.rstrip("/") or "/v1"
        return str(url.copy_with(path=path))
    except (ValueError, httpx.InvalidURL):
        raise ValueError("Endpoint must be an absolute HTTP(S) API base without credentials, query or fragment") from None


def _normalize_endpoints(values) -> tuple[str, ...]:
    endpoints = tuple(_normalize_endpoint(value) for value in values)
    if not endpoints:
        raise ValueError("At least one endpoint is required")
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("Duplicate endpoints are not allowed")
    return endpoints


def read_endpoints(path: str | Path) -> tuple[str, ...]:
    """Read one API base per line, ignoring blank lines and full-line comments."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return _normalize_endpoints(line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#"))


@dataclass(frozen=True)
class ProxyConfig:
    endpoints: tuple[str, ...]
    model: str
    capacity: int = 16
    timeout: float = 600.0
    queue_timeout: float = 60.0
    retries: int = 1
    cooldown: float = 15.0
    api_key: str = field(default_factory=lambda: os.environ.get("BILGEVAL_API_KEY") or "EMPTY", repr=False)

    def __post_init__(self):
        object.__setattr__(self, "endpoints", _normalize_endpoints(self.endpoints))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Model must be a nonempty advertised model name")
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        if isinstance(self.retries, bool) or not isinstance(self.retries, int) or self.retries < 0:
            raise ValueError("Retries must be a nonnegative integer")
        for name in ("timeout", "queue_timeout", "cooldown"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass
class _Endpoint:
    url: str
    active: int = 0
    requests: int = 0
    completed: int = 0
    failures: int = 0
    total_latency_seconds: float = 0.0
    cooldown_until: float = 0.0


class _CapacityTimeout(Exception):
    pass


class _Router:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.endpoints = [_Endpoint(url) for url in config.endpoints]
        self.condition = asyncio.Condition()
        self.next_tie = 0

    async def acquire(self, excluded: set[str]) -> _Endpoint:
        deadline = time.monotonic() + self.config.queue_timeout
        async with self.condition:
            while True:
                now = time.monotonic()
                available = [
                    (index, endpoint) for index, endpoint in enumerate(self.endpoints)
                    if endpoint.active < self.config.capacity and endpoint.cooldown_until <= now
                ]
                preferred = [(i, e) for i, e in available if e.url not in excluded]
                available = preferred or available
                if available:
                    index, endpoint = min(available, key=lambda item: (item[1].active, (item[0] - self.next_tie) % len(self.endpoints)))
                    self.next_tie = (index + 1) % len(self.endpoints)
                    endpoint.active += 1
                    endpoint.requests += 1
                    return endpoint
                remaining = deadline - now
                if remaining <= 0:
                    raise _CapacityTimeout
                cooling = [e.cooldown_until - now for e in self.endpoints if e.cooldown_until > now and e.active < self.config.capacity]
                delay = min([remaining, *cooling])
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.condition.wait(), timeout=delay)

    def failed(self, endpoint: _Endpoint):
        endpoint.failures += 1
        endpoint.cooldown_until = time.monotonic() + self.config.cooldown

    def has_alternative(self, endpoint: _Endpoint, attempted: set[str]) -> bool:
        now = time.monotonic()
        return any(e is not endpoint and e.url not in attempted and e.cooldown_until <= now for e in self.endpoints)

    async def release(self, endpoint: _Endpoint, started: float, completed: bool = False):
        async with self.condition:
            endpoint.active -= 1
            endpoint.completed += int(completed)
            endpoint.total_latency_seconds += time.monotonic() - started
            self.condition.notify_all()

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "model": self.config.model,
            "endpoints": [{
                "url": e.url,
                "active": e.active,
                "capacity": self.config.capacity,
                "healthy": e.cooldown_until <= now,
                "cooldown_remaining_seconds": max(0.0, e.cooldown_until - now),
                "requests": e.requests,
                "completed": e.completed,
                "failures": e.failures,
                "total_latency_seconds": e.total_latency_seconds,
            } for e in self.endpoints],
        }


def _end_to_end_headers(raw_headers) -> list[tuple[bytes, bytes]]:
    headers = [(name.lower(), value) for name, value in raw_headers]
    blocked = set(_HOP_HEADERS)
    for name, value in headers:
        if name == b"connection":
            blocked.update(token.strip().lower() for token in value.split(b","))
    return [(name, value) for name, value in headers if name not in blocked]


def _target_url(endpoint: str, path: str, query: bytes, raw_path: bytes | None) -> str:
    encoded_path = raw_path.decode("ascii") if raw_path is not None else quote(path, safe="/")
    if path == "/models":
        target = endpoint + encoded_path
    elif path.startswith("/v1/"):
        prefix_end = 0
        for _ in "/v1":
            prefix_end += 3 if encoded_path[prefix_end] == "%" else 1
        target = endpoint + encoded_path[prefix_end:]
    else:
        root = endpoint[:-3] if endpoint.endswith("/v1") else endpoint
        target = root + encoded_path
    return target + ("?" + query.decode("ascii") if query else "")


def create_app(config: ProxyConfig):
    """Construct the ASGI app. Lifespan checks every replica before accepting traffic."""
    try:
        import aiohttp
        import anyio
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse, Response, StreamingResponse
        from starlette.routing import Route
        from yarl import URL
    except ImportError:
        raise ImportError("The proxy requires optional dependencies: pip install 'bilgeval[proxy]'") from None

    router = _Router(config)

    @asynccontextmanager
    async def lifespan(app):
        resources = AsyncExitStack()
        app.state.clients = {}
        try:
            for endpoint in config.endpoints:
                connector = await resources.enter_async_context(aiohttp.TCPConnector(
                    limit=config.capacity, limit_per_host=config.capacity,
                    keepalive_timeout=5.0,
                ))
                app.state.clients[endpoint] = await resources.enter_async_context(aiohttp.ClientSession(
                    connector=connector, connector_owner=False,
                    timeout=aiohttp.ClientTimeout(total=None, sock_read=config.timeout),
                    auto_decompress=False, trust_env=False,
                    cookie_jar=aiohttp.DummyCookieJar(),
                    skip_auto_headers={"Accept", "Accept-Encoding", "User-Agent", "Content-Type"},
                    headers={"Authorization": f"Bearer {config.api_key}"},
                ))

            async def validate(endpoint):
                try:
                    with anyio.fail_after(config.timeout):
                        async with app.state.clients[endpoint].get(
                            URL(endpoint + "/models", encoded=True), allow_redirects=False,
                        ) as response:
                            if not 200 <= response.status < 300:
                                raise ValueError("Model discovery requires a successful response")
                            payload = await response.json(content_type=None)
                    names = {entry["id"] for entry in payload["data"]}
                except (aiohttp.ClientError, TimeoutError, ValueError, TypeError, KeyError):
                    raise ValueError(f"Endpoint {endpoint} model discovery failed") from None
                if config.model not in names:
                    raise ValueError(f"Endpoint {endpoint} does not advertise expected model {config.model!r}")
            results = await asyncio.gather(*(validate(endpoint) for endpoint in config.endpoints), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            yield
        finally:
            with anyio.CancelScope(shield=True):
                await resources.aclose()

    class UpstreamResponse(StreamingResponse):
        def __init__(self, upstream, endpoint, started):
            self.upstream = upstream
            self.endpoint = endpoint
            self.started = started
            self.closed = False
            self.fully_read = False
            super().__init__(self.chunks(), status_code=upstream.status)
            self.raw_headers = _end_to_end_headers(upstream.raw_headers)

        async def chunks(self):
            try:
                async for chunk in self.upstream.content.iter_any():
                    yield chunk
                self.fully_read = True
            except (aiohttp.ClientError, TimeoutError) as exc:
                router.failed(self.endpoint)
                _LOGGER.warning("Upstream transport failure exception=%s endpoint=%s", type(exc).__name__, self.endpoint.url)
                raise RuntimeError("Upstream response stream interrupted") from None

        async def close(self):
            if not self.closed:
                self.closed = True
                with anyio.CancelScope(shield=True):
                    try:
                        self.upstream.close()
                    finally:
                        await router.release(self.endpoint, self.started, self.fully_read)

        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                await self.close()

    async def prepare(request, body, headers):
        attempted: set[str] = set()
        for attempt in range(config.retries + 1):
            try:
                endpoint = await router.acquire(attempted)
            except _CapacityTimeout:
                return JSONResponse({"error": {"message": "No replica capacity available before queue timeout", "type": "proxy_unavailable"}}, status_code=503)
            started = time.monotonic()
            upstream = None
            transferred = False
            try:
                attempted.add(endpoint.url)
                try:
                    with anyio.fail_after(config.timeout):
                        upstream = await request.app.state.clients[endpoint.url].request(
                            request.method,
                            URL(_target_url(endpoint.url, request.url.path,
                                            request.scope["query_string"], request.scope.get("raw_path")), encoded=True),
                            data=body, headers=headers, allow_redirects=False,
                        )
                except (aiohttp.ClientError, TimeoutError) as exc:
                    router.failed(endpoint)
                    _LOGGER.warning("Upstream transport failure exception=%s endpoint=%s", type(exc).__name__, endpoint.url)
                    if attempt < config.retries and router.has_alternative(endpoint, attempted):
                        continue
                    status = 504 if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) else 502
                    return JSONResponse({"error": {"message": "Upstream request timed out" if status == 504 else "Upstream connection failed", "type": "proxy_upstream_error"}}, status_code=status)
                if upstream.status in _TRANSIENT_STATUS:
                    router.failed(endpoint)
                    if attempt < config.retries and router.has_alternative(endpoint, attempted):
                        continue
                result = UpstreamResponse(upstream, endpoint, started)
                transferred = True
                return result
            finally:
                if not transferred:
                    with anyio.CancelScope(shield=True):
                        try:
                            if upstream is not None:
                                upstream.close()
                        finally:
                            await router.release(endpoint, started)
        raise AssertionError("Retry loop exhausted without a response")

    async def forward(request):
        try:
            headers = [(name.decode("ascii"), value.decode("utf-8"))
                       for name, value in _end_to_end_headers(request.headers.raw)]
        except UnicodeDecodeError:
            return JSONResponse({"error": {"message": "Request headers must contain ASCII names and ASCII or valid UTF-8 values", "type": "proxy_invalid_headers"}}, status_code=400)
        body = await request.body()

        async def disconnected():
            while True:
                if (await request.receive())["type"] == "http.disconnect":
                    return

        worker = asyncio.create_task(prepare(request, body, headers))
        watcher = asyncio.create_task(disconnected())
        handed_off = False
        try:
            done, _ = await asyncio.wait({worker, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if worker in done:
                result = worker.result()
                handed_off = True
                return result
            return Response(status_code=499)
        finally:
            watcher.cancel()
            if not handed_off:
                worker.cancel()
                with anyio.CancelScope(shield=True):
                    with suppress(asyncio.CancelledError):
                        await watcher
                    with suppress(asyncio.CancelledError):
                        result = await worker
                        if isinstance(result, UpstreamResponse):
                            await result.close()

    async def stats(request):
        return JSONResponse(router.stats())

    async def health(request):
        data = router.stats()
        healthy = sum(e["healthy"] for e in data["endpoints"])
        data["status"] = "ok" if healthy == len(router.endpoints) else "degraded" if healthy else "unavailable"
        return JSONResponse(data, status_code=200 if healthy else 503)

    app = Starlette(
        routes=[Route("/health", health), Route("/stats", stats), *[Route(path, forward, methods=methods) for path, methods in _ROUTES.items()]],
        lifespan=lifespan,
    )
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Route OpenAI-compatible requests across identical model replicas. Retries may duplicate computation.")
    parser.add_argument("--endpoints-file", required=True, help="Text file with one HTTP(S) API base per line")
    parser.add_argument("--model", required=True, help="Expected model ID advertised by every replica")
    parser.add_argument("--api-key", default=os.environ.get("BILGEVAL_API_KEY") or "EMPTY", help="Replica API key (default: BILGEVAL_API_KEY or EMPTY); incoming Authorization overrides it")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--capacity", type=int, default=16, help="Maximum active requests per endpoint (default: 16)")
    parser.add_argument("--timeout", type=float, default=600, help="Seconds for the complete upstream operation through response headers (including upload), then per idle upstream read (default: 600)")
    parser.add_argument("--queue-timeout", type=float, default=60, help="Maximum wait for replica capacity in seconds (default: 60)")
    parser.add_argument("--retries", type=int, default=1, help="Maximum retries on other healthy replicas before response starts (default: 1)")
    parser.add_argument("--cooldown", type=float, default=15, help="Transient failure cooldown in seconds (default: 15)")
    parser.add_argument("--keepalive-timeout", type=float, default=60, help="Idle client connection lifetime in seconds; 0 disables reuse (default: 60). Configure upstream server keep-alive separately.")
    parser.add_argument("--shutdown-timeout", type=float, default=30, help="Grace period before cancelling active requests on shutdown, in seconds (default: 30)")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.shutdown_timeout) or args.shutdown_timeout <= 0:
            raise ValueError("Shutdown timeout must be positive and finite")
        if not math.isfinite(args.keepalive_timeout) or args.keepalive_timeout < 0:
            raise ValueError("Keep-alive timeout must be nonnegative and finite")
        if not 1 <= args.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        config = ProxyConfig(read_endpoints(args.endpoints_file), args.model, args.capacity, args.timeout, args.queue_timeout, args.retries, args.cooldown, args.api_key)
        app = create_app(config)
        import uvicorn
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    uvicorn.run(app, host=args.host, port=args.port, access_log=False,
                timeout_keep_alive=args.keepalive_timeout,
                timeout_graceful_shutdown=args.shutdown_timeout)


if __name__ == "__main__":
    main()
