"""HTTP contract tests for the optional replica proxy (no model/GPU required)."""
from __future__ import annotations

import asyncio
from contextlib import ExitStack
import gzip
from contextlib import contextmanager
import socket
import socketserver
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")
aiohttp = pytest.importorskip("aiohttp")

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
import uvicorn


@contextmanager
def serve_proxy_test_app(app, *, shutdown_timeout=1, **server_options):
    """Run an actual HTTP server, including ASGI lifespan and disconnects."""
    sock = socket.socket()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False,
                                         timeout_graceful_shutdown=shutdown_timeout, **server_options))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.005)
    if not server.started:
        server.should_exit = True
        thread.join(5)
        sock.close()
        raise RuntimeError("Test HTTP server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(5)
        if thread.is_alive():
            server.force_exit = True
            thread.join(2)
        sock.close()
        assert not thread.is_alive(), "Test HTTP server did not stop"


@contextmanager
def replica(name="one", *, model="smoke-model", handler=None, models_status=200, required_key=None):
    async def dispatch(request: Request):
        if required_key and request.headers.get("authorization") != "Bearer " + required_key:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if request.url.path.endswith("/models"):
            return JSONResponse({"object": "list", "data": [{"id": model, "object": "model"}]}, status_code=models_status)
        if handler is not None:
            return await handler(request)
        body = await request.body()
        return JSONResponse({
            "replica": name,
            "path": request.url.path,
            "query": request.scope["query_string"].decode(),
            "body": body.decode(),
            "headers": dict(request.headers),
        })

    app = Starlette(routes=[Route("/{path:path}", dispatch, methods=["GET", "POST"])])
    with serve_proxy_test_app(app) as url:
        yield url + "/v1"


def proxy_app(endpoints, **kwargs):
    from bilgeval.proxy import ProxyConfig, create_app
    return create_app(ProxyConfig(endpoints=tuple(endpoints), model="smoke-model", **kwargs))


async def wait_for_stat(client, predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        data = (await client.get("/stats")).json()
        if predicate(data):
            return data
        await asyncio.sleep(0.01)
    pytest.fail(f"Proxy stats did not settle: {data}")


def test_endpoints_file_normalizes_comments_origins_and_duplicates(tmp_path):
    from bilgeval.proxy import read_endpoints
    path = tmp_path / "endpoints.txt"
    path.write_text("# replicas\n\n http://LOCALHOST:8080/v1/ \nhttps://localhost:8081\n", encoding="utf-8")
    assert read_endpoints(path) == ("http://localhost:8080/v1", "https://localhost:8081/v1")
    path.write_text("http://localhost:8080/v1\nhttp://LOCALHOST:8080/v1/\n")
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        read_endpoints(path)
    path.write_text("# nothing here\n")
    with pytest.raises(ValueError, match="[Ee]ndpoint"):
        read_endpoints(path)


@pytest.mark.parametrize("error_type,status", [(aiohttp.ClientConnectionError, 502), (TimeoutError, 504)])
def test_transport_failure_logs_only_type_and_configured_endpoint(caplog, error_type, status):
    secret = "private-header-token-and-request-body"
    endpoint = "http://replica.invalid/v1"

    async def exercise():
        async def upstream(*args, **kwargs):
            raise error_type(secret)

        app = proxy_app([endpoint], retries=0)
        app.state.clients = {endpoint: SimpleNamespace(request=upstream)}
        sent_body = False

        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": secret.encode(), "more_body": False}
            await asyncio.Future()

        request = Request({"type": "http", "method": "POST", "path": "/v1/completions",
                           "raw_path": b"/v1/completions", "query_string": b"token=private-query",
                           "headers": [(b"authorization", secret.encode())], "scheme": "http",
                           "server": ("proxy", 80), "app": app}, receive)
        forward = next(route.endpoint for route in app.routes if route.path == "/v1/completions")
        response = await forward(request)
        assert response.status_code == status

    with caplog.at_level("WARNING", logger="bilgeval.proxy"):
        asyncio.run(exercise())
    records = [record for record in caplog.records if record.name == "bilgeval.proxy"]
    assert len(records) == 1
    assert error_type.__name__ in records[0].getMessage()
    assert endpoint in records[0].getMessage()
    assert secret not in caplog.text
    assert "private-query" not in caplog.text
    assert records[0].exc_info is None


@pytest.mark.parametrize("url", ["localhost:8080", "ftp://localhost/v1", "http://", "http://localhost:bad/v1", "http://u:secret@localhost/v1", "http://localhost/v1?q=secret", "http://localhost/v1#fragment", "http://local host/v1"])
def test_invalid_endpoint_is_rejected_without_echoing_secrets(url):
    from bilgeval.proxy import ProxyConfig
    with pytest.raises(ValueError) as caught:
        ProxyConfig(endpoints=(url,), model="m")
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("kwargs", [{"capacity": 0}, {"capacity": 1.5}, {"timeout": 0}, {"timeout": float("inf")}, {"queue_timeout": float("nan")}, {"queue_timeout": -1}, {"retries": -1}, {"retries": 1.5}, {"cooldown": 0}, {"model": ""}])
def test_invalid_settings_fail_before_startup(kwargs):
    from bilgeval.proxy import ProxyConfig
    options = {"endpoints": ("http://localhost:1234/v1",), "model": "m"}
    options.update(kwargs)
    with pytest.raises(ValueError):
        ProxyConfig(**options)


@pytest.mark.parametrize("mode", ["mismatch", "unreachable", "bad_status", "redirect"])
def test_startup_rejects_invalid_model_endpoints(mode):
    async def check(endpoint):
        app = proxy_app([endpoint], timeout=0.1)
        with pytest.raises(ValueError, match="[Mm]odel|[Ee]ndpoint"):
            async with app.router.lifespan_context(app):
                pytest.fail("Invalid upstream should prevent startup")
    if mode == "unreachable":
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            endpoint = f"http://127.0.0.1:{sock.getsockname()[1]}/v1"
        asyncio.run(check(endpoint))
    else:
        with replica(model="wrong" if mode == "mismatch" else "smoke-model",
                     models_status={"bad_status": 503, "redirect": 307}.get(mode, 200)) as endpoint:
            asyncio.run(check(endpoint))


def test_round_robin_ties_and_transparent_tools_native_paths_headers():
    with replica("one") as first, replica("two") as second:
        with serve_proxy_test_app(proxy_app([first, second])) as proxy:
            with httpx.Client(base_url=proxy) as client:
                body = b'{"model":"smoke-model","messages":[],"tools":[{"type":"function","function":{"name":"tool"}}]}'
                response = client.post("/v1/chat/completions?request_id=a%2Fb", content=body, headers={"authorization": "Bearer test-key", "connection": "x-remove, keep-alive", "x-remove": "secret", "x-end-to-end": "preserved", "content-type": "application/json"})
                data = response.json()
                assert data["replica"] == "one"
                assert data["body"] == body.decode()
                assert data["query"] == "request_id=a%2Fb"
                assert data["headers"]["authorization"] == "Bearer test-key"
                assert data["headers"]["x-end-to-end"] == "preserved"
                assert "x-remove" not in data["headers"]
                assert data["headers"]["host"] == httpx.URL(first).netloc.decode()
                assert client.post("/v1/completions", json={"prompt": "x"}).json()["replica"] == "two"
                assert client.post("/tokenize", json={"prompt": "x"}).json()["path"] == "/tokenize"
                assert client.get("/get_model_info").json()["path"] == "/get_model_info"
                assert client.post("/generate", json={"text": "x"}).json()["path"] == "/generate"
                assert client.get("/v1/models").json()["data"][0]["id"] == "smoke-model"
                assert client.post("/admin/delete", json={}).status_code == 404
                assert client.get("/v1/completions").status_code == 405
                stats = client.get("/stats").json()
                assert sum(item["requests"] for item in stats["endpoints"]) == 6
                assert all(item["active"] == 0 for item in stats["endpoints"])
                assert client.get("/health").json()["status"] == "ok"


def test_upstream_nonretryable_status_and_headers_are_preserved():
    async def handler(request):
        return Response(b'{"error":"invalid request"}', status_code=422, headers={"content-type": "application/json", "x-request-id": "upstream-id", "connection": "x-remove", "x-remove": "private"})
    with replica(handler=handler) as endpoint:
        with serve_proxy_test_app(proxy_app([endpoint])) as proxy:
            response = httpx.post(proxy + "/v1/completions", json={})
            assert response.status_code == 422
            assert response.content == b'{"error":"invalid request"}'
            assert response.headers["x-request-id"] == "upstream-id"
            assert "x-remove" not in response.headers


def test_capacity_is_bounded_and_queued_request_times_out():
    entered = threading.Event()
    release = threading.Event()
    async def handler(request):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return JSONResponse({"done": True})
    async def exercise(proxy):
        async with httpx.AsyncClient(base_url=proxy, timeout=3) as client:
            first = asyncio.create_task(client.post("/v1/completions", json={}))
            try:
                await asyncio.to_thread(entered.wait, 2)
                assert entered.is_set()
                started = time.monotonic()
                second = await client.post("/v1/completions", json={})
                assert second.status_code == 503
                assert time.monotonic() - started >= 0.04
                stats = (await client.get("/stats")).json()["endpoints"][0]
                assert stats["active"] == 1
                assert stats["requests"] == 1
            finally:
                release.set()
                assert (await first).status_code == 200
            await wait_for_stat(client, lambda s: s["endpoints"][0]["active"] == 0)
    with replica(handler=handler) as endpoint:
        with serve_proxy_test_app(proxy_app([endpoint], capacity=1, queue_timeout=0.05)) as proxy:
            asyncio.run(exercise(proxy))


def test_least_active_routes_around_busy_replica():
    entered = threading.Event()
    release = threading.Event()
    async def handler(request):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return JSONResponse({"replica": "slow"})
    async def exercise(proxy):
        async with httpx.AsyncClient(base_url=proxy, timeout=3) as client:
            first = asyncio.create_task(client.post("/v1/completions", json={}))
            try:
                await asyncio.to_thread(entered.wait, 2)
                assert entered.is_set()
                for _ in range(3):
                    assert (await client.post("/v1/completions", json={})).json()["replica"] == "fast"
            finally:
                release.set()
                await first
    with replica(handler=handler) as first, replica("fast") as second:
        with serve_proxy_test_app(proxy_app([first, second], capacity=2)) as proxy:
            asyncio.run(exercise(proxy))


@pytest.mark.parametrize("failure", ["status", "timeout"])
def test_transient_failure_retries_other_replica_and_cools_down(failure):
    async def handler(request):
        if failure == "timeout":
            await asyncio.sleep(0.15)
        return JSONResponse({"error": "busy"}, status_code=503)
    with replica(handler=handler) as first, replica("working") as second:
        with serve_proxy_test_app(proxy_app([first, second], timeout=0.04, cooldown=10)) as proxy:
            with httpx.Client(base_url=proxy) as client:
                assert client.post("/v1/completions", json={}).json()["replica"] == "working"
                assert client.post("/v1/completions", json={}).json()["replica"] == "working"
                stats = client.get("/stats").json()["endpoints"]
                assert stats[0]["requests"] == 1
                assert stats[0]["failures"] == 1
                assert stats[0]["active"] == 0
                assert stats[0]["healthy"] is False
                assert stats[1]["requests"] == 2


def test_all_down_is_bounded_and_final_upstream_status_is_preserved():
    async def handler(request):
        return JSONResponse({"error": "busy"}, status_code=429, headers={"retry-after": "1"})
    with replica(handler=handler) as first, replica(handler=handler) as second:
        with serve_proxy_test_app(proxy_app([first, second], queue_timeout=0.05, cooldown=10)) as proxy:
            with httpx.Client(base_url=proxy) as client:
                response = client.post("/v1/completions", json={})
                assert response.status_code == 429
                assert response.json() == {"error": "busy"}
                assert response.headers["retry-after"] == "1"
                started = time.monotonic()
                assert client.post("/v1/completions", json={}).status_code == 503
                assert time.monotonic() - started < 1
                health = client.get("/health")
                assert health.status_code == 503
                assert health.json()["status"] == "unavailable"
                stats = client.get("/stats").json()["endpoints"]
                assert [item["requests"] for item in stats] == [1, 1]
                assert [item["active"] for item in stats] == [0, 0]


def test_stream_is_incremental_and_disconnect_releases_upstream_slot():
    upstream_closed = threading.Event()
    async def handler(request):
        async def chunks():
            try:
                yield b'data: {"first":true}\n\n'
                while True:
                    await asyncio.sleep(0.05)
                    yield b": keepalive\n\n"
            finally:
                upstream_closed.set()
        return StreamingResponse(chunks(), media_type="text/event-stream")
    async def exercise(proxy):
        async with httpx.AsyncClient(base_url=proxy, timeout=2) as client:
            async with client.stream("POST", "/v1/chat/completions", json={"stream": True}) as response:
                assert response.status_code == 200
                chunks = response.aiter_raw()
                assert await anext(chunks) == b'data: {"first":true}\n\n'
                assert (await client.get("/stats")).json()["endpoints"][0]["active"] == 1
            await wait_for_stat(client, lambda s: s["endpoints"][0]["active"] == 0)
            assert await asyncio.to_thread(upstream_closed.wait, 2)
    with replica(handler=handler) as endpoint:
        with serve_proxy_test_app(proxy_app([endpoint], capacity=1)) as proxy:
            asyncio.run(exercise(proxy))


def test_cancelled_queued_request_does_not_consume_capacity():
    entered = threading.Event()
    release = threading.Event()
    async def handler(request):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return JSONResponse({"ok": True})
    async def exercise(proxy):
        async with httpx.AsyncClient(base_url=proxy, timeout=3) as client:
            first = asyncio.create_task(client.post("/v1/completions", json={}))
            try:
                await asyncio.to_thread(entered.wait, 2)
                queued = asyncio.create_task(client.post("/v1/completions", json={}))
                await asyncio.sleep(0.05)
                queued.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await queued
            finally:
                release.set()
                await first
            await asyncio.sleep(0.1)
            stats = (await client.get("/stats")).json()["endpoints"][0]
            assert stats["active"] == 0
            assert stats["requests"] == 1
    with replica(handler=handler) as endpoint:
        with serve_proxy_test_app(proxy_app([endpoint], capacity=1, queue_timeout=0.5)) as proxy:
            asyncio.run(exercise(proxy))


def test_models_root_alias_preserves_configured_api_prefix():
    with replica() as endpoint:
        prefix = endpoint.removesuffix("/v1") + "/service/v1"
        with serve_proxy_test_app(proxy_app([prefix])) as proxy:
            with httpx.Client(base_url=proxy) as client:
                assert client.get("/models").json()["data"][0]["id"] == "smoke-model"
                assert client.post("/v1/completions", json={}).json()["path"] == "/service/v1/completions"
                assert client.post("/tokenize", json={}).json()["path"] == "/service/tokenize"


def test_model_discovery_and_forwarding_use_configured_api_key(monkeypatch):
    monkeypatch.setenv("BILGEVAL_API_KEY", "private-test-key")
    with replica(required_key="private-test-key") as endpoint:
        from bilgeval.proxy import ProxyConfig, create_app
        config = ProxyConfig(endpoints=(endpoint,), model="smoke-model")
        assert "private-test-key" not in repr(config)
        with serve_proxy_test_app(create_app(config)) as proxy:
            with httpx.Client(base_url=proxy) as client:
                assert client.post("/v1/completions", json={}).status_code == 200
                assert "private-test-key" not in client.get("/stats").text
                assert client.post("/v1/completions", json={}, headers={"authorization": "Bearer override"}).status_code == 401


def test_transport_failure_after_startup_releases_slot():
    with ExitStack() as upstreams:
        endpoint = upstreams.enter_context(replica())
        with serve_proxy_test_app(proxy_app([endpoint], retries=0)) as proxy:
            upstreams.close()
            with httpx.Client(base_url=proxy) as client:
                response = client.post("/v1/completions", json={"prompt": "sensitive-prompt"})
                assert response.status_code == 502
                assert "sensitive-prompt" not in response.text
                assert "http://" not in response.text
                stats = client.get("/stats").json()["endpoints"][0]
                assert stats["active"] == 0
                assert stats["failures"] == 1


def test_cooldown_expires_and_replica_can_recover():
    attempts = 0
    async def handler(request):
        nonlocal attempts
        attempts += 1
        return JSONResponse({"attempt": attempts}, status_code=503 if attempts == 1 else 200)
    with replica(handler=handler) as endpoint:
        with serve_proxy_test_app(proxy_app([endpoint], retries=0, cooldown=0.05)) as proxy:
            with httpx.Client(base_url=proxy) as client:
                assert client.post("/v1/completions", json={}).status_code == 503
                assert client.get("/health").status_code == 503
                time.sleep(0.06)
                assert client.post("/v1/completions", json={}).json() == {"attempt": 2}
                assert client.get("/health").status_code == 200


def test_compressed_response_bytes_are_not_transformed_twice():
    async def handler(request):
        return Response(gzip.compress(b'{"payload":"compressed"}'), headers={"content-encoding": "gzip", "content-type": "application/json"})
    with replica(handler=handler) as endpoint:
        with serve_proxy_test_app(proxy_app([endpoint])) as proxy:
            response = httpx.post(proxy + "/v1/completions", json={})
            assert response.status_code == 200
            assert response.json() == {"payload": "compressed"}
            assert response.headers["content-encoding"] == "gzip"


@pytest.mark.parametrize("failure", ["disconnect", "idle_timeout"])
def test_failed_stream_is_not_replayed_after_bytes_were_delivered(failure):
    async def handler(request):
        async def chunks():
            yield b"data: first\n\n"
            await asyncio.sleep(0.2 if failure == "idle_timeout" else 0.02)
            raise RuntimeError("Intentional upstream disconnection")
        return StreamingResponse(chunks(), media_type="text/event-stream")
    async def exercise(proxy):
        async with httpx.AsyncClient(base_url=proxy, timeout=2) as client:
            async with client.stream("POST", "/v1/chat/completions", json={"stream": True}) as response:
                chunks = response.aiter_raw()
                assert await anext(chunks) == b"data: first\n\n"
                with pytest.raises(httpx.RemoteProtocolError):
                    await anext(chunks)
            stats = await wait_for_stat(client, lambda s: all(e["active"] == 0 for e in s["endpoints"]))
            assert stats["endpoints"][0]["failures"] == 1
            assert stats["endpoints"][1]["requests"] == 0
    with replica(handler=handler) as first, replica("other") as second:
        with serve_proxy_test_app(proxy_app([first, second], retries=3, timeout=0.1)) as proxy:
            asyncio.run(exercise(proxy))


def test_disconnected_request_waiting_for_upstream_headers_releases_slot():
    entered = threading.Event()
    release = threading.Event()
    async def handler(request):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return JSONResponse({"ok": True})
    async def exercise(proxy):
        async with httpx.AsyncClient(base_url=proxy, timeout=3) as client:
            request = asyncio.create_task(client.post("/v1/completions", json={}))
            try:
                await asyncio.to_thread(entered.wait, 2)
                assert entered.is_set()
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                await wait_for_stat(client, lambda s: s["endpoints"][0]["active"] == 0)
            finally:
                release.set()
    with replica(handler=handler) as endpoint:
        with serve_proxy_test_app(proxy_app([endpoint], capacity=1)) as proxy:
            asyncio.run(exercise(proxy))


def test_handler_cancellation_as_headers_arrive_closes_unclaimed_response():
    """Cancellation after prepare completes must still close its response/slot."""
    import json
    async def handler(request):
        async def chunks():
            yield b"not consumed"
            await asyncio.sleep(1)
        return StreamingResponse(chunks())
    async def exercise(endpoint):
        app = proxy_app([endpoint], capacity=1)
        async with app.router.lifespan_context(app):
            client = app.state.clients[endpoint]
            captured = []
            async def upstream(*args, **kwargs):
                response = await client.request(*args, **kwargs)
                captured.append(response)
                asyncio.get_running_loop().call_soon(outer.cancel)
                return response
            app.state.clients[endpoint] = SimpleNamespace(request=upstream)
            sent_body = False
            async def receive():
                nonlocal sent_body
                if not sent_body:
                    sent_body = True
                    return {"type": "http.request", "body": b"{}", "more_body": False}
                await asyncio.Future()
            request = Request({"type": "http", "method": "POST", "path": "/v1/completions", "raw_path": b"/v1/completions", "query_string": b"", "headers": [], "scheme": "http", "server": ("proxy", 80), "app": app}, receive)
            forward = next(route.endpoint for route in app.routes if route.path == "/v1/completions")
            outer = asyncio.create_task(forward(request))
            with pytest.raises(asyncio.CancelledError):
                await outer
            stats_endpoint = next(route.endpoint for route in app.routes if route.path == "/stats")
            response = await stats_endpoint(request)
            assert json.loads(response.body)["endpoints"][0]["active"] == 0
            assert captured[0].closed
    with replica(handler=handler) as endpoint:
        asyncio.run(exercise(endpoint))


def test_cli_applies_finite_graceful_shutdown_timeout(tmp_path, monkeypatch):
    from bilgeval.proxy import main
    endpoints = tmp_path / "replicas.txt"
    endpoints.write_text("http://localhost:8000/v1\n")
    options = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: options.update(kwargs))
    main(["--endpoints-file", str(endpoints), "--model", "m", "--shutdown-timeout", "0.05"])
    assert options["timeout_graceful_shutdown"] == 0.05
    assert options["access_log"] is False
    options.clear()
    main(["--endpoints-file", str(endpoints), "--model", "m"])
    assert options["timeout_graceful_shutdown"] == 30


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_shutdown_timeout(tmp_path, value):
    from bilgeval.proxy import main
    endpoints = tmp_path / "replicas.txt"
    endpoints.write_text("http://localhost:8000/v1\n")
    with pytest.raises(SystemExit) as caught:
        main(["--endpoints-file", str(endpoints), "--model", "m", "--shutdown-timeout", value])
    assert caught.value.code == 2


@pytest.mark.parametrize("value, expected", [(None, 60), ("120", 120), ("0", 0)])
def test_cli_configures_idle_connection_timeout(tmp_path, monkeypatch, value, expected):
    from bilgeval.proxy import main
    endpoints = tmp_path / "replicas.txt"
    endpoints.write_text("http://localhost:8000/v1\n")
    options = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: options.update(kwargs))
    args = ["--endpoints-file", str(endpoints), "--model", "m"]
    if value is not None:
        args += ["--keepalive-timeout", value]
    main(args)
    assert options["timeout_keep_alive"] == expected


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_keepalive_timeout(tmp_path, monkeypatch, capsys, value):
    from bilgeval.proxy import main
    endpoints = tmp_path / "replicas.txt"
    endpoints.write_text("http://localhost:8000/v1\n")
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: pytest.fail("Invalid timeout started server"))
    with pytest.raises(SystemExit) as caught:
        main(["--endpoints-file", str(endpoints), "--model", "m", f"--keepalive-timeout={value}"])
    assert caught.value.code == 2
    assert "Keep-alive timeout must be nonnegative and finite" in capsys.readouterr().err


def test_shutdown_cancels_endless_stream_and_closes_resources():
    import json
    upstream_closed = threading.Event()
    async def handler(request):
        async def chunks():
            try:
                while True:
                    yield b"data: heartbeat\n\n"
                    await asyncio.sleep(0.01)
            finally:
                upstream_closed.set()
        return StreamingResponse(chunks(), media_type="text/event-stream")
    with replica(handler=handler) as endpoint:
        app = proxy_app([endpoint], capacity=1)
        with ExitStack() as servers:
            proxy = servers.enter_context(serve_proxy_test_app(app, shutdown_timeout=0.05))
            async def exercise():
                async with httpx.AsyncClient(base_url=proxy, timeout=2) as client:
                    async with client.stream("POST", "/v1/chat/completions", json={"stream": True}) as response:
                        chunks = response.aiter_raw()
                        first = await anext(chunks)
                        assert first and first.replace(b"data: heartbeat\n\n", b"") == b""
                        started = time.monotonic()
                        await asyncio.to_thread(servers.close)
                        assert time.monotonic() - started < 2
                        assert await asyncio.to_thread(upstream_closed.wait, 2)
                        stats_endpoint = next(route.endpoint for route in app.routes if route.path == "/stats")
                        stats = json.loads((await stats_endpoint(None)).body)
                        assert stats["endpoints"][0]["active"] == 0
                        assert all(session.closed for session in app.state.clients.values())
            asyncio.run(exercise())


@contextmanager
def raw_replica(handle_generation):
    """A real peer whose upload/header timing is independent of ASGI buffering."""
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(3)
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = self.request.recv(65536)
                if not chunk:
                    return
                data += chunk
            try:
                if data.startswith(b"GET /v1/models "):
                    body = b'{"data":[{"id":"smoke-model"}]}'
                    self.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: " +
                                         str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
                else:
                    handle_generation(self.request, data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
    with Server(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}/v1"
        finally:
            server.shutdown()
            thread.join(2)


@pytest.mark.parametrize("invalid_model", [False, True])
def test_endpoint_sessions_close_after_success_or_partial_discovery_failure(monkeypatch, invalid_model):
    """Session ownership must survive another endpoint failing discovery."""
    sessions = []
    original = aiohttp.ClientSession
    def record_session(*args, **kwargs):
        session = original(*args, **kwargs)
        sessions.append(session)
        return session
    monkeypatch.setattr(aiohttp, "ClientSession", record_session)
    with replica() as first, replica(model="wrong" if invalid_model else "smoke-model") as second:
        async def exercise():
            app = proxy_app([first, second], capacity=2)
            try:
                async with app.router.lifespan_context(app):
                    assert not invalid_model
                    assert len(sessions) == 2
                    assert sessions[0].connector is not sessions[1].connector
                    assert all(s.connector.limit == s.connector.limit_per_host == 2 for s in sessions)
            except ValueError:
                assert invalid_model
            assert len(sessions) == 2
            assert all(session.closed for session in sessions)
        asyncio.run(exercise())


def test_forwarding_preserves_raw_paths_queries_repeated_headers_and_does_not_add_defaults():
    seen = []
    async def handler(request):
        seen.append((request.scope["raw_path"], request.scope["query_string"], request.headers.raw))
        response = Response(b"redirect bytes", status_code=307)
        response.raw_headers.extend([
            (b"location", b"/never-follow"),
            (b"x-repeat", b"one"), (b"x-repeat", b"two"),
            (b"x-opaque", b"caf\xe9"),
            (b"set-cookie", b"private=one; Path=/"),
        ])
        return response
    with replica(handler=handler) as endpoint:
        with serve_proxy_test_app(proxy_app([endpoint])) as proxy:
            with httpx.Client() as client:
                for _ in range(2):
                    request = httpx.Request("POST", proxy + "/v1/%63ompletions?q=a%2Fb&q=%2f&plus=a+b&empty=",
                                            content=b"{}", headers=[(b"x-repeat", b"one"), (b"x-repeat", b"two"),
                                                                   (b"x-utf8", "café".encode())])
                    response = client.send(request)
                    assert response.status_code == 307
                    assert response.content == b"redirect bytes"
                    assert response.headers.get_list("x-repeat") == ["one", "two"]
                    assert (b"x-opaque", b"caf\xe9") in response.headers.raw
    assert len(seen) == 2
    for path, query, headers in seen:
        assert path == b"/v1/%63ompletions"
        assert query == b"q=a%2Fb&q=%2f&plus=a+b&empty="
        assert [value for name, value in headers if name == b"x-repeat"] == [b"one", b"two"]
        assert (b"x-utf8", b"caf\xc3\xa9") in headers
        assert not ({name for name, _ in headers} & {b"cookie", b"accept", b"accept-encoding", b"user-agent", b"content-type"})


def test_non_utf8_request_header_is_explicitly_rejected_without_upstream_attempt():
    with replica() as endpoint:
        with serve_proxy_test_app(proxy_app([endpoint], capacity=1)) as proxy:
            with httpx.Client(base_url=proxy) as client:
                response = client.post("/v1/completions", content=b"{}", headers=[(b"x-opaque", b"caf\xe9")])
                assert response.status_code == 400
                assert "UTF-8" in response.json()["error"]["message"]
                stats = client.get("/stats").json()["endpoints"][0]
                assert stats["active"] == stats["requests"] == stats["failures"] == 0
                assert client.post("/v1/completions", json={}).status_code == 200


def test_preheader_timeout_bounds_progressively_arriving_headers():
    """Repeated socket activity cannot extend the complete pre-header budget."""
    def trickle(sock, data):
        for chunk in (b"HTTP/1.1 200 OK\r\n", b"X-One: 1\r\n", b"X-Two: 2\r\n",
                      b"X-Three: 3\r\n", b"Content-Length: 2\r\n\r\n{}"):
            time.sleep(0.035)
            sock.sendall(chunk)
    with raw_replica(trickle) as endpoint:
        with serve_proxy_test_app(proxy_app([endpoint], timeout=0.08, retries=0)) as proxy:
            response = httpx.post(proxy + "/v1/completions", content=b"{}")
            assert response.status_code == 504
            stats = httpx.get(proxy + "/stats").json()["endpoints"][0]
            assert stats["active"] == 0
            assert stats["failures"] == 1


@pytest.mark.parametrize("cancel", [False, True])
def test_blocked_upload_timeout_or_disconnect_releases_capacity(cancel):
    entered = threading.Event()
    release = threading.Event()
    def never_read(sock, data):
        entered.set()
        release.wait(2)
    async def exercise(proxy):
        async with httpx.AsyncClient(base_url=proxy, timeout=3) as client:
            started = time.monotonic()
            task = asyncio.create_task(client.post("/v1/completions", content=b"x" * (16 * 1024 * 1024)))
            assert await asyncio.to_thread(entered.wait, 2)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                assert (await task).status_code == 504
                assert time.monotonic() - started < 1.5
            stats = await wait_for_stat(client, lambda s: s["endpoints"][0]["active"] == 0)
            assert stats["endpoints"][0]["requests"] == 1
            assert stats["endpoints"][0]["failures"] == (0 if cancel else 1)
    with raw_replica(never_read) as endpoint:
        with serve_proxy_test_app(proxy_app([endpoint], timeout=0.15, retries=0, capacity=1)) as proxy:
            try:
                asyncio.run(exercise(proxy))
            finally:
                release.set()


def test_stream_timeout_is_idle_budget_not_total_lifetime():
    async def handler(request):
        async def chunks():
            for _ in range(8):
                yield b"chunk\n"
                await asyncio.sleep(0.03)
        return StreamingResponse(chunks())
    with replica(handler=handler) as endpoint:
        with serve_proxy_test_app(proxy_app([endpoint], timeout=0.1, retries=0)) as proxy:
            response = httpx.post(proxy + "/v1/completions", content=b"{}")
            assert response.status_code == 200
            assert response.content == b"chunk\n" * 8


@pytest.mark.parametrize("raw_path, expected", [
    (b"/v1%2fcompletions", "/prefix/v1%2fcompletions"),
    (b"/%761/%63ompletions", "/prefix/v1/%63ompletions"),
    (None, "/prefix/v1/completions"),
])
def test_target_url_preserves_encoded_prefix_boundary_and_optional_raw_path(raw_path, expected):
    from bilgeval.proxy import _target_url
    assert _target_url("http://replica/prefix/v1", "/v1/completions", b"q=%2f", raw_path) == "http://replica" + expected + "?q=%2f"


def test_startup_verifies_https_certificates(tmp_path):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is required to generate the local self-signed test certificate")
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-days", "1", "-subj", "/CN=127.0.0.1",
                    "-addext", "subjectAltName=IP:127.0.0.1",
                    "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    async def models(request):
        return JSONResponse({"data": [{"id": "smoke-model"}]})
    upstream = Starlette(routes=[Route("/v1/models", models)])
    with serve_proxy_test_app(upstream, ssl_certfile=str(cert), ssl_keyfile=str(key)) as url:
        async def exercise():
            app = proxy_app([url.replace("http:", "https:") + "/v1"], timeout=1)
            with pytest.raises(ValueError, match="model discovery failed"):
                async with app.router.lifespan_context(app):
                    pytest.fail("Self-signed upstream must not pass discovery")
            assert all(session.closed for session in app.state.clients.values())
        asyncio.run(exercise())
