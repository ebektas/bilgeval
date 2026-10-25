"""Async generation keeps synchronous protocol and request-accounting contracts."""
import asyncio
import importlib
import json
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bilgeval.batching import BatchRequestError
from bilgeval.evaluators import OpenAICompatibleTextGenerationClient
from bilgeval.registry_bundle import PromptRendererSpec


def test_async_client_is_available():
    try:
        module = importlib.import_module('bilgeval.async_generation')
    except ModuleNotFoundError:
        module = None
    assert module is not None, 'bounded async text generation is not implemented'
    assert callable(module.AsyncOpenAICompatibleTextGenerationClient)


@pytest.fixture
def async_dependencies():
    pytest.importorskip('httpx_aiohttp')
    pytest.importorskip('aiohttp')


@pytest.fixture
def server(async_dependencies):
    state = {'requests': [], 'active': 0, 'peak': 0, 'finished': [], 'counts': Counter()}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            prompt = body.get('prompt', [None])[0] if 'prompt' in body else body['messages'][-1]['content']
            with lock:
                state['requests'].append((self.path, dict(self.headers), body))
                state['active'] += 1
                state['peak'] = max(state['peak'], state['active'])
                state['counts'][prompt] += 1
                attempt = state['counts'][prompt]
            try:
                time.sleep(.3 if prompt == 'cancel' else .04 if prompt == 'slow' else .005)
                response = {'id': 'response', 'object': 'chat.completion', 'created': 0, 'model': 'fixture'}
                if 'prompt' in body:
                    indices = [0, 0] if prompt == 'duplicate' else list(reversed(range(len(body['prompt']))))
                    response['choices'] = [dict(index=i, text=body['prompt'][i], finish_reason='stop') for i in indices]
                else:
                    reasoning = prompt == 'reasoning' or (prompt == 'retry' and attempt < 3)
                    response['choices'] = [dict(index=0, message=dict(role='assistant', content='' if reasoning else prompt, reasoning_content='thought' if reasoning else None), finish_reason='length' if reasoning else 'stop')]
                status = 400 if prompt == 'fail' else 200
                if body.get('stream'):
                    chunks = [dict(id='response', object='chat.completion.chunk', created=0, model='fixture', choices=[dict(index=0, delta={'content': 'hello', 'reasoning_content': 'thought'}, finish_reason=None)])]
                    if prompt != 'partial':
                        chunks += [dict(id='response', object='chat.completion.chunk', created=0, model='fixture', choices=[dict(index=0, delta={}, finish_reason='stop')], usage={'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5})]
                    payload = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks).encode() + b'data: [DONE]\n\n'
                    content_type = 'text/event-stream'
                else:
                    payload = json.dumps(response).encode()
                    content_type = 'application/json'
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                try:
                    if prompt == 'idle' and body.get('stream'):
                        first, separator, rest = payload.partition(b'\n\n')
                        self.wfile.write(first + separator)
                        self.wfile.flush()
                        time.sleep(.1)
                        self.wfile.write(rest)
                    else:
                        self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            finally:
                with lock:
                    state['active'] -= 1
                    state['finished'].append(prompt)

    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_port}/v1', state
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def client(base, **kwargs):
    from bilgeval.async_generation import AsyncOpenAICompatibleTextGenerationClient
    return AsyncOpenAICompatibleTextGenerationClient(base_url=base, model='fixture', **kwargs)


def test_chat_order_concurrency_tally_and_repeat_calls(server):
    base, state = server
    model = client(base, generation_api='chat', max_concurrency=2)
    prompts = ['slow', 'fast', 'third', 'fourth']
    results = model.generate_texts(prompts)
    assert [r.text for r in results] == prompts
    assert state['finished'].index('fast') < state['finished'].index('slow')
    assert state['peak'] == 2
    assert model.request_tally.snapshot() == (4, 0)
    assert model.generate_texts(['again'])[0].text == 'again'
    assert model.request_tally.snapshot() == (5, 0)


@pytest.mark.parametrize('api', ['chat', 'completions'])
def test_sync_async_wire_and_result_parity(server, api):
    base, state = server
    settings = dict(base_url=base, model='fixture', generation_api=api, add_bos_token=True,
                    prompt_renderer=PromptRendererSpec(mode='template', template_id='fixture', template='Q:{prompt}'),
                    default_headers={'x-role': 'candidate'}, default_extra_body={'top_k': 12, 'seed': 7})
    sync = OpenAICompatibleTextGenerationClient(**settings)
    from bilgeval.async_generation import AsyncOpenAICompatibleTextGenerationClient
    async_model = AsyncOpenAICompatibleTextGenerationClient(**settings)
    try:
        first = sync.generate_texts(['hi'], max_tokens=19, stop=['END'], temperature=.3, top_p=.8, top_k=4)
        second = async_model.generate_texts(['hi'], max_tokens=19, stop=['END'], temperature=.3, top_p=.8, top_k=4)
    finally:
        sync.client.close()
    assert [r.text for r in first] == [r.text for r in second] == [('Q:hi' if api == 'chat' else '<bos>Q:hi')]
    assert state['requests'][0][2] == state['requests'][1][2]
    assert state['requests'][1][2]['top_k'] == 4
    assert state['requests'][1][1]['x-role'] == 'candidate'


def test_generation_retries_errors_but_conversations_do_not(server):
    base, state = server
    model = client(base, generation_api='chat')
    assert model.generate_texts(['retry'])[0].text == 'retry'
    assert state['counts']['retry'] == 3
    assert model.request_tally.snapshot() == (1, 0)
    with pytest.raises(BatchRequestError):
        model.generate_conversations([[{'role': 'user', 'content': 'reasoning'}]])
    assert state['counts']['reasoning'] == 1
    assert model.request_tally.snapshot() == (2, 0)


def test_reasoning_disabled_fallback_and_partial_failure_tally(server):
    base, _ = server
    model = client(base, generation_api='chat', request_role='judge', default_extra_body={'chat_template_kwargs': {'thinking': False}})
    results = model.generate_texts(['reasoning', 'fail'])
    assert results[0].text == 'thought'
    assert results[0].reasoning_chars is None
    assert results[1].request_failed
    assert model.request_tally.snapshot() == (2, 1)


def test_completion_permutation_and_sibling_accounting(server):
    base, state = server
    model = client(base, completion_concurrency=2, completion_batch_size=2)
    assert [r.text for r in model.generate_texts(['a', 'b', 'c', 'd'])] == ['a', 'b', 'c', 'd']
    with pytest.raises(BatchRequestError, match='indices'):
        model.generate_texts(['duplicate', 'bad', 'slow', 'good'])
    assert model.request_tally.snapshot() == (8, 2)
    assert state['counts']['slow'] == 1


def test_stream_accumulates_and_retries_partial_stream(server, monkeypatch):
    monkeypatch.setenv('BILGEVAL_BATCH_RETRY_DELAYS', '')
    base, state = server
    model = client(base, generation_api='chat', stream=True, request_role='judge')
    result = model.generate_texts(['normal'])[0]
    assert result.text == 'hello' and result.reasoning_chars == 7
    assert result.finish_reason == 'stop'
    with pytest.raises(BatchRequestError, match='without a finish_reason'):
        model.generate_texts(['partial'])
    assert state['counts']['partial'] == 3
    assert model.request_tally.snapshot() == (2, 1)


def test_synchronous_call_inside_running_loop(server):
    base, _ = server
    model = client(base, generation_api='chat')
    before = {t.ident for t in threading.enumerate() if not t.daemon}
    async def caller():
        return model.generate_texts(['loop'])[0].text
    assert asyncio.run(caller()) == 'loop'
    assert {t.ident for t in threading.enumerate() if not t.daemon} == before


def test_missing_optional_dependencies_actionable(monkeypatch):
    import builtins
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == 'httpx_aiohttp':
            raise ImportError('dependency absent')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    with pytest.raises(ImportError, match=r'bilgeval\[async\]'):
        client('http://unused/v1')


def test_stream_rejects_multiple_choices_even_with_duplicate_zero_indices():
    from types import SimpleNamespace
    from bilgeval.evaluators import ChatStreamAccumulator
    accumulator = ChatStreamAccumulator()
    choice = SimpleNamespace(index=0, delta=SimpleNamespace(content='text'), finish_reason='stop')
    with pytest.raises(ValueError, match='more than one choice'):
        accumulator.add(SimpleNamespace(choices=[choice, choice]))


def test_preheader_deadline_bounds_blocked_upload_and_closes_transport(async_dependencies):
    import httpx
    from bilgeval.async_generation import _transport_type
    closed = []
    class Upload(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'first'
            await asyncio.sleep(10)
    class ConsumeUpload(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            async for _ in request.stream:
                pass
            raise AssertionError('blocked upload should have timed out')
        async def aclose(self):
            closed.append(True)
    async def run():
        async with _transport_type()(transport=ConsumeUpload()) as transport:
            transport.header_timeout = .025
            with pytest.raises(httpx.ReadTimeout, match='before response headers'):
                await asyncio.wait_for(transport.post('http://fixture/v1', content=Upload()), .5)
    asyncio.run(run())
    assert closed == [True]


def test_preheader_deadline_does_not_bound_body_after_headers(async_dependencies):
    import httpx
    from bilgeval.async_generation import _transport_type
    closed = []
    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(.06)
            yield b'answer'
        async def aclose(self):
            closed.append(True)
    async def handler(request):
        return httpx.Response(200, stream=SlowBody())
    async def run():
        async with _transport_type()(transport=httpx.MockTransport(handler)) as transport:
            transport.header_timeout = .01
            response = await transport.get('http://fixture/v1')
            assert response.content == b'answer'
    asyncio.run(run())
    assert closed == [True]


def test_sdk_preserves_retry_budget_for_header_deadlines(async_dependencies):
    import httpx
    import openai
    from bilgeval.async_generation import _transport_type
    calls = []
    async def handler(request):
        calls.append(1)
        await asyncio.sleep(10)
        raise AssertionError('unreachable')
    async def run():
        async with _transport_type()(transport=httpx.MockTransport(handler)) as transport:
            transport.header_timeout = .01
            async with openai.AsyncOpenAI(api_key='EMPTY', http_client=transport) as sdk:
                with pytest.raises(openai.APITimeoutError):
                    await sdk.chat.completions.create(model='fixture', messages=[])
    asyncio.run(run())
    assert len(calls) == 3


def test_cancellation_drains_workers_before_session_close(server, monkeypatch):
    import aiohttp
    from bilgeval.async_generation import _bounded_map
    base, state = server
    model = client(base, generation_api='chat', max_concurrency=2)
    sessions = []
    original = aiohttp.ClientSession.close
    async def close(session):
        sessions.append(session)
        return await original(session)
    monkeypatch.setattr(aiohttp.ClientSession, 'close', close)
    cancelled = []
    async def run():
        async def batch(sdk):
            async def request(_):
                try:
                    return await model._chat(sdk, [{'role': 'user', 'content': 'cancel'}], 8, None, 0, None, None)
                finally:
                    cancelled.append(True)
            return await _bounded_map(range(20), 2, request)
        task = asyncio.create_task(model._with_client(2, batch))
        for _ in range(500):
            if state['active'] == 2:
                break
            await asyncio.sleep(.002)
        assert state['active'] == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(cancelled) == 2
        assert all(t.done() for t in asyncio.all_tasks() if t is not asyncio.current_task())
    asyncio.run(run())
    assert sessions and all(session.closed for session in sessions)


def test_trace_contains_stream_usage_and_wire_extra_body(server, tmp_path):
    import gzip
    from bilgeval.artifacts import ApiTraceCollector
    base, _ = server
    trace = ApiTraceCollector(tmp_path / 'trace.jsonl.gz')
    trace.start()
    model = client(base, generation_api='chat', stream=True, trace_collector=trace,
                   default_extra_body={'seed': 7}, request_role='judge')
    result = model.generate_texts(['normal'], top_k=4)[0]
    trace.close()
    with gzip.open(trace.path, 'rt') as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 1
    record = records[0]
    assert record['request']['seed'] == 7 and record['request']['top_k'] == 4
    assert 'extra_body' not in record['request']
    assert record['response']['usage']['total_tokens'] == 5
    assert record['response']['id'] == 'response'
    assert result.api_call['trace_id'] == record['trace_id']


def test_interrupted_loop_bridge_drains_before_reraising():
    import subprocess
    import sys
    code = '''
import asyncio, os, signal, threading
from bilgeval.async_generation import _run_sync
started = threading.Event()
cleaned = threading.Event()
async def work():
    started.set()
    try:
        await asyncio.sleep(10)
    finally:
        await asyncio.sleep(.05)
        cleaned.set()
def interrupt():
    assert started.wait(3)
    os.kill(os.getpid(), signal.SIGINT)
async def caller():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    interrupter = threading.Thread(target=interrupt, daemon=True)
    interrupter.start()
    try:
        _run_sync(work)
    except KeyboardInterrupt:
        assert cleaned.is_set(), 'bridge returned before async cleanup'
    else:
        raise AssertionError('interruption not propagated')
    interrupter.join()
    assert not any(t.name == 'bilgeval-async-generation' for t in threading.enumerate())
asyncio.run(caller())
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('indices', [[0, 0], [1], [0, 2], [True, 0], [None, 0]])
def test_completion_rejects_nonpermutations(indices):
    from types import SimpleNamespace
    from bilgeval.async_generation import AsyncOpenAICompatibleTextGenerationClient
    response = SimpleNamespace(choices=[SimpleNamespace(index=i, text='bad') for i in indices])
    with pytest.raises(BatchRequestError, match='indices'):
        AsyncOpenAICompatibleTextGenerationClient._completion_results(response, ['a', 'b'], None, 'generation-1')


def test_bounded_workers_do_not_schedule_entire_input():
    from bilgeval.async_generation import _bounded_map
    peak = 0
    async def run():
        async def work(value):
            nonlocal peak
            peak = max(peak, len(asyncio.all_tasks()))
            await asyncio.sleep(.001)
            return value
        assert await _bounded_map(list(range(1000)), 7, work) == list(range(1000))
    asyncio.run(run())
    assert peak == 8


def test_all_failed_retry_passes_record_only_final_outcomes(server, monkeypatch):
    monkeypatch.setenv('BILGEVAL_BATCH_RETRY_DELAYS', '0')
    base, state = server
    model = client(base, generation_api='chat')
    with pytest.raises(BatchRequestError):
        model.generate_texts(['fail'])
    assert state['counts']['fail'] == 6
    assert model.request_tally.snapshot() == (1, 1)


def test_conversations_preserve_messages_without_prompt_rendering(server):
    base, state = server
    model = client(base, generation_api='chat', add_bos_token=True,
                   prompt_renderer=PromptRendererSpec(mode='template', template_id='fixture', template='Q:{prompt}'))
    messages = [{'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'reply'},
                {'role': 'user', 'content': 'next'}]
    assert model.generate_conversations([messages])[0].text == 'next'
    assert state['requests'][0][2]['messages'] == messages


def test_single_chat_does_not_increment_batch_tally(server):
    base, _ = server
    model = client(base, generation_api='chat')
    result = model.generate_chat_text('one', 8, None, 0, None, None)
    assert result.text == 'one'
    assert model.request_tally.snapshot() == (0, 0)


def test_stream_idle_timeouts_retry_without_leaking_sessions(server, monkeypatch):
    import aiohttp
    monkeypatch.setenv('BILGEVAL_BATCH_RETRY_DELAYS', '')
    base, state = server
    sessions = []
    original = aiohttp.ClientSession.close
    async def close(session):
        sessions.append(session)
        return await original(session)
    monkeypatch.setattr(aiohttp.ClientSession, 'close', close)
    model = client(base, generation_api='chat', stream=True, request_role='judge', request_timeout=.04)
    with pytest.raises(BatchRequestError):
        model.generate_texts(['idle'])
    assert state['counts']['idle'] == 3
    assert sessions and all(session.closed for session in sessions)
    assert model.request_tally.snapshot() == (1, 1)
