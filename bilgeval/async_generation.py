"""Opt-in, bounded asynchronous transport behind the synchronous text API.

Each public call owns its loop, SDK client and connection pool. Calls made from
an existing event loop block that caller through a joined helper thread; callers
needing a responsive loop can use ``await asyncio.to_thread(...)``.
"""

import asyncio
import json
import threading
from itertools import count
from typing import Any

import httpx
import openai

from .artifacts import ApiTraceCollector
from .batching import BatchRequestError, all_failed, batch_retry_delays, require_batch_success
from .evaluators import (
    AnswerVerifierModelClient, ChatStreamAccumulator, DEFAULT_REQUEST_TIMEOUT_SECONDS,
    IncompleteStream, JudgeModelClient, OpenAICompatibleTextGenerationClient,
    STREAM_RETRIES, add_bos_prefix, capture_api_call, stream_idle_timeout,
    validate_completion_settings,
)
from .registry_bundle import PromptRendererSpec
from .results import GeneratedTextResult
from .serialization import render_model_prompt


def _transport_type():
    try:
        from httpx_aiohttp import HttpxAiohttpClient
        import aiohttp
    except ImportError as exc:
        raise ImportError(
            "Async generation requires optional dependencies; install pip install 'bilgeval[async]'."
        ) from exc

    class HeaderTimeoutClient(HttpxAiohttpClient):
        """Bound each HTTP attempt through upload/headers, retaining idle reads."""

        async def send(self, request, *, stream=False, **kwargs):
            try:
                response = await asyncio.wait_for(
                    super().send(request, stream=True, **kwargs),
                    timeout=self.header_timeout,
                )
            except asyncio.TimeoutError as exc:
                raise httpx.ReadTimeout("Timed out before response headers", request=request) from exc
            if not stream:
                try:
                    await response.aread()
                except BaseException:
                    await response.aclose()
                    raise
            return response

    return HeaderTimeoutClient


def _run_sync(factory):
    """Run a coroutine factory, owning and draining its loop even on interruption."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    ready = threading.Event()
    finished = threading.Event()
    state = {}

    def run():
        async def own():
            state['loop'] = asyncio.get_running_loop()
            state['task'] = asyncio.current_task()
            ready.set()
            return await factory()
        try:
            state['result'] = asyncio.run(own())
        except BaseException as exc:
            state['error'] = exc
        finally:
            ready.set()
            finished.set()

    thread = threading.Thread(target=run, name='bilgeval-async-generation')
    try:
        thread.start()
        finished.wait()
    except BaseException:
        if thread.ident is None:
            raise
        ready.wait()
        loop, task = state.get('loop'), state.get('task')
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        finished.wait()
        thread.join()
        raise
    thread.join()
    if 'error' in state:
        raise state['error']
    return state['result']


async def _bounded_map(items, concurrency, call):
    """Only worker-count tasks; drain siblings before raising ordered errors."""
    results = [None] * len(items)
    errors = {}
    pending = iter(enumerate(items))

    async def worker():
        for index, item in pending:
            try:
                results[index] = await call(item)
            except Exception as exc:
                errors[index] = exc

    workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(items)))]
    try:
        await asyncio.gather(*workers)
    finally:
        for worker in workers:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    if errors:
        raise errors[min(errors)]
    return results


class AsyncOpenAICompatibleTextGenerationClient(OpenAICompatibleTextGenerationClient):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8181/v1",
        model: str = "",
        api_key: str = "EMPTY",
        insecure_skip_verify: bool = False,
        add_bos_token: bool = False,
        bos_token: str = "<bos>",
        generation_api: str = "completions",
        prompt_renderer: PromptRendererSpec | None = None,
        trace_collector: ApiTraceCollector | None = None,
        request_role: str = "generation",
        default_headers: dict[str, str] | None = None,
        default_extra_body: dict[str, Any] | None = None,
        max_concurrency: int = 16,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        stream: bool = False,
        completion_concurrency: int = 1,
        completion_batch_size: int | None = None,
    ):
        if generation_api not in {"completions", "chat"}:
            raise ValueError("generation_api must be 'completions' or 'chat'.")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        validate_completion_settings(completion_concurrency, completion_batch_size)
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive.")
        self._transport_class = _transport_type()
        self.base_url, self.model, self.api_key = base_url, model, api_key
        self.insecure_skip_verify = insecure_skip_verify
        self.add_bos_token, self.bos_token = add_bos_token, bos_token
        self.generation_api, self.prompt_renderer = generation_api, prompt_renderer
        self.trace_collector, self.request_role = trace_collector, request_role
        self.default_headers = dict(default_headers or {})
        self.default_extra_body = json.loads(json.dumps(default_extra_body)) if default_extra_body is not None else None
        self.max_concurrency = max_concurrency
        self.completion_concurrency = completion_concurrency
        self.completion_batch_size = completion_batch_size
        self.request_timeout, self.stream = request_timeout, stream
        self._request_ids = count(1)

    async def _with_client(self, concurrency, call):
        read_timeout = min(self.request_timeout, stream_idle_timeout()) if self.stream else self.request_timeout
        async with self._transport_class(
            verify=not self.insecure_skip_verify, trust_env=True, follow_redirects=False,
            timeout=httpx.Timeout(self.request_timeout, connect=30.0, read=read_timeout),
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        ) as transport:
            transport.header_timeout = self.request_timeout
            async with openai.AsyncOpenAI(
                base_url=self.base_url, api_key=self.api_key,
                default_headers=self.default_headers, http_client=transport,
            ) as sdk:
                return await call(sdk)

    async def _chat(self, sdk, messages, max_tokens, stop, temperature, top_p, top_k):
        request_id = f"{self.request_role}-{next(self._request_ids)}"
        request = self._chat_request(messages, max_tokens, stop, temperature, top_p, top_k)
        wire_request = self.wire_request(request)
        try:
            if self.stream:
                for attempt in range(STREAM_RETRIES + 1):
                    try:
                        stream = await sdk.chat.completions.create(**request)
                        try:
                            accumulator = ChatStreamAccumulator()
                            async for chunk in stream:
                                accumulator.add(chunk)
                            response = accumulator.finish()
                        finally:
                            await stream.close()
                        break
                    except (httpx.TimeoutException, openai.APITimeoutError, IncompleteStream):
                        if attempt == STREAM_RETRIES:
                            raise
            else:
                response = await sdk.chat.completions.create(**request)
        except Exception as exc:
            return GeneratedTextResult(
                request_failed=True, error=f"Chat completion request failed: {exc}",
                api_call=capture_api_call(self.trace_collector, role=self.request_role,
                    base_url=self.base_url, request=wire_request, error=str(exc), endpoint="chat"),
                request_id=request_id,
            )
        api_call = capture_api_call(self.trace_collector, role=self.request_role,
            base_url=self.base_url, request=wire_request, response=response, endpoint="chat")
        return self._chat_result(response, api_call, request_id)

    def generate_chat_text(self, prompt, max_tokens, stop, temperature, top_p, top_k):
        messages = [{"role": "user", "content": render_model_prompt(prompt, self.prompt_renderer)}]
        return self._generate_chat_messages(messages, max_tokens, stop, temperature, top_p, top_k)

    def _generate_chat_messages(self, messages, max_tokens, stop, temperature, top_p, top_k):
        return _run_sync(lambda: self._with_client(1, lambda sdk: self._chat(
            sdk, messages, max_tokens, stop, temperature, top_p, top_k)))

    def generate_conversations(self, conversations, max_tokens=8, stop=None,
                               temperature=0.0, top_p=None, top_k=None):
        if self.generation_api != "chat":
            raise ValueError("Multi-turn conversations require generation_api='chat'.")
        if not conversations:
            return []
        async def generate(sdk):
            results = await _bounded_map(conversations, self.max_concurrency, lambda messages:
                self._chat(sdk, messages, max_tokens, stop, temperature, top_p, top_k))
            self.request_tally.record(results)
            return require_batch_success(results, "Chat conversation generation")
        return _run_sync(lambda: self._with_client(self.max_concurrency, generate))

    def generate_texts(self, prompts, max_tokens=8, stop=None, temperature=0.0,
                       top_p=None, top_k=None):
        if not prompts:
            return []
        async def generate(sdk):
            if self.generation_api != "chat":
                effective = [render_model_prompt(prompt, self.prompt_renderer) for prompt in prompts]
                if self.add_bos_token:
                    effective = [add_bos_prefix(prompt, self.bos_token) for prompt in effective]
                size = self.completion_batch_size or ((len(effective) + self.completion_concurrency - 1) // self.completion_concurrency)
                batches = [(effective[start:start + size], f"{self.request_role}-{next(self._request_ids)}")
                           for start in range(0, len(effective), size)]
                async def complete(batch):
                    return await self._complete(sdk, *batch, max_tokens, stop, temperature, top_p, top_k)
                if self.completion_concurrency == 1 or len(batches) == 1:
                    grouped = [await complete(batch) for batch in batches]
                else:
                    grouped = await _bounded_map(batches, self.completion_concurrency, complete)
                return [result for batch in grouped for result in batch]

            async def chat(prompt):
                messages = [{"role": "user", "content": render_model_prompt(prompt, self.prompt_renderer)}]
                return await self._chat(sdk, messages, max_tokens, stop, temperature, top_p, top_k)

            async def batch():
                results = await _bounded_map(prompts, self.max_concurrency, chat)
                if self.request_role == "generation":
                    for concurrency in (128, 32):
                        failed = [i for i, result in enumerate(results) if result.error is not None]
                        if not failed:
                            break
                        retried = await _bounded_map([prompts[i] for i in failed],
                            min(self.max_concurrency, concurrency), chat)
                        for index, result in zip(failed, retried):
                            results[index] = result
                return results
            results = await batch()
            for delay in batch_retry_delays():
                if not all_failed(results):
                    break
                await asyncio.sleep(delay)
                results = await batch()
            self.request_tally.record(results)
            return require_batch_success(results, "Chat generation")

        concurrency = self.max_concurrency if self.generation_api == "chat" else self.completion_concurrency
        return _run_sync(lambda: self._with_client(concurrency, generate))

    async def _complete(self, sdk, prompts, request_id, max_tokens, stop, temperature, top_p, top_k):
        request = self._completion_request(prompts, max_tokens, stop, temperature, top_p, top_k)
        delays = batch_retry_delays()
        for attempt in range(len(delays) + 1):
            try:
                response = await sdk.completions.create(**request)
                break
            except Exception as exc:
                if attempt == len(delays):
                    self.request_tally.add(len(prompts), len(prompts))
                    raise BatchRequestError(f"Completion generation: batch of {len(prompts)} failed "
                                            f"{attempt + 1} times; {exc}") from exc
                await asyncio.sleep(delays[attempt])
        api_call = capture_api_call(self.trace_collector, role=self.request_role,
            base_url=self.base_url, request=self.wire_request(request), response=response)
        try:
            results = self._completion_results(response, prompts, api_call, request_id)
        except BatchRequestError:
            self.request_tally.add(len(prompts), len(prompts))
            raise
        self.request_tally.add(len(prompts), 0)
        return results


class AsyncOpenAICompatibleJudgeClient(AsyncOpenAICompatibleTextGenerationClient, JudgeModelClient):
    pass


class AsyncOpenAICompatibleAnswerVerifierClient(AsyncOpenAICompatibleTextGenerationClient, AnswerVerifierModelClient):
    pass
