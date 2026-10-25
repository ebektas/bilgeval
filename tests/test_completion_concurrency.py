"""Completion HTTP batching must preserve the identity of every input."""

import gzip
import json
import threading
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from bilgeval.artifacts import ApiTraceCollector
from bilgeval.batching import BatchRequestError
from bilgeval.cli import build_parser
from bilgeval.evaluators import OpenAICompatibleTextGenerationClient
from bilgeval.registry_bundle import PromptRendererSpec
from bilgeval.results import generated_request_count


@pytest.fixture
def fixture_registry():
    from bilgeval import main as runtime
    from bilgeval.generation_prompts import GENERATION_PROMPTS
    from bilgeval.judge_prompts import JUDGE_PROMPTS
    from bilgeval.registry import DATASETS

    mappings = [DATASETS, GENERATION_PROMPTS, JUDGE_PROMPTS,
                runtime.ACTIVE_REGISTRY_CONTEXT, runtime.ACTIVE_MODEL_PROFILES]
    snapshots = [dict(mapping) for mapping in mappings]
    bundle = runtime.ACTIVE_REGISTRY_BUNDLE
    try:
        runtime.load_registry_file(str(Path(__file__).parent / "fixtures" / "registry_v1.json"))
        yield
    finally:
        for mapping, snapshot in zip(mappings, snapshots):
            mapping.clear()
            mapping.update(snapshot)
        runtime.ACTIVE_REGISTRY_BUNDLE = bundle


def completion_response(request, *, choices=None):
    body = json.loads(request.content)
    if choices is None:
        choices = [
            {"index": index, "text": prompt, "finish_reason": "stop"}
            for index, prompt in reversed(list(enumerate(body["prompt"])))
        ]
    return httpx.Response(
        200,
        json={"id": "fixture", "object": "text_completion", "created": 0,
              "model": body["model"], "choices": choices},
    )


def make_client(handler, **kwargs):
    kwargs.setdefault("model", "fixture")
    client = OpenAICompatibleTextGenerationClient(**kwargs)
    client.client.close()
    client.client = OpenAI(
        base_url="http://fixture.test/v1", api_key="EMPTY", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client


def test_default_completion_chunk_remains_one_http_request():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion_response(request)

    client = make_client(handler)
    try:
        results = client.generate_texts(["zero", "one", "two"])
        assert [result.text for result in results] == ["zero", "one", "two"]
        assert len(requests) == 1
        assert requests[0]["prompt"] == ["zero", "one", "two"]
        assert generated_request_count(results) == 1
        assert client.request_tally.snapshot() == (3, 0)
    finally:
        client.client.close()


def test_parallel_completions_bound_concurrency_and_restore_input_order(tmp_path):
    barrier = threading.Barrier(2)
    later_finished = threading.Event()
    lock = threading.Lock()
    active = maximum_active = 0
    finished = []
    trace = ApiTraceCollector(tmp_path / "trace.jsonl.gz")
    trace.start()

    def handler(request):
        nonlocal active, maximum_active
        prompt = json.loads(request.content)["prompt"][0]
        with lock:
            active += 1
            maximum_active = max(active, maximum_active)
        try:
            if prompt in {"0", "1"}:
                barrier.wait(timeout=5)
            if prompt == "0":
                assert later_finished.wait(timeout=5)
            with lock:
                finished.append(prompt)
            if prompt == "1":
                later_finished.set()
            return completion_response(request)
        finally:
            with lock:
                active -= 1

    client = make_client(handler, completion_concurrency=2, completion_batch_size=1,
                         trace_collector=trace)
    try:
        prompts = [str(index) for index in range(20)]
        results = client.generate_texts(prompts)
        assert maximum_active == 2
        assert finished.index("1") < finished.index("0")
        assert [result.text for result in results] == prompts
        assert generated_request_count(results) == 20
        assert len({result.request_id for result in results}) == 20
        assert client.request_tally.snapshot() == (20, 0)
    finally:
        client.client.close()
        trace.close()
    with gzip.open(trace.path, "rt") as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 20
    assert len({record["trace_id"] for record in records}) == 20


@pytest.mark.parametrize("settings, expected_sizes", [
    ({"completion_concurrency": 2}, [3, 2]),
    ({"completion_concurrency": 1, "completion_batch_size": 2}, [2, 2, 1]),
    ({"completion_concurrency": 3, "completion_batch_size": 2}, [2, 2, 1]),
])
def test_completion_batch_sizes_and_serialization(settings, expected_sizes):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion_response(request)

    client = make_client(
        handler, **settings, add_bos_token=True,
        prompt_renderer=PromptRendererSpec(mode="template", template_id="fixture",
                                           template="<user>{prompt}</user>"),
        default_extra_body={"seed": 42},
    )
    try:
        results = client.generate_texts(["0", "1", "2", "3", "4"], max_tokens=9,
                                        stop=["stop"], temperature=0.4, top_p=0.7, top_k=3)
        assert sorted(map(lambda value: len(value["prompt"]), requests)) == sorted(expected_sizes)
        assert [result.text for result in results] == [
            f"<bos><user>{index}</user>" for index in range(5)
        ]
        for request in requests:
            assert request["max_tokens"] == 9
            assert request["stop"] == ["stop"]
            assert request["temperature"] == 0.4
            assert request["top_p"] == 0.7
            assert request["top_k"] == 3
            assert request["seed"] == 42
        assert generated_request_count(results) == len(expected_sizes)
        assert client.request_tally.snapshot() == (5, 0)
    finally:
        client.client.close()


@pytest.mark.parametrize("indices", [
    [0, 0], [1, 2], [-1, 0], [0], [0, 1, 2], [0, None], [0, "1"], [0, 1.0], [0, True],
])
def test_completion_indices_must_match_the_batch(indices):
    def handler(request):
        return completion_response(request, choices=[
            {"index": index, "text": str(index), "finish_reason": "stop"}
            for index in indices
        ])

    client = make_client(handler)
    try:
        with pytest.raises(BatchRequestError, match="Completion generation"):
            client.generate_texts(["zero", "one"])
        assert client.request_tally.snapshot() == (2, 2)
    finally:
        client.client.close()


@pytest.mark.parametrize("choices", [None, 5, {"index": 0}, "invalid"])
def test_malformed_completion_payload_aborts_with_failure_accounting(choices):
    def handler(request):
        return httpx.Response(200, json={"choices": choices})

    client = make_client(handler)
    try:
        with pytest.raises(BatchRequestError, match="Completion generation"):
            client.generate_texts(["zero", "one"])
        assert client.request_tally.snapshot() == (2, 2)
    finally:
        client.client.close()


def test_failed_parallel_batch_aborts_and_accounts_for_successful_siblings(monkeypatch):
    monkeypatch.setenv("BILGEVAL_BATCH_RETRY_DELAYS", "")
    barrier = threading.Barrier(2)

    def handler(request):
        barrier.wait(timeout=5)
        if json.loads(request.content)["prompt"][0] == "bad":
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return completion_response(request)

    client = make_client(handler, completion_concurrency=2, completion_batch_size=1)
    try:
        with pytest.raises(BatchRequestError, match="Completion generation"):
            client.generate_texts(["bad", "good"])
        assert client.request_tally.snapshot() == (2, 1)
    finally:
        client.client.close()


def test_parallel_batch_retry_preserves_request_identity_and_item_accounting(monkeypatch):
    monkeypatch.setenv("BILGEVAL_BATCH_RETRY_DELAYS", "0")
    attempts = {}
    lock = threading.Lock()

    def handler(request):
        first_prompt = json.loads(request.content)["prompt"][0]
        with lock:
            attempts[first_prompt] = attempts.get(first_prompt, 0) + 1
            attempt = attempts[first_prompt]
        if first_prompt == "retry0" and attempt == 1:
            return httpx.Response(503, json={"error": {"message": "try again"}})
        return completion_response(request)

    client = make_client(handler, completion_concurrency=2, completion_batch_size=2)
    try:
        prompts = ["retry0", "retry1", "good0", "good1"]
        results = client.generate_texts(prompts)
        assert [result.text for result in results] == prompts
        assert attempts == {"retry0": 2, "good0": 1}
        assert results[0].request_id == results[1].request_id
        assert results[2].request_id == results[3].request_id
        assert generated_request_count(results) == 2
        assert client.request_tally.snapshot() == (4, 0)
    finally:
        client.client.close()


def test_failed_parallel_chunk_finishes_and_counts_every_submitted_batch(monkeypatch):
    monkeypatch.setenv("BILGEVAL_BATCH_RETRY_DELAYS", "")
    barrier = threading.Barrier(2)
    completed = []

    def handler(request):
        prompt = json.loads(request.content)["prompt"][0]
        barrier.wait(timeout=5)
        completed.append(prompt)
        if prompt == "bad":
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return completion_response(request)

    client = make_client(handler, completion_concurrency=2, completion_batch_size=1)
    try:
        with pytest.raises(BatchRequestError):
            client.generate_texts(["bad", "good1", "good2", "good3"])
        assert sorted(completed) == ["bad", "good1", "good2", "good3"]
        assert client.request_tally.snapshot() == (4, 1)
    finally:
        client.client.close()


@pytest.mark.parametrize("kwargs", [
    {"completion_concurrency": 0}, {"completion_concurrency": -1},
    {"completion_batch_size": 0}, {"completion_batch_size": -1},
    {"completion_concurrency": 1.5}, {"completion_batch_size": 1.5},
    {"completion_concurrency": True}, {"completion_batch_size": True},
])
def test_invalid_completion_settings_rejected(kwargs):
    with pytest.raises(ValueError, match="completion_"):
        OpenAICompatibleTextGenerationClient(**kwargs)


def test_cli_completion_settings_defaults_flags_and_environment(monkeypatch):
    monkeypatch.delenv("BILGEVAL_COMPLETION_CONCURRENCY", raising=False)
    monkeypatch.delenv("BILGEVAL_COMPLETION_BATCH_SIZE", raising=False)
    args = build_parser().parse_args([])
    assert args.completion_concurrency == 1
    assert args.completion_batch_size is None
    args = build_parser().parse_args(["--completion-concurrency", "3", "--completion-batch-size", "7"])
    assert (args.completion_concurrency, args.completion_batch_size) == (3, 7)
    monkeypatch.setenv("BILGEVAL_COMPLETION_CONCURRENCY", "4")
    monkeypatch.setenv("BILGEVAL_COMPLETION_BATCH_SIZE", "8")
    args = build_parser().parse_args([])
    assert (args.completion_concurrency, args.completion_batch_size) == (4, 8)


def test_completion_settings_are_written_to_run_manifest(tmp_path, fixture_registry):
    from bilgeval.main import run_benchmark

    report = run_benchmark(
        dataset_names=["fixture_math_regex"], output_dir=str(tmp_path),
        model="fixture-model", completion_concurrency=3, completion_batch_size=2,
        dry_run=True,
    )
    manifest = json.loads(Path(report["manifest_file"]).read_text())
    assert manifest["runtime"]["completion_concurrency"] == 3
    assert manifest["runtime"]["completion_batch_size"] == 2


@pytest.mark.parametrize("kwargs", [
    {"completion_concurrency": 0}, {"completion_batch_size": 0},
    {"completion_concurrency": 1.5}, {"completion_batch_size": True},
])
def test_run_benchmark_validates_completion_settings(tmp_path, kwargs):
    from bilgeval.main import run_benchmark

    with pytest.raises(ValueError, match="completion_"):
        run_benchmark(dataset_names=[], output_dir=str(tmp_path), **kwargs)


def test_cli_writes_completion_settings_into_real_manifest(tmp_path, monkeypatch, fixture_registry):
    from bilgeval.cli import main

    registry = Path(__file__).parent / "fixtures" / "registry_v1.json"
    monkeypatch.setattr("sys.argv", [
        "bilgeval", "fixture_math_regex", "--registry-file", str(registry),
        "--output-dir", str(tmp_path), "--model", "fixture-model", "--dry-run", "--silent",
        "--completion-concurrency", "5", "--completion-batch-size", "3",
    ])
    main()
    (manifest_file,) = tmp_path.glob("*/run_manifest.json")
    runtime = json.loads(manifest_file.read_text())["runtime"]
    assert runtime["completion_concurrency"] == 5
    assert runtime["completion_batch_size"] == 3


def test_benchmark_candidate_and_completion_judge_use_global_completion_settings(
    tmp_path, monkeypatch, fixture_registry,
):
    from bilgeval import main as runtime

    fixtures = Path(__file__).parent / "fixtures"
    registry = json.loads((fixtures / "registry_v1.json").read_text())
    for dataset in registry["datasets"].values():
        dataset["uri"] = str((fixtures / dataset["uri"]).resolve())
    task = registry["tasks"]["fixture_mc_logprob"]
    task["request_type"] = "generate"
    task["generation_api"] = "completions"
    task["scorer"] = {"id": "mc_judge", "score_target": "label",
                      "judge_prompt": "turkish_mc_correctness_v2"}
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))
    runtime.load_registry_file(str(registry_path))
    requests = []
    clients = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return completion_response(request, choices=[
            {"index": index, "text": "CORRECT" if body["model"] == "judge" else "A",
             "finish_reason": "stop"}
            for index in range(len(body["prompt"]))
        ])

    def create_client(**kwargs):
        client = make_client(handler, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(runtime, "OpenAICompatibleTextGenerationClient", create_client)
    monkeypatch.setattr(runtime, "OpenAICompatibleJudgeClient", create_client)
    try:
        report = runtime.run_benchmark(
            dataset_names=["fixture_mc_logprob"], output_dir=str(tmp_path), model="candidate",
            judge_model="judge", judge_generation_api="completions", skip_preflight=True,
            completion_concurrency=2, completion_batch_size=1, bootstrap_resamples=20,
        )
        assert len(requests) == 4
        assert all(len(request["prompt"]) == 1 for request in requests)
        assert sum(request["model"] == "candidate" for request in requests) == 2
        assert sum(request["model"] == "judge" for request in requests) == 2
        (dataset,) = report["datasets"].values()
        assert dataset["accuracy"] == 1.0
        assert dataset["num_requests"] == 4
        assert [client.request_tally.snapshot() for client in clients] == [(2, 0), (2, 0)]
    finally:
        for client in clients:
            client.client.close()


def test_completion_options_do_not_change_chat_requests():
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={
            "id": "fixture", "object": "chat.completion", "created": 0, "model": "fixture",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": body["messages"][0]["content"]}}],
        })

    client = make_client(handler, generation_api="chat", max_concurrency=1,
                         completion_concurrency=5, completion_batch_size=2)
    try:
        results = client.generate_texts(["zero", "one", "two"])
        assert [result.text for result in results] == ["zero", "one", "two"]
        assert len(requests) == 3
        assert all("messages" in request and "prompt" not in request for request in requests)
    finally:
        client.client.close()
