"""The optional async transport must reach all text roles and run artifacts."""

import builtins
import json
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from openai import OpenAI

from bilgeval.cli import build_parser
from bilgeval.evaluators import OpenAICompatibleTextGenerationClient


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


def test_cli_generation_client_default_flag_and_environment(monkeypatch):
    monkeypatch.delenv("BILGEVAL_GENERATION_CLIENT", raising=False)
    assert build_parser().parse_args([]).generation_client == "sync"
    assert build_parser().parse_args(["--generation-client", "async"]).generation_client == "async"
    monkeypatch.setenv("BILGEVAL_GENERATION_CLIENT", "async")
    assert build_parser().parse_args([]).generation_client == "async"
    assert build_parser().parse_args(["--generation-client", "sync"]).generation_client == "sync"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--generation-client", "invalid"])


@pytest.mark.parametrize("selection", ["", "threaded", None, 1])
def test_invalid_client_selection_fails_before_creating_output(tmp_path, selection):
    from bilgeval.main import run_benchmark

    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="generation_client"):
        run_benchmark([], str(output), generation_client=selection)
    assert not output.exists()


@pytest.mark.parametrize("selection", ["sync", "async"])
def test_dry_run_records_client_without_loading_optional_module(
    tmp_path, monkeypatch, fixture_registry, selection,
):
    from bilgeval.main import run_benchmark

    original_import = builtins.__import__

    def deny_optional(name, *args, **kwargs):
        if name in {"async_generation", "bilgeval.async_generation", "httpx_aiohttp", "aiohttp"}:
            raise AssertionError("A dry run must not require the async transport.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_optional)
    report = run_benchmark(
        ["fixture_math_regex"], str(tmp_path), model="fixture",
        generation_client=selection, dry_run=True,
    )
    manifest = json.loads(Path(report["manifest_file"]).read_text())
    assert manifest["runtime"]["generation_client"] == selection


def test_cli_carries_async_choice_into_manifest(tmp_path, monkeypatch, fixture_registry):
    from bilgeval.cli import main

    registry = Path(__file__).parent / "fixtures" / "registry_v1.json"
    monkeypatch.setattr(sys, "argv", [
        "bilgeval", "fixture_math_regex", "--registry-file", str(registry),
        "--output-dir", str(tmp_path), "--model", "fixture",
        "--generation-client", "async", "--dry-run", "--silent",
    ])
    main()
    (manifest_file,) = tmp_path.glob("*/run_manifest.json")
    assert json.loads(manifest_file.read_text())["runtime"]["generation_client"] == "async"


def test_cli_missing_async_extra_exits_with_install_instruction(
    tmp_path, monkeypatch, capsys, fixture_registry,
):
    from bilgeval.cli import main

    def missing_extra(**kwargs):
        raise ImportError("Async generation requires pip install 'bilgeval[async]'")

    optional = ModuleType("bilgeval.async_generation")
    for name in ("TextGeneration", "Judge", "AnswerVerifier"):
        setattr(optional, f"AsyncOpenAICompatible{name}Client", missing_extra)
    monkeypatch.setitem(sys.modules, "bilgeval.async_generation", optional)
    registry = Path(__file__).parent / "fixtures" / "registry_v1.json"
    monkeypatch.setattr(sys, "argv", [
        "bilgeval", "fixture_math_regex", "--registry-file", str(registry),
        "--output-dir", str(tmp_path), "--model", "fixture",
        "--generation-client", "async", "--bos-mode", "never",
        "--skip-preflight", "--silent",
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "bilgeval[async]" in error
    assert "Traceback" not in error


@pytest.mark.parametrize("selection", ["sync", "async"])
def test_candidate_judge_verifier_and_report_use_selected_client(
    tmp_path, monkeypatch, fixture_registry, selection,
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
    registry["tasks"]["fixture_math_regex"]["scorer"]["answer_verification"] = "llm_as_regex"
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))
    runtime.load_registry_file(str(registry_path))
    constructed = []
    clients = []
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        text = {"candidate": "A", "judge": "CORRECT", "verifier": '{"match":true}'}[body["model"]]
        if "messages" in body:
            return httpx.Response(200, json={
                "id": "chat", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
            })
        return httpx.Response(200, json={
            "id": "completion", "object": "text_completion", "created": 0, "model": body["model"],
            "choices": [{"index": index, "text": text, "finish_reason": "stop"}
                        for index in range(len(body["prompt"]))],
        })

    def factory(transport, role):
        def create(**kwargs):
            constructed.append((transport, role, kwargs))
            client = OpenAICompatibleTextGenerationClient(**kwargs)
            client.client.close()
            client.client = OpenAI(
                base_url="http://fixture.test/v1", api_key="EMPTY", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            )
            clients.append(client)
            return client
        return create

    optional = ModuleType("bilgeval.async_generation")
    for role, name in [
        ("generation", "TextGeneration"), ("judge", "Judge"), ("answer_verifier", "AnswerVerifier"),
    ]:
        monkeypatch.setattr(runtime, f"OpenAICompatible{name}Client", factory("sync", role))
        setattr(optional, f"AsyncOpenAICompatible{name}Client", factory("async", role))
    monkeypatch.setitem(sys.modules, "bilgeval.async_generation", optional)

    try:
        report = runtime.run_benchmark(
            ["fixture_mc_logprob", "fixture_math_regex"], str(tmp_path / "run"),
            model="candidate", judge_model="judge", verifier_model="verifier",
            judge_generation_api="completions",
            generation_concurrency=7, judge_concurrency=5, verifier_concurrency=3,
            completion_concurrency=2, completion_batch_size=1,
            skip_preflight=True, bos_mode="off", bootstrap_resamples=20,
            **({"generation_client": selection} if selection == "async" else {}),
        )
        assert [(transport, role) for transport, role, _ in constructed] == [
            (selection, "generation"), (selection, "judge"), (selection, "answer_verifier"),
        ]
        assert [kwargs["max_concurrency"] for _, _, kwargs in constructed] == [7, 5, 3]
        assert {request["model"] for request in requests} == {"candidate", "judge", "verifier"}
        assert all(dataset["accuracy"] == 1.0 for dataset in report["datasets"].values())
        assert report["generation_client"] == selection
        saved = json.loads(Path(report["manifest_file"]).read_text())
        assert saved["runtime"]["generation_client"] == selection
        assert all(client.request_tally.snapshot()[1] == 0 for client in clients)
    finally:
        for client in clients:
            client.client.close()
