import gzip
import json

import pytest

from bilgeval.evaluators import GeneratedTextResult, generated_request_count
from bilgeval.replay import ArtifactReplayTextGenerationClient


def write_artifact(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


class FallbackClient:
    def __init__(self):
        self.prompts = []

    def generate_texts(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        return [GeneratedTextResult(text="regenerated", request_id="live-1") for _ in prompts]


def test_replay_uses_saved_text_and_live_fallback_only_for_missing(tmp_path):
    path = tmp_path / "responses.jsonl.gz"
    write_artifact(
        path,
        [
            {
                "schema_version": "bilgeval.response/v1",
                "run_name": "task:run",
                "idx": 0,
                "generation_prompt_text": "p0",
                "candidate_response": "saved",
                "candidate_finish_reason": "stop",
            },
            {
                "schema_version": "bilgeval.response/v1",
                "run_name": "task:run",
                "idx": 1,
                "generation_prompt_text": "p1",
                "candidate_response": None,
                "error": "old timeout",
            },
        ],
    )
    fallback = FallbackClient()
    client = ArtifactReplayTextGenerationClient.from_path(
        path, expected_run_name="task:run", fallback_client=fallback
    )

    results = client.generate_texts(["p0", "p1"])

    assert [result.text for result in results] == ["saved", "regenerated"]
    assert fallback.prompts == ["p1"]
    assert client.num_replayed == 1
    assert client.num_fallback == 1
    assert client.remaining == 0
    assert generated_request_count(results) == 1


def test_replay_rejects_prompt_mismatch(tmp_path):
    path = tmp_path / "responses.jsonl.gz"
    write_artifact(
        path,
        [
            {
                "schema_version": "bilgeval.response/v1",
                "run_name": "task:run",
                "idx": 0,
                "generation_prompt_text": "old prompt",
                "candidate_response": "saved",
            }
        ],
    )
    client = ArtifactReplayTextGenerationClient.from_path(
        path, expected_run_name="task:run"
    )

    with pytest.raises(ValueError, match="Replay prompt mismatch"):
        client.generate_texts(["new prompt"])


def test_replay_only_results_count_as_zero_requests(tmp_path):
    path = tmp_path / "responses.jsonl.gz"
    write_artifact(
        path,
        [
            {
                "schema_version": "bilgeval.response/v1",
                "run_name": "task:run",
                "idx": 0,
                "prompt": "p0",
                "candidate_response": "saved",
            }
        ],
    )
    client = ArtifactReplayTextGenerationClient.from_path(path, expected_run_name="task:run")

    assert generated_request_count(client.generate_texts(["p0"])) == 0


def test_replay_client_reports_the_live_client_transport():
    from types import SimpleNamespace

    from bilgeval.replay import ArtifactReplayTextGenerationClient

    live = SimpleNamespace(generation_api="chat", model="m")
    client = ArtifactReplayTextGenerationClient([], fallback_client=live)
    assert (client.generation_api, client.model) == ("chat", "m")
    assert ArtifactReplayTextGenerationClient([]).generation_api == "chat"
