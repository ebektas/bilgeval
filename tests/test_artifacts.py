import gzip
import json

from bilgeval.artifacts import (
    ApiTraceCollector,
    response_artifact_filename,
    write_response_artifact,
)


def read_jsonl_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def test_trace_collector_ignores_calls_until_started(tmp_path):
    path = tmp_path / "api_traces.jsonl.gz"
    collector = ApiTraceCollector(path)

    assert collector.capture({"request": {"prompt": "ignored"}}) is None
    metadata = collector.close()

    assert not path.exists()
    assert metadata["num_requests"] == 0
    assert metadata["path"] is None


def test_trace_collector_deduplicates_and_caps_calls(tmp_path):
    path = tmp_path / "api_traces.jsonl.gz"
    collector = ApiTraceCollector(path, max_requests=1)
    collector.start()

    first = collector.capture({"request": {"prompt": "first"}})
    duplicate = collector.capture({"request": {"prompt": "first"}})
    capped = collector.capture({"request": {"prompt": "second"}})
    metadata = collector.close()

    assert first == duplicate
    assert capped is None
    assert metadata["num_requests"] == 1
    assert metadata["truncated"] is True
    records = read_jsonl_gzip(path)
    assert len(records) == 1
    assert records[0]["trace_id"] == first["trace_id"]
    assert records[0]["schema_version"] == "bilgeval.api-trace/v1"


def test_response_artifact_replaces_api_calls_with_trace_ids(tmp_path):
    path = tmp_path / "run_responses.jsonl.gz"
    metadata = write_response_artifact(
        path,
        run_name="run",
        results=[
            {
                "dataset": "example",
                "samples": [
                    {
                        "prediction": "A",
                        "api_calls": [
                            {"trace_id": "trace-one"},
                            {"trace_id": "trace-one"},
                            {"trace_id": "trace-two"},
                        ],
                    }
                ],
            }
        ],
    )

    records = read_jsonl_gzip(path)
    sample = records[0]["samples"][0]
    assert "api_calls" not in sample
    assert sample["api_trace_ids"] == ["trace-one", "trace-two"]
    assert records[0]["schema_version"] == "bilgeval.response/v1"
    assert metadata["rows"] == 1
    assert metadata["compression"] == "gzip"
    assert metadata["content_hash"].startswith("sha256:")


def test_response_artifact_filename_is_windows_safe_and_injective():
    first = response_artifact_filename("tr_arc:posttrain-chat")
    second = response_artifact_filename("tr_arc%3Aposttrain-chat")

    assert first == "tr_arc%3Aposttrain-chat_responses.jsonl.gz"
    assert ":" not in first
    assert first != second
