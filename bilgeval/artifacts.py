import gzip
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import quote


OUTPUT_LEVELS = ("summary", "responses", "debug")
TRACE_MAX_REQUESTS = 50
TRACE_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024


def response_artifact_filename(run_name: str) -> str:
    safe_run_name = quote(run_name, safe="._-")
    return f"{safe_run_name}_responses.jsonl.gz"


def _json_line(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class ApiTraceCollector:
    def __init__(
        self,
        path: Path,
        *,
        max_requests: int = TRACE_MAX_REQUESTS,
        max_uncompressed_bytes: int = TRACE_MAX_UNCOMPRESSED_BYTES,
    ):
        self.path = path
        self.max_requests = max_requests
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.active = False
        self.truncated = False
        self.num_requests = 0
        self.uncompressed_bytes = 0
        self._seen: set[str] = set()
        self._stream: TextIO | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self.active = True

    def capture(self, call: dict[str, Any]) -> dict[str, str] | None:
        if not self.active:
            return None
        serialized = _json_line(call)
        trace_id = "trace-" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]
        with self._lock:
            if trace_id in self._seen:
                return {"trace_id": trace_id}
            record = {
                "schema_version": "bilgeval.api-trace/v1",
                "trace_id": trace_id,
                **call,
            }
            line = _json_line(record) + "\n"
            line_size = len(line.encode("utf-8"))
            if (
                self.num_requests >= self.max_requests
                or self.uncompressed_bytes + line_size > self.max_uncompressed_bytes
            ):
                self.truncated = True
                return None
            if self._stream is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._stream = gzip.open(self.path, "wt", encoding="utf-8")
            self._stream.write(line)
            self._stream.flush()
            self._seen.add(trace_id)
            self.num_requests += 1
            self.uncompressed_bytes += line_size
        return {"trace_id": trace_id}

    def close(self) -> dict[str, Any]:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            self.active = False
        return {
            "schema_version": "bilgeval.api-trace/v1",
            "path": self.path.name if self.path.exists() else None,
            "compression": "gzip",
            "num_requests": self.num_requests,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compressed_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "content_hash": _sha256_file(self.path) if self.path.exists() else None,
            "truncated": self.truncated,
            "max_requests": self.max_requests,
            "max_uncompressed_bytes": self.max_uncompressed_bytes,
        }


def _response_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_response_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    output = {}
    for key, item in value.items():
        if key == "api_calls":
            trace_ids = [
                call.get("trace_id")
                for call in item
                if isinstance(call, dict) and call.get("trace_id")
            ] if isinstance(item, list) else []
            if trace_ids:
                output["api_trace_ids"] = list(dict.fromkeys(trace_ids))
            continue
        output[key] = _response_value(item)
    return output


def write_response_artifact(
    path: Path,
    *,
    run_name: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for result in results:
            record = {
                "schema_version": "bilgeval.response/v1",
                "run_name": run_name,
                **_response_value(result),
            }
            file.write(_json_line(record) + "\n")
    return {
        "schema_version": "bilgeval.response/v1",
        "path": path.name,
        "format": "jsonl",
        "compression": "gzip",
        "rows": len(results),
        "compressed_bytes": path.stat().st_size,
        "content_hash": _sha256_file(path),
    }
