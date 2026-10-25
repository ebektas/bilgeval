import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluators import GeneratedTextResult, TextGenerationModelClient


RESPONSE_SCHEMA_VERSION = "bilgeval.response/v1"
REPLAY_REQUEST_ID_PREFIX = "artifact-replay:"


@dataclass(frozen=True)
class ReplayCandidate:
    text: str | None
    finish_reason: str | None
    error: str | None
    expected_prompt: str | None = None
    expected_messages: list[dict[str, str]] | None = None


def read_response_artifact(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(source, "rt", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") != RESPONSE_SCHEMA_VERSION:
                    raise ValueError(
                        f"Unsupported response artifact schema on line {line_number}: "
                        f"{row.get('schema_version')!r}."
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read replay response artifact {source}: {exc}") from exc
    if not rows:
        raise ValueError(f"Replay response artifact is empty: {source}")
    return rows


def _candidate_from_record(
    record: dict[str, Any],
    *,
    expected_prompt: str | None = None,
    expected_messages: list[dict[str, str]] | None = None,
) -> ReplayCandidate:
    text = record.get("candidate_response")
    text = text if isinstance(text, str) else None
    finish_reason = record.get("candidate_finish_reason")
    finish_reason = finish_reason if isinstance(finish_reason, str) else None
    error = None
    if not (text or "").strip():
        raw_error = record.get("error")
        error = (
            str(raw_error)
            if raw_error
            else "Saved candidate response is empty; live fallback is required."
        )
    return ReplayCandidate(
        text=text,
        finish_reason=finish_reason,
        error=error,
        expected_prompt=expected_prompt,
        expected_messages=expected_messages,
    )


def replay_candidates_from_rows(rows: list[dict[str, Any]]) -> tuple[str, list[ReplayCandidate]]:
    run_names = {str(row.get("run_name") or "") for row in rows}
    if len(run_names) != 1 or "" in run_names:
        raise ValueError("Replay artifact must contain exactly one non-empty run_name.")
    run_name = next(iter(run_names))

    indices = [row.get("idx") for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("Replay artifact rows must be ordered by contiguous zero-based idx.")

    if all(isinstance(row.get("turns"), list) for row in rows):
        candidates = []
        for row in rows:
            for turn in row["turns"]:
                messages = turn.get("candidate_messages")
                candidates.append(
                    _candidate_from_record(
                        turn,
                        expected_messages=messages if isinstance(messages, list) else None,
                    )
                )
        return run_name, candidates

    sample_counts = [
        len(row["samples"])
        for row in rows
        if isinstance(row.get("samples"), list) and row["samples"]
    ]
    num_samples = sample_counts[0] if sample_counts else 1
    if sample_counts and any(count != num_samples for count in sample_counts):
        raise ValueError("Replay artifact rows have inconsistent sample counts.")

    def prompt_for(row: dict[str, Any]) -> str | None:
        for key in ("generation_prompt_text", "prompt"):
            value = row.get(key)
            if isinstance(value, str):
                return value
        return None

    if num_samples > 1:
        if len(sample_counts) != len(rows):
            raise ValueError("Every replay artifact row must contain samples when num_samples > 1.")
        candidates = [
            _candidate_from_record(
                row["samples"][sample_index],
                expected_prompt=prompt_for(row),
            )
            for sample_index in range(num_samples)
            for row in rows
        ]
    else:
        candidates = [
            _candidate_from_record(row, expected_prompt=prompt_for(row)) for row in rows
        ]
    return run_name, candidates


class ArtifactReplayTextGenerationClient(TextGenerationModelClient):
    def __init__(
        self,
        candidates: list[ReplayCandidate],
        *,
        fallback_client: TextGenerationModelClient | None = None,
    ):
        self._candidates = candidates
        self._cursor = 0
        self.fallback_client = fallback_client
        self.num_replayed = 0
        self.num_fallback = 0
        self.generation_api = getattr(fallback_client, "generation_api", "chat")
        self.model = getattr(fallback_client, "model", "artifact-replay")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        expected_run_name: str,
        fallback_client: TextGenerationModelClient | None = None,
    ) -> "ArtifactReplayTextGenerationClient":
        run_name, candidates = replay_candidates_from_rows(read_response_artifact(path))
        if run_name != expected_run_name:
            raise ValueError(
                f"Replay artifact run_name {run_name!r} does not match selected run "
                f"{expected_run_name!r}."
            )
        return cls(candidates, fallback_client=fallback_client)

    @property
    def remaining(self) -> int:
        return len(self._candidates) - self._cursor

    def _take(self, count: int) -> list[tuple[int, ReplayCandidate]]:
        end = self._cursor + count
        if end > len(self._candidates):
            raise ValueError(
                f"Replay artifact exhausted: requested {count} more candidates with "
                f"only {self.remaining} remaining."
            )
        items = list(enumerate(self._candidates[self._cursor:end], start=self._cursor))
        self._cursor = end
        return items

    @staticmethod
    def _saved_result(index: int, candidate: ReplayCandidate) -> GeneratedTextResult:
        return GeneratedTextResult(
            text=candidate.text,
            error=candidate.error,
            finish_reason=candidate.finish_reason,
            request_id=f"{REPLAY_REQUEST_ID_PREFIX}{index}",
        )

    def generate_texts(
        self,
        prompts: list[str],
        max_tokens: int = 8,
        stop: list[str] | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> list[GeneratedTextResult]:
        items = self._take(len(prompts))
        results: list[GeneratedTextResult | None] = [None] * len(prompts)
        fallback_positions = []
        fallback_prompts = []
        for position, ((source_index, candidate), prompt) in enumerate(zip(items, prompts)):
            if candidate.expected_prompt is not None and candidate.expected_prompt != prompt:
                raise ValueError(
                    f"Replay prompt mismatch at candidate {source_index}; artifact and current "
                    "task/config do not match."
                )
            if candidate.error is None:
                results[position] = self._saved_result(source_index, candidate)
                self.num_replayed += 1
            else:
                fallback_positions.append(position)
                fallback_prompts.append(prompt)
        if fallback_positions:
            if self.fallback_client is None:
                for position in fallback_positions:
                    source_index, candidate = items[position]
                    results[position] = self._saved_result(source_index, candidate)
            else:
                generated = self.fallback_client.generate_texts(
                    fallback_prompts,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                if len(generated) != len(fallback_positions):
                    raise ValueError("Live replay fallback returned a mismatched result count.")
                for position, result in zip(fallback_positions, generated):
                    results[position] = result
                self.num_fallback += len(fallback_positions)
        return [result for result in results if result is not None]

    def generate_conversations(
        self,
        conversations: list[list[dict[str, str]]],
        max_tokens: int = 8,
        stop: list[str] | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> list[GeneratedTextResult]:
        items = self._take(len(conversations))
        results: list[GeneratedTextResult | None] = [None] * len(conversations)
        fallback_positions = []
        fallback_conversations = []
        for position, ((source_index, candidate), messages) in enumerate(
            zip(items, conversations)
        ):
            if candidate.expected_messages is not None and candidate.expected_messages != messages:
                raise ValueError(
                    f"Replay conversation mismatch at candidate {source_index}; artifact and "
                    "current task/config do not match."
                )
            if candidate.error is None:
                results[position] = self._saved_result(source_index, candidate)
                self.num_replayed += 1
            else:
                fallback_positions.append(position)
                fallback_conversations.append(messages)
        if fallback_positions:
            if self.fallback_client is None:
                for position in fallback_positions:
                    source_index, candidate = items[position]
                    results[position] = self._saved_result(source_index, candidate)
            else:
                generated = self.fallback_client.generate_conversations(
                    fallback_conversations,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                if len(generated) != len(fallback_positions):
                    raise ValueError("Live replay fallback returned a mismatched result count.")
                for position, result in zip(fallback_positions, generated):
                    results[position] = result
                self.num_fallback += len(fallback_positions)
        return [result for result in results if result is not None]
