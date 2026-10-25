"""What a model call returns, and how samples of one example aggregate."""

import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .statistics import pass_at_k_estimate


@dataclass(frozen=True)
class CompletionScoreResult:
    score: float | None = None
    error: str | None = None
    num_completion_tokens: int = 0
    normalization: str = "none"
    normalization_denominator: int | None = None
    api_call: dict[str, Any] | None = None


@dataclass(frozen=True)
class GeneratedTextResult:
    text: str | None = None
    error: str | None = None
    api_call: dict[str, Any] | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    reasoning_chars: int | None = None
    request_failed: bool = False


@dataclass(frozen=True)
class GeneratedToolCallResult:
    tool_calls: list[dict[str, Any]] | None = None
    content: str | None = None
    error: str | None = None
    api_call: dict[str, Any] | None = None
    request_id: str | None = None
    request_failed: bool = False


@dataclass
class RequestTally:
    """What one client asked its endpoint for (generations, scored prompts) and
    how many of those it never got: the request still failed after bilgeval's
    own retries. A lost item is scored as a wrong answer, so a run that lost
    many measures the endpoint, not the model."""

    sent: int = 0
    failed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, sent: int, failed: int) -> None:
        with self._lock:
            self.sent += sent
            self.failed += failed

    def record(self, results: list[Any]) -> None:
        self.add(len(results), sum(bool(getattr(result, "request_failed", False)) for result in results))

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.sent, self.failed


def generated_request_count(results: list[GeneratedTextResult]) -> int:
    """How many endpoint requests a batch of results came from."""
    request_ids = {
        result.request_id
        for result in results
        if result.request_id is not None
        and not result.request_id.startswith("artifact-replay:")
    }
    if request_ids:
        return len(request_ids)
    if results and all(
        result.request_id is not None
        and result.request_id.startswith("artifact-replay:")
        for result in results
    ):
        return 0
    calls = {id(result.api_call) for result in results if result.api_call is not None}
    return len(calls) or 1


MULTI_SAMPLE_AGGREGATIONS = frozenset({"first", "consensus", "mean"})


def consensus_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    vote_keys = [
        str(sample["prediction_key"])
        for sample in samples
        if sample.get("prediction_key") is not None
    ]
    if not vote_keys:
        return {
            "prediction_key": None,
            "num_votes": 0,
            "vote_counts": {},
            "tie": False,
            "correct": False,
            "failed": True,
        }
    counts = Counter(vote_keys)
    max_votes = max(counts.values())
    winners = sorted(key for key, count in counts.items() if count == max_votes)
    tied = len(winners) > 1
    winner = None if tied else winners[0]
    winner_samples = [
        sample for sample in samples if str(sample.get("prediction_key")) == winner
    ]
    return {
        "prediction_key": winner,
        "num_votes": max_votes,
        "vote_counts": dict(sorted(counts.items())),
        "tie": tied,
        "correct": bool(winner_samples and winner_samples[0].get("correct")) if not tied else False,
        "failed": False,
    }


def sample_metrics(
    samples: list[dict[str, Any]],
    pass_at_k: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    num_samples = len(samples)
    num_correct = sum(bool(sample.get("correct")) for sample in samples)
    return {
        "num_samples": num_samples,
        "num_correct_samples": num_correct,
        "sample_accuracy": num_correct / num_samples if num_samples else 0.0,
        "pass_at_k": {
            str(k): pass_at_k_estimate(num_samples, num_correct, k)
            for k in pass_at_k
        },
    }


def aggregate_multi_sample_metrics(
    results: list[dict[str, Any]],
    *,
    num_samples: int,
    pass_at_k: list[int] | tuple[int, ...],
    aggregation: str,
) -> dict[str, Any]:
    total_attempts = len(results) * num_samples
    total_correct = sum(
        int(bool(sample.get("correct")))
        for result in results
        for sample in result.get("samples", [])
    )
    consensus_values = [result["consensus"] for result in results]
    return {
        "num_samples": num_samples,
        "primary_aggregation": aggregation,
        "total_generation_attempts": total_attempts,
        "total_correct_samples": total_correct,
        "average_sample_accuracy": (
            total_correct / total_attempts if total_attempts else 0.0
        ),
        "pass_at_k": {
            str(k): (
                sum(result["sample_metrics"]["pass_at_k"][str(k)] for result in results)
                / len(results)
                if results
                else 0.0
            )
            for k in pass_at_k
        },
        "consensus_accuracy": (
            sum(bool(value["correct"]) for value in consensus_values) / len(consensus_values)
            if consensus_values
            else 0.0
        ),
        "consensus_ties": sum(bool(value["tie"]) for value in consensus_values),
        "consensus_no_vote": sum(bool(value["failed"]) for value in consensus_values),
    }


def mean_sample_report(report: dict[str, Any], metric_name: str) -> dict[str, Any]:
    """Headline = mean over all generated samples of all examples.

    Turns a per-example (first sample) report into a continuous one; a failed
    or unparseable attempt is an unsuccessful attempt.
    """
    samples = [
        sample
        for result in report["results"]
        for sample in result.get("samples", [])
    ]
    if not samples:
        raise ValueError("A mean-over-samples headline needs per-sample results.")
    scored = [sample for sample in samples if not sample.get("failed")]
    value = sum(bool(sample.get("correct")) for sample in samples) / len(samples)
    converted = {
        key: item
        for key, item in report.items()
        if key not in {"accuracy", "correct"}
    }
    converted.update(
        {
            "metric_name": metric_name,
            "score": value,
            "score_on_scored": (
                sum(bool(sample.get("correct")) for sample in scored) / len(scored)
                if scored
                else None
            ),
            "score_scale": {"min": 0.0, "max": 1.0, "higher_is_better": True},
            "failure_policy": (
                "mean over every generated sample of every example; a failed or "
                "unparseable attempt is an unsuccessful attempt"
            ),
            "first_sample_accuracy": (
                report["correct"] / report["num_examples"] if report["num_examples"] else 0.0
            ),
            "num_samples_total": len(samples),
            "num_samples_failed": len(samples) - len(scored),
        }
    )
    return converted
