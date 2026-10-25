"""The one generation loop: requests in batches, failures as per-item results.

Every evaluator sends its prompts through `generate_batches`. A batch whose
call raises becomes one error result per input (an endpoint that is really
gone raises BatchRequestError from the client, which propagates and aborts
the run); a batch that answers with the wrong number of results is reported
as such; request counts come from the results' request ids.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable

from .results import GeneratedTextResult, generated_request_count


class BatchRequestError(RuntimeError):
    """Raised when an entire user-sized model request batch fails."""


def require_batch_success(results: list[Any], role: str) -> list[Any]:
    if results and all(getattr(result, "error", None) for result in results):
        first_error = next(str(result.error) for result in results if result.error)
        raise BatchRequestError(
            f"{role}: all {len(results)} requests in the batch failed; {first_error}"
        )
    return results


def batch_retry_delays() -> tuple[float, ...]:
    """Pauses before re-sending a batch whose every request failed (an endpoint
    restart, a gateway hiccup). Only after the last one does the run abort:
    an endpoint that is really gone must not turn into a report of zeros."""
    value = os.environ.get("BILGEVAL_BATCH_RETRY_DELAYS", "15,45,90")
    return tuple(float(item) for item in value.split(",") if item.strip())


def all_failed(results: list[Any]) -> bool:
    return bool(results) and all(getattr(result, "error", None) for result in results)


@dataclass(frozen=True)
class BatchMessages:
    """The error texts a loop writes into results it could not obtain."""

    failed: str
    mismatch: str
    not_generated: str = "The example was not generated."
    propagate_batch_errors: bool = True


def _text_error(message: str) -> GeneratedTextResult:
    return GeneratedTextResult(error=message)


def generate_batches(
    inputs: list,
    call: Callable[[list], list],
    *,
    request_batch_size: int,
    messages: BatchMessages,
    on_batch: Callable[[int, list, int], None] | None = None,
    error_result: Callable[[str], Any] = _text_error,
    count_requests: Callable[[list], int] = generated_request_count,
) -> tuple[list, int]:
    """`call` each slice of `inputs`; one result per input, and the request count.

    `on_batch(start, batch_results, request_count_so_far)` runs after each
    batch. `error_result` builds the per-item result of a failed batch (tool
    calls have their own result type); `count_requests` counts a batch's
    requests from its results.
    """
    results = [error_result(messages.not_generated) for _ in inputs]
    request_count = 0
    for start in range(0, len(inputs), request_batch_size):
        batch = inputs[start : start + request_batch_size]
        try:
            batch_results = call(batch)
        except BatchRequestError as exc:
            if messages.propagate_batch_errors:
                raise
            batch_results = [error_result(f"{messages.failed}{exc}") for _ in batch]
        except Exception as exc:
            batch_results = [error_result(f"{messages.failed}{exc}") for _ in batch]
        if len(batch_results) != len(batch):
            batch_results = [error_result(messages.mismatch) for _ in batch]
        request_count += count_requests(batch_results)
        results[start : start + len(batch_results)] = batch_results
        if on_batch is not None:
            on_batch(start, batch_results, request_count)
    return results, request_count


def batch_progress(progress_callback, total: int, *, done_before: int = 0, requests_before: int = 0):
    """An `on_batch` that reports inputs done so far out of `total` (plus
    what earlier passes of the same run completed)."""
    if progress_callback is None:
        return None
    return lambda start, batch_results, count: progress_callback(
        done_before + min(start + len(batch_results), total),
        done_before + total if done_before else total,
        requests_before + count,
    )


def generate_sample_matrix(
    *,
    prompts: list[str],
    model_client,
    max_tokens: int,
    generation_settings: dict[str, Any] | None,
    num_samples: int,
    request_batch_size: int,
    error_prefix: str,
    progress_callback: Callable[[int, int, int], None] | None,
    conversations: list[list[dict[str, str]]] | None = None,
) -> tuple[list[list[GeneratedTextResult]], int]:
    """One row of `num_samples` results per prompt. With `conversations`, each
    request is that message list (system + user) instead of a bare prompt."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    inputs = conversations if conversations is not None else prompts
    generate = model_client.generate_texts if conversations is None else model_client.generate_conversations
    settings = dict(generation_settings or {})
    matrix = [
        [GeneratedTextResult(error="The sample was not generated.") for _ in range(num_samples)]
        for _ in inputs
    ]
    messages = BatchMessages(
        failed=f"{error_prefix} failed: ",
        mismatch=f"{error_prefix} returned a mismatched number of results.",
        not_generated="The sample was not generated.",
    )
    count = len(inputs)
    completed_units = 0
    total_units = count * num_samples

    def on_batch(start: int, batch_results: list, request_count: int) -> None:
        nonlocal completed_units
        for offset, generated_result in enumerate(batch_results):
            sample_index, prompt_index = divmod(start + offset, count)
            matrix[prompt_index][sample_index] = generated_result
        completed_units += len(batch_results)
        if progress_callback is not None:
            progress_callback(completed_units, total_units, request_count)

    _, request_count = generate_batches(
        [item for _ in range(num_samples) for item in inputs],
        lambda batch: generate(batch, max_tokens=max_tokens, **settings),
        request_batch_size=request_batch_size,
        messages=messages,
        on_batch=on_batch,
    )
    return matrix, request_count
