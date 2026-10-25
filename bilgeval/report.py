"""The final report: what one run contributes, and the totals over runs.

`run_benchmark` evaluates each run and hands the evaluator's report here:
`summarize_run` turns a binary report's headline into all-examples accuracy
with a bootstrap interval and subgroup metrics, `dataset_report_entry`
builds the run's entry of `final_report["datasets"]`, `accumulate_totals`
adds its counts, and `finalize_report` computes the macro accuracies, the
replay summary and the coverage gate. Key order is the report contract
(`bilgeval.report/v4`); the literals below define it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scorer_table import dataset_type
from .statistics import (
    bootstrap_accuracy_interval,
    counted_correctness,
    extract_sample_outcomes,
    outcome_fingerprint,
    scored_correctness,
    subgroup_metrics as build_subgroup_metrics,
)


REPORT_SCHEMA_VERSION = "bilgeval.report/v4"
MULTI_MODEL_REPORT_SCHEMA_VERSION = "bilgeval.multi-model-report/v1"
DEFAULT_MAX_REQUEST_FAILURE_RATE = 0.01


def metric_family(config) -> str:
    task_type = str(getattr(config, "task_type", "") or "")
    try:
        return dataset_type(task_type).family
    except ValueError:
        return task_type or "other"


def new_final_report(
    *,
    targets: list[str],
    output_path: Path,
    manifest_path: Path,
    model: str,
    judge_model: str | None,
    verifier_model: str | None,
    judge_disable_thinking: bool,
    judge_mtbench_guided_rating: bool,
    verifier_disable_thinking: bool,
    verifier_json_schema: bool,
    candidate_disable_thinking: bool,
    candidate_extra_body: dict[str, Any] | None,
    request_timeout: float,
    stream_generation: bool,
    generation_max_tokens: int | None,
    request_batch_size: int,
    generation_concurrency: int,
    judge_concurrency: int,
    verifier_concurrency: int,
    output_level: str,
    preflight_enabled: bool,
    external_responses_file: str | None,
    replay_responses_file: str | None,
    minimum_coverage: float | None,
    max_request_failure_rate: float | None,
    max_examples: int | None,
    bos_mode: str,
    bos_token: str,
    effective_add_bos_token: bool,
    bos_diagnostics: dict[str, Any],
    logprob_api: str,
    logprob_transport: str | None,
    generation_api: str,
    code_execution: dict[str, Any],
    bootstrap_resamples: int,
    confidence_level: float,
    statistics_seed: str,
    generation_client: str = "sync",
) -> dict[str, Any]:
    """The report before any run: settings first, then the empty totals."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "targets": targets,
        "report_dir": str(output_path.resolve()),
        "model": model,
        "judge_model": judge_model,
        "verifier_model": verifier_model,
        "judge_disable_thinking": judge_disable_thinking,
        "judge_mtbench_guided_rating": judge_mtbench_guided_rating,
        "verifier_disable_thinking": verifier_disable_thinking,
        "verifier_json_schema": verifier_json_schema,
        "candidate_disable_thinking": candidate_disable_thinking,
        "candidate_extra_body": candidate_extra_body,
        "request_timeout_seconds": request_timeout,
        "stream_generation": stream_generation,
        "generation_max_tokens_override": generation_max_tokens,
        "request_batch_size": request_batch_size,
        "generation_client": generation_client,
        "generation_concurrency": generation_concurrency,
        "judge_concurrency": judge_concurrency,
        "verifier_concurrency": verifier_concurrency,
        "output_level": output_level,
        "preflight_enabled": preflight_enabled,
        "external_responses_file": external_responses_file,
        "replay_responses_file": replay_responses_file,
        "minimum_coverage": minimum_coverage,
        "max_request_failure_rate": max_request_failure_rate,
        "max_examples_override": max_examples,
        "bos_mode": bos_mode,
        "bos_token": bos_token,
        "effective_add_bos_token": effective_add_bos_token,
        "bos_diagnostics": bos_diagnostics,
        "logprob_api": logprob_api,
        "logprob_transport": logprob_transport,
        "generation_api": generation_api,
        "code_execution": code_execution,
        "statistics": {
            "schema_version": "bilgeval.statistics/v1",
            "bootstrap_resamples": bootstrap_resamples,
            "confidence_level": confidence_level,
            "seed": statistics_seed,
            "accuracy_denominator": "all_examples",
            "failure_policy": "count_as_incorrect",
        },
        "manifest_file": str(manifest_path.resolve()),
        "datasets": {},
        "macro_accuracy": None,
        "macro_accuracy_by_family": {},
        "total_candidate_scores": 0,
        "total_model_scores": 0,
        "total_requests": 0,
        "total_failed_examples": 0,
        "total_scored_examples": 0,
        "primary_metrics": {},
        "artifacts": {"responses": {}, "api_traces": None},
    }


@dataclass
class RunSummary:
    """An evaluator's report with its headline settled."""

    report: dict[str, Any]
    is_binary: bool
    confidence_interval: dict[str, Any] | None = None
    subgroup_report: dict[str, Any] | None = None
    sample_outcomes: list[dict[str, Any]] | None = None


def summarize_run(
    report: dict[str, Any],
    *,
    run_name: str,
    examples: list,
    subgroup_fields: list[str],
    seed_context: str,
    bootstrap_resamples: int,
    confidence_level: float,
    statistics_seed: str,
) -> RunSummary:
    """A binary report's headline becomes accuracy over EVERY selected example
    (a truncated, empty or unverifiable response is a wrong answer, as in
    lm-eval, simple-evals and the original harnesses), with its bootstrap
    interval and subgroup metrics; a continuous report is kept as it is."""
    if not ("accuracy" in report and "correct" in report):
        return RunSummary(report, False, None, {}, None)
    sample_outcomes = extract_sample_outcomes(report["results"])
    scored_only = scored_correctness(sample_outcomes)
    if (
        len(scored_only) != report["num_scored_examples"]
        or sum(scored_only) != report["correct"]
    ):
        raise RuntimeError(
            f"Result denominator mismatch for {run_name}: "
            "evaluator summary and sample outcomes disagree."
        )
    correctness = counted_correctness(sample_outcomes)
    report["accuracy_on_scored"] = report["accuracy"]
    report["accuracy"] = (
        report["correct"] / report["num_examples"] if report["num_examples"] else 0.0
    )
    confidence_interval = bootstrap_accuracy_interval(
        correctness,
        resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=statistics_seed,
        seed_context=seed_context,
    )
    subgroup_report = build_subgroup_metrics(
        examples,
        report["results"],
        subgroup_fields,
        resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=statistics_seed,
        seed_context=seed_context,
    )
    return RunSummary(report, True, confidence_interval, subgroup_report, sample_outcomes)


CONTINUOUS_DETAIL_KEYS = (
    "exact_match_accuracy",
    "num_exact_match",
    "mean_f1_on_scored",
    "first_sample_accuracy",
    "num_samples_total",
    "num_samples_failed",
)
OPTIONAL_METRIC_KEYS = (
    "mean_precision",
    "mean_recall",
    "mean_f1",
    "ifeval_metrics",
    "ifeval_instruction_scores",
    "code_execution",
    "livecodebench_metrics",
    "safety_metrics",
    "document_metrics",
    "function_call_metrics",
    "mtbench_metrics",
    "graphwalks_metrics",
    "td_metrics",
    "td_rows",
)


def dataset_report_entry(
    run,
    config,
    summary: RunSummary,
    manifest_run: dict[str, Any],
    *,
    num_few_shot_examples: int,
    output_level: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The run's entry of final_report["datasets"], and its primary metric
    when the report is continuous."""
    report = summary.report
    entry = {
        "dataset_name": run.dataset_name,
        "benchmark_name": run.dataset_name,
        "task_name": run.task_name,
        "task_version": run.task_spec.version,
        "requested_target": run.requested_target,
        "suite_name": run.suite_name,
        "suite_version": run.suite_version,
        "evaluation_fingerprint": manifest_run["evaluation_fingerprint"],
        "comparison_group_fingerprint": manifest_run["comparison_group_fingerprint"],
        "fingerprint_version": manifest_run["fingerprint_version"],
        "max_examples": run.max_examples,
        "sample_seed": run.sample_seed,
        "eval_mode": config.eval_mode,
        "generation_prompt": config.generation_prompt,
        "formatter_version": config.formatter_version,
        "generation_max_tokens": config.generation_max_tokens,
        "generation_temperature": config.generation_temperature,
        "generation_top_p": config.generation_top_p,
        "generation_top_k": config.generation_top_k,
        "generation_stop": list(config.generation_stop),
        "generation_num_samples": config.generation_num_samples,
        "generation_pass_at_k": list(config.generation_pass_at_k),
        "generation_aggregation": config.generation_aggregation,
        "judge_prompt": config.judge_prompt,
        "answer_verification": config.answer_verification,
        "append": config.append,
        "score_target": config.score_target,
        "normalization": config.normalization,
        "pmi_reference_prompt": config.pmi_reference_prompt,
        "length_normalization": config.length_normalization,
        "few_shot_count": config.few_shot_count,
        "few_shot_seed": config.few_shot_seed,
        "request_type": config.request_type,
        "generation_api": config.generation_api,
        "subgroup_fields": list(config.subgroup_fields),
        "few_shot_dataset_name": config.few_shot_dataset_name,
        "few_shot_path": str(config.few_shot_path) if config.few_shot_path is not None else None,
        "num_few_shot_examples": num_few_shot_examples,
        "num_examples": report["num_examples"],
        "num_scored_examples": report["num_scored_examples"],
        "num_failed_examples": report["num_failed_examples"],
        "coverage": report["coverage"],
        "subgroup_metrics": summary.subgroup_report,
        "num_candidate_scores": report["num_candidate_scores"],
        "num_model_scores": report.get("num_model_scores", report["num_candidate_scores"]),
        "num_requests": report["num_requests"],
    }
    primary_metric = None
    if summary.is_binary:
        entry.update(
            {
                "correct": report["correct"],
                "accuracy": report["accuracy"],
                "accuracy_denominator": "all_examples",
                "failure_policy": "count_as_incorrect",
                "accuracy_on_scored": report["accuracy_on_scored"],
                "num_unscored_examples": (
                    report["num_examples"] - report["num_scored_examples"]
                    if report.get("num_examples") is not None
                    else None
                ),
                "accuracy_confidence_interval": summary.confidence_interval,
                "sample_outcome_fingerprint": outcome_fingerprint(summary.sample_outcomes),
            }
        )
    else:
        primary_metric = {
            "name": report["metric_name"],
            "value": report["score"],
            "scale": report["score_scale"],
        }
        entry.update(
            {
                "primary_metric": primary_metric,
                "score": report["score"],
                "score_on_scored": report.get("score_on_scored"),
                "score_denominator": "all_examples",
                "failure_policy": report.get(
                    "failure_policy",
                    "every example counts; an unscored example takes the lowest score",
                ),
            }
        )
        for key in CONTINUOUS_DETAIL_KEYS:
            if key in report:
                entry[key] = report[key]
    if "multi_sample_metrics" in report:
        entry["multi_sample_metrics"] = report["multi_sample_metrics"]
        entry["multi_sample_failure_policy"] = (
            "failed or unparseable generation attempts count as unsuccessful attempts"
        )
    if "answer_verification" in report:
        entry["answer_verification_summary"] = report["answer_verification"]
        entry["num_generation_requests"] = report["num_generation_requests"]
        entry["num_verifier_requests"] = report["num_verifier_requests"]
    for key in OPTIONAL_METRIC_KEYS:
        if key in report:
            entry[key] = report[key]
    if output_level != "summary" and "analysis" in report:
        entry["analysis"] = report["analysis"]
    if "td_metrics" in report:
        entry["score_denominator"] = "length_eligible_questions"
        entry["model_coverage"] = report["td_metrics"]["model_coverage"]
        entry["ci"] = None
    return entry, primary_metric


def accumulate_totals(final_report: dict[str, Any], report: dict[str, Any]) -> None:
    final_report["total_candidate_scores"] += report["num_candidate_scores"]
    final_report["total_model_scores"] += report.get(
        "num_model_scores", report["num_candidate_scores"]
    )
    final_report["total_requests"] += report["num_requests"]
    final_report["total_failed_examples"] += report["num_failed_examples"]
    final_report["total_scored_examples"] += report["num_scored_examples"]


def request_counts(
    clients: dict[str, Any],
    before: dict[str, tuple[int, int]],
) -> dict[str, dict[str, int]]:
    """Per endpoint role, the items one run requested and how many it lost,
    from the clients' tallies now and at the start of the run."""
    counts = {}
    for role, client in clients.items():
        sent, failed = client.request_tally.snapshot()
        sent -= before[role][0]
        failed -= before[role][1]
        if sent:
            counts[role] = {"sent": sent, "failed": failed}
    return counts


def finalize_report(
    final_report: dict[str, Any],
    *,
    accuracies: list[float],
    family_accuracies: dict[str, list[float]],
    replay_client,
    replay_responses_file: str | None,
    minimum_coverage: float | None,
    max_request_failure_rate: float | None,
) -> list[str]:
    """Macro accuracies, the replay summary and the gates; returns why each
    failed gate failed."""
    final_report["macro_accuracy_by_family"] = {
        family: {"macro_accuracy": sum(values) / len(values), "num_runs": len(values)}
        for family, values in sorted(family_accuracies.items())
    }
    if len(family_accuracies) == 1:
        final_report["macro_accuracy"] = sum(accuracies) / len(accuracies)

    if replay_client is not None:
        if replay_client.remaining:
            raise ValueError(
                f"Replay artifact has {replay_client.remaining} unused candidate responses; "
                "artifact and current task/config do not match."
            )
        final_report["artifact_replay"] = {
            "source": replay_responses_file,
            "num_replayed_candidates": replay_client.num_replayed,
            "num_live_fallback_candidates": replay_client.num_fallback,
        }

    coverage_failures = []
    if minimum_coverage is not None:
        coverage_failures = [
            {
                "run_name": run_name,
                "coverage": dataset_report["coverage"],
            }
            for run_name, dataset_report in final_report["datasets"].items()
            if dataset_report["coverage"] < minimum_coverage
        ]
    final_report["coverage_gate"] = {
        "minimum": minimum_coverage,
        "passed": not coverage_failures,
        "failures": coverage_failures,
    }

    request_failures = []
    if max_request_failure_rate is not None:
        request_failures = [
            {"run_name": run_name, "role": role, **counts, "rate": counts["failed"] / counts["sent"]}
            for run_name, dataset_report in final_report["datasets"].items()
            for role, counts in dataset_report.get("requests", {}).items()
            if counts["failed"] / counts["sent"] > max_request_failure_rate
        ]
    final_report["request_gate"] = {
        "maximum_failure_rate": max_request_failure_rate,
        "passed": not request_failures,
        "failures": request_failures,
    }

    messages = []
    if coverage_failures:
        details = ", ".join(f"{item['run_name']}={item['coverage']:.4f}" for item in coverage_failures)
        messages.append(f"Coverage gate failed (minimum={minimum_coverage:.4f}): {details}.")
    if request_failures:
        details = ", ".join(
            f"{item['run_name']} {item['role']} lost {item['failed']} of {item['sent']}"
            for item in request_failures
        )
        messages.append(
            f"Request gate failed (maximum {max_request_failure_rate:.2%} lost to endpoint errors "
            f"after retries): {details}."
        )
    return messages
