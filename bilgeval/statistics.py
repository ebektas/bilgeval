import hashlib
from itertools import combinations
import json
import math
from typing import Any

import numpy as np

from .manifest import stable_json_sha256


STATISTICS_SCHEMA_VERSION = "bilgeval.statistics/v1"


def pass_at_k_estimate(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if num_correct < 0 or num_correct > num_samples:
        raise ValueError("num_correct must be between 0 and num_samples.")
    if k < 1 or k > num_samples:
        raise ValueError("k must be between 1 and num_samples.")
    num_incorrect = num_samples - num_correct
    if num_incorrect < k:
        return 1.0
    return 1.0 - (math.comb(num_incorrect, k) / math.comb(num_samples, k))


def validate_statistics_config(resamples: int, confidence_level: float) -> None:
    if resamples < 1:
        raise ValueError("bootstrap_resamples must be at least 1.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")


def derived_seed(seed: str, *parts: str) -> int:
    material = "\0".join((seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def percentile_interval(
    samples: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        samples,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return float(lower), float(upper)


def bootstrap_accuracy_interval(
    correctness: list[bool],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: str = "bilgeval-statistics-v1",
    seed_context: str = "accuracy",
) -> dict[str, Any] | None:
    validate_statistics_config(resamples, confidence_level)
    if not correctness:
        return None

    num_examples = len(correctness)
    num_correct = sum(correctness)
    accuracy = num_correct / num_examples
    rng_seed = derived_seed(seed, seed_context)
    rng = np.random.default_rng(rng_seed)

    bootstrap_correct = rng.binomial(num_examples, accuracy, size=resamples)
    samples = bootstrap_correct.astype(np.float64) / num_examples
    lower, upper = percentile_interval(samples, confidence_level)
    return {
        "method": "nonparametric_percentile_bootstrap",
        "metric": "accuracy",
        "confidence_level": confidence_level,
        "lower": lower,
        "upper": upper,
        "point_estimate": accuracy,
        "num_scored_examples": num_examples,
        "num_correct": num_correct,
        "resamples": resamples,
        "seed": seed,
        "derived_seed_sha256": stable_json_sha256(
            {"seed": seed, "context": seed_context}
        ),
    }


def extract_sample_outcomes(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outcomes = []
    seen_ids: set[str] = set()
    for position, result in enumerate(results):
        index = result.get("idx", position)
        raw_id = str(result.get("id", "")).strip()
        sample_id = raw_id or f"__index__:{index}"
        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate sample id {sample_id!r} prevents trustworthy paired comparison."
            )
        seen_ids.add(sample_id)
        failed = bool(result.get("failed", False))
        correct = result.get("correct")
        scored = not failed and isinstance(correct, bool)
        outcomes.append(
            {
                "sample_id": sample_id,
                "index": index,
                "scored": scored,
                "correct": bool(correct) if scored else None,
            }
        )
    return outcomes


def outcome_fingerprint(outcomes: list[dict[str, Any]]) -> str:
    return stable_json_sha256(outcomes)


def scored_correctness(outcomes: list[dict[str, Any]]) -> list[bool]:
    """Diagnostic view: correctness of the examples that produced a verdict."""
    return [outcome["correct"] for outcome in outcomes if outcome["scored"]]


def counted_correctness(outcomes: list[dict[str, Any]]) -> list[bool]:
    """Headline view: every selected example counts, an unscored one as incorrect."""
    return [bool(outcome["correct"]) if outcome["scored"] else False for outcome in outcomes]


def resolve_subgroup_value(raw: dict[str, Any], field: str) -> tuple[bool, Any]:
    current: Any = raw
    found = True
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            found = False
            break
        current = current[part]
    if not found and "." not in field:
        metadata = raw.get("metadata")
        if isinstance(metadata, dict) and field in metadata:
            current = metadata[field]
            found = True
    if not found or current is None or current == "":
        return False, None
    return True, current


def subgroup_metrics(
    examples: list[Any],
    results: list[dict[str, Any]],
    subgroup_fields: tuple[str, ...] | list[str],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: str = "bilgeval-statistics-v1",
    seed_context: str = "subgroups",
) -> dict[str, Any]:
    validate_statistics_config(resamples, confidence_level)
    result_by_index = {int(result.get("idx", index)): result for index, result in enumerate(results)}
    output: dict[str, Any] = {}
    for field in subgroup_fields:
        grouped: dict[str, dict[str, Any]] = {}
        for index, example in enumerate(examples):
            result = result_by_index.get(index)
            if result is None:
                continue
            is_correct = (
                bool(result["correct"])
                if not bool(result.get("failed", False)) and isinstance(result.get("correct"), bool)
                else False
            )
            raw = getattr(example, "raw", {})
            found, value = resolve_subgroup_value(raw if isinstance(raw, dict) else {}, field)
            key = (
                "value:" + json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                if found
                else "missing:"
            )
            group = grouped.setdefault(
                key,
                {
                    "value": value if found else None,
                    "missing": not found,
                    "correctness": [],
                },
            )
            group["correctness"].append(is_correct)

        groups = []
        for key in sorted(grouped):
            group = grouped[key]
            correctness = group.pop("correctness")
            num_scored = len(correctness)
            num_correct = sum(correctness)
            groups.append(
                {
                    **group,
                    "num_scored_examples": num_scored,
                    "num_correct": num_correct,
                    "accuracy": num_correct / num_scored,
                    "accuracy_confidence_interval": bootstrap_accuracy_interval(
                        correctness,
                        resamples=resamples,
                        confidence_level=confidence_level,
                        seed=seed,
                        seed_context=f"{seed_context}\0{field}\0{key}",
                    ),
                }
            )

        all_groups = [group for group in groups if group["num_scored_examples"]]
        declared_groups = [group for group in all_groups if not group["missing"]]
        total_scored = sum(group["num_scored_examples"] for group in all_groups)
        total_correct = sum(group["num_correct"] for group in all_groups)
        output[field] = {
            "num_scored_examples": total_scored,
            "num_groups": len(declared_groups),
            "num_missing_examples": sum(
                group["num_scored_examples"] for group in all_groups if group["missing"]
            ),
            "micro_accuracy": total_correct / total_scored if total_scored else None,
            "macro_accuracy": (
                sum(group["accuracy"] for group in declared_groups) / len(declared_groups)
                if declared_groups
                else None
            ),
            "groups": groups,
        }
    return output


def paired_accuracy_delta(
    model_a_outcomes: list[dict[str, Any]],
    model_b_outcomes: list[dict[str, Any]],
    *,
    model_a: str,
    model_b: str,
    dataset_name: str,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: str = "bilgeval-statistics-v1",
) -> dict[str, Any]:
    validate_statistics_config(resamples, confidence_level)
    by_id_a = {outcome["sample_id"]: outcome for outcome in model_a_outcomes}
    by_id_b = {outcome["sample_id"]: outcome for outcome in model_b_outcomes}
    if len(by_id_a) != len(model_a_outcomes) or len(by_id_b) != len(model_b_outcomes):
        raise ValueError("Duplicate sample IDs prevent paired comparison.")

    ids_a = set(by_id_a)
    ids_b = set(by_id_b)
    base = {
        "model_a": model_a,
        "model_b": model_b,
        "delta_definition": "accuracy(model_b) - accuracy(model_a)",
        "sample_sets_match": ids_a == ids_b,
        "num_selected_model_a": len(ids_a),
        "num_selected_model_b": len(ids_b),
        "num_only_model_a": len(ids_a - ids_b),
        "num_only_model_b": len(ids_b - ids_a),
    }
    if ids_a != ids_b:
        return {
            **base,
            "status": "incompatible_sample_sets",
            "error": "Paired deltas require identical selected sample IDs.",
        }

    ordered_ids = sorted(ids_a)

    def counted(outcome: dict[str, Any]) -> dict[str, Any]:
        return {**outcome, "correct": bool(outcome["correct"]) if outcome["scored"] else False}

    paired = [
        (counted(by_id_a[sample_id]), counted(by_id_b[sample_id]))
        for sample_id in ordered_ids
    ]
    num_model_a_failed = sum(not by_id_a[sample_id]["scored"] for sample_id in ordered_ids)
    num_model_b_failed = sum(not by_id_b[sample_id]["scored"] for sample_id in ordered_ids)
    num_paired = len(paired)
    num_both_scored = sum(
        by_id_a[sample_id]["scored"] and by_id_b[sample_id]["scored"]
        for sample_id in ordered_ids
    )
    if not paired:
        return {
            **base,
            "status": "no_paired_scores",
            "num_paired_scored": 0,
            "num_model_a_unscored": num_model_a_failed,
            "num_model_b_unscored": num_model_b_failed,
        }

    model_a_correct = sum(item_a["correct"] for item_a, _ in paired)
    model_b_correct = sum(item_b["correct"] for _, item_b in paired)
    model_a_wins = sum(
        item_a["correct"] and not item_b["correct"]
        for item_a, item_b in paired
    )
    model_b_wins = sum(
        item_b["correct"] and not item_a["correct"]
        for item_a, item_b in paired
    )
    ties = num_paired - model_a_wins - model_b_wins
    delta = (model_b_correct - model_a_correct) / num_paired

    rng_seed = derived_seed(seed, dataset_name, model_a, model_b, "paired_accuracy_delta")
    rng = np.random.default_rng(rng_seed)
    probabilities = np.array([model_a_wins, ties, model_b_wins], dtype=np.float64) / num_paired
    bootstrap_counts = rng.multinomial(num_paired, probabilities, size=resamples)
    bootstrap_deltas = (
        bootstrap_counts[:, 2] - bootstrap_counts[:, 0]
    ).astype(np.float64) / num_paired
    lower, upper = percentile_interval(bootstrap_deltas, confidence_level)

    return {
        **base,
        "status": "ok",
        "num_paired_examples": num_paired,
        "num_paired_scored": num_both_scored,
        "num_model_a_unscored": num_model_a_failed,
        "num_model_b_unscored": num_model_b_failed,
        "model_a_accuracy_on_paired": model_a_correct / num_paired,
        "model_b_accuracy_on_paired": model_b_correct / num_paired,
        "delta_model_b_minus_model_a": delta,
        "model_a_wins": model_a_wins,
        "model_b_wins": model_b_wins,
        "ties": ties,
        "confidence_interval": {
            "method": "paired_nonparametric_percentile_bootstrap",
            "confidence_level": confidence_level,
            "lower": lower,
            "upper": upper,
            "resamples": resamples,
            "seed": seed,
            "derived_seed_sha256": stable_json_sha256(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": "paired_accuracy_delta",
                }
            ),
        },
    }


def build_multi_model_comparisons(
    reports: list[dict[str, Any]],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: str = "bilgeval-statistics-v1",
) -> dict[str, Any]:
    validate_statistics_config(resamples, confidence_level)
    comparisons: dict[str, Any] = {}
    for report_a, report_b in combinations(reports, 2):
        model_a = report_a["model"]
        model_b = report_b["model"]
        outcomes_a = report_a.get("_sample_outcomes", {})
        outcomes_b = report_b.get("_sample_outcomes", {})
        common_runs = sorted(set(outcomes_a) & set(outcomes_b))
        pair_key = f"{model_a}__vs__{model_b}"
        comparisons[pair_key] = {
            "model_a": model_a,
            "model_b": model_b,
            "datasets": {
                run_name: paired_accuracy_delta(
                    outcomes_a[run_name],
                    outcomes_b[run_name],
                    model_a=model_a,
                    model_b=model_b,
                    dataset_name=run_name,
                    resamples=resamples,
                    confidence_level=confidence_level,
                    seed=seed,
                )
                for run_name in common_runs
            },
            "num_only_model_a_datasets": len(set(outcomes_a) - set(outcomes_b)),
            "num_only_model_b_datasets": len(set(outcomes_b) - set(outcomes_a)),
        }
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "method": "paired_nonparametric_percentile_bootstrap",
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "comparisons": comparisons,
    }
