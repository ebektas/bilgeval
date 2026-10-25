import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from .answer_extraction import ANSWER_PARSER_VERSION
from . import ifeval_verifier, instruction_following
from .scorer_table import ANSWER_PARSED_EVAL_MODES, LOGPROB_EVAL_MODE, ONERULER_TR_RULES_VERSION
from .generation_prompts import FEW_SHOT_RENDER_VERSION, prompt_template_source
from .judge_prompts import JUDGE_PROMPTS
from .registry import PACKAGE_ROOT


MANIFEST_SCHEMA_VERSION = "bilgeval.run-manifest/v1"
FINGERPRINT_VERSION = 2


def sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_file_cached(path, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=128)
def _sha256_file_cached(path: Path, mtime_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(serialized).hexdigest()


def few_shot_selection_groups(prepared) -> dict[str, list[str]]:
    match_field = prepared.config.few_shot_match_field
    if not match_field:
        return {"all": selected_example_ids(prepared.few_shot_examples)}

    groups: dict[str, list[str]] = {}
    for evaluation_example, selected in zip(
        prepared.examples,
        prepared.few_shot_examples_per_example,
    ):
        value = getattr(evaluation_example, "raw", {})
        for part in match_field.split("."):
            value = value[part]
        groups[str(value)] = selected_example_ids(selected)
    return dict(sorted(groups.items()))


def _git_revision_from_files(repository_root: Path) -> str | None:
    git_dir = repository_root / ".git"
    if git_dir.is_file():
        value = git_dir.read_text(encoding="utf-8").strip()
        if not value.startswith("gitdir:"):
            return None
        git_dir = (repository_root / value.split(":", 1)[1].strip()).resolve()
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head or None
    reference = head.split(":", 1)[1].strip()
    reference_path = git_dir / reference
    if reference_path.is_file():
        return reference_path.read_text(encoding="utf-8").strip() or None
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                revision, name = line.split(" ", 1)
                if name == reference:
                    return revision
    return None


def git_state(repository_root: Path) -> dict[str, Any]:
    if os.name == "nt":
        return {
            "revision": _git_revision_from_files(repository_root),
            "dirty": None,
            "tracked_diff_hash": None,
            "untracked_paths": [],
            "untracked_content_hash": stable_json_sha256([]),
        }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        untracked_output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            "revision": None,
            "dirty": None,
            "tracked_diff_hash": None,
            "untracked_paths": [],
            "untracked_content_hash": stable_json_sha256([]),
        }
    untracked_paths = sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in untracked_output.split(b"\0")
        if item
    )
    untracked_fingerprints = []
    for relative_path in untracked_paths:
        path = repository_root / relative_path
        if path.is_file():
            untracked_fingerprints.append(
                {"path": relative_path, "content_hash": sha256_file(path)}
            )
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "tracked_diff_hash": "sha256:" + hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_paths": untracked_paths,
        "untracked_content_hash": stable_json_sha256(untracked_fingerprints),
    }


def content_hash_for_path(path, declared_hash: str | None, label: str) -> str | None:
    if isinstance(path, Path) and path.is_file():
        actual_hash = sha256_file(path)
        if declared_hash and actual_hash != declared_hash:
            raise ValueError(
                f"Dataset {label} content hash mismatch: "
                f"expected {declared_hash}, got {actual_hash}."
            )
        return actual_hash
    return declared_hash


def dataset_content_hash(config) -> str | None:
    return content_hash_for_path(config.path, config.content_hash, config.dataset_name)


def manifest_uri(path) -> str | None:
    """A dataset's uri as the manifest records it: relative to the bilgeval
    package for bundled data (the same identity in every checkout), as given
    otherwise."""
    if path is None:
        return None
    if isinstance(path, Path):
        try:
            return path.resolve().relative_to(PACKAGE_ROOT).as_posix()
        except ValueError:
            return str(path)
    return str(path)


def judge_prompt_hash(prompt_name: str | None) -> str | None:
    """The judge prompt's content, not its Python representation."""
    config = JUDGE_PROMPTS.get(prompt_name) if prompt_name else None
    if config is None:
        return None
    return stable_json_sha256({"template": config.template, "output_type": config.output_type})


def oneruler_rules(config) -> dict[str, str]:
    """The Turkish OneRuler rules' revision. Only these tasks carry it, so
    every other task keeps its fingerprint."""
    if config.eval_mode == "oneruler_tr":
        return {"oneruler_rules_version": ONERULER_TR_RULES_VERSION}
    return {}


def instruction_verifier(config) -> dict[str, str]:
    """The checker revision of an instruction-following task. Only these
    tasks carry it, so every other task keeps its fingerprint."""
    package = instruction_following.PROTOCOLS.get(config.schema_version)
    if package is not None:
        return {"instruction_verifier": instruction_following.SOURCES[package]}
    if config.eval_mode == "ifeval":
        return {"instruction_verifier": f"bilgeval ifeval_verifier {ifeval_verifier.VERSION}"}
    return {}


def selected_example_ids(examples: list) -> list[str]:
    return [str(getattr(example, "id", "")) for example in examples]


def leaderboard_fingerprints(run_manifest: dict[str, Any]) -> tuple[str, str]:
    evaluation_identity = {
        "fingerprint_version": run_manifest.get("fingerprint_version", 1),
        "dataset": run_manifest["dataset"],
        "formatter": run_manifest["formatter"],
        "request_type": run_manifest["request_type"],
        "scorer": run_manifest["scorer"],
        "filters": run_manifest["filters"],
        "generation": run_manifest["generation"],
        "sampling": run_manifest["sampling"],
        "few_shot": run_manifest["few_shot"],
        "subgroup_fields": run_manifest["subgroup_fields"],
    }
    if "td_identity" in run_manifest:
        evaluation_identity["td_identity"] = run_manifest["td_identity"]
    comparison_identity = {
        key: value
        for key, value in evaluation_identity.items()
        if key != "scorer"
    }
    return (
        stable_json_sha256(evaluation_identity),
        stable_json_sha256(comparison_identity),
    )


def build_run_manifest(
    *,
    targets: list[str],
    model: str,
    runtime: dict[str, Any],
    prepared_runs: list,
    registry_context: dict[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    runs = []
    for prepared in prepared_runs:
        run = prepared.run
        config = prepared.config
        prompt_template = prompt_template_source(config.generation_prompt)
        selected_ids = selected_example_ids(prepared.examples)
        few_shot_ids = selected_example_ids(prepared.few_shot_examples)
        few_shot_ids_by_match_value = few_shot_selection_groups(prepared)
        dataset_hash = dataset_content_hash(config)
        prompt_hash = (
            stable_json_sha256(prompt_template)
            if prompt_template is not None
            else None
        )
        few_shot_hash = content_hash_for_path(
            config.few_shot_path,
            config.few_shot_content_hash,
            config.few_shot_dataset_name or f"{config.dataset_name} few-shot",
        )
        verifier_runtime = runtime.get("answer_verifier") or {}
        candidate_generation = (
            {
                "disable_thinking": bool(runtime.get("candidate_disable_thinking")),
                "extra_body": runtime.get("candidate_extra_body"),
            }
            if config.eval_mode != LOGPROB_EVAL_MODE
            else None
        )
        generation_settings = {
            "candidate": candidate_generation,
            "max_tokens": config.generation_max_tokens,
            "temperature": config.generation_temperature,
            "top_p": config.generation_top_p,
            "top_k": config.generation_top_k,
            "seed": None,
            "stop": list(config.generation_stop),
            "num_samples": config.generation_num_samples,
            "pass_at_k": list(config.generation_pass_at_k),
            "aggregation": config.generation_aggregation,
        }
        task_spec_fingerprint = stable_json_sha256(
            {
                "task": {
                    "name": run.task_name,
                    "version": run.task_spec.version,
                },
                "dataset": {
                    "name": config.dataset_name,
                    "version": config.dataset_version,
                    "split": config.dataset_split,
                    "content_hash": dataset_hash,
                },
                "formatter": {
                    "id": config.generation_prompt,
                    "version": config.formatter_version,
                    "template_hash": prompt_hash,
                    "append_choices": config.append,
                },
                "request_type": config.request_type,
                "generation_api": config.generation_api,
                "scorer": {
                    "id": config.scorer_id,
                    "eval_mode": config.eval_mode,
                    "score_target": config.score_target,
                    "normalization": config.normalization,
                    "pmi_reference_prompt": config.pmi_reference_prompt,
                    "length_normalization": config.length_normalization,
                    "answer_verification": config.answer_verification,
                },
                "generation": generation_settings,
                "few_shot": {
                    "count": config.few_shot_count,
                    "dataset": config.few_shot_dataset_name,
                    "seed": config.few_shot_seed,
                    "match_field": config.few_shot_match_field,
                    "content_hash": few_shot_hash,
                    "render_version": FEW_SHOT_RENDER_VERSION if config.few_shot_count else None,
                },
                "subgroup_fields": list(config.subgroup_fields),
            }
        )
        run_manifest = {
            "run_name": run.run_name,
            "requested_target": run.requested_target,
            "fingerprint_version": FINGERPRINT_VERSION,
            "task": {
                "name": run.task_name,
                "version": run.task_spec.version,
                "spec_fingerprint": task_spec_fingerprint,
            },
            "suite": {
                "name": run.suite_name,
                "version": run.suite_version,
            },
            "dataset": {
                "name": config.dataset_name,
                "version": config.dataset_version,
                "schema_version": config.schema_version,
                "uri": manifest_uri(config.path),
                "content_hash": dataset_hash,
                "split": config.dataset_split,
                "language": config.language,
                "declared_row_count": config.row_count,
            },
            "formatter": {
                "id": config.generation_prompt,
                "version": config.formatter_version,
                "template_hash": prompt_hash,
                "append_choices": config.append,
            },
            "request_type": config.request_type,
            "generation_api": config.generation_api,
            "scorer": {
                "id": config.scorer_id,
                "eval_mode": config.eval_mode,
                "score_target": config.score_target,
                "normalization": config.normalization,
                "pmi_reference_prompt": config.pmi_reference_prompt,
                "length_normalization": config.length_normalization,
                "answer_verification": config.answer_verification,
                "answer_parser_version": (
                    ANSWER_PARSER_VERSION
                    if config.eval_mode in ANSWER_PARSED_EVAL_MODES
                    else None
                ),
                **instruction_verifier(config),
                **oneruler_rules(config),
                "judge_prompt": config.judge_prompt,
                "judge_prompt_hash": judge_prompt_hash(config.judge_prompt),
                "verifier": (
                    {
                        "json_schema": bool(verifier_runtime.get("json_schema")),
                        "disable_thinking": bool(verifier_runtime.get("disable_thinking")),
                    }
                    if config.answer_verification not in (None, "regex")
                    else None
                ),
            },
            "filters": [],
            "metrics": [metric.id for metric in run.task_spec.metrics],
            "subgroup_fields": list(config.subgroup_fields),
            "generation": {
                **generation_settings,
                "settings_fingerprint": stable_json_sha256(generation_settings),
            },
            "sampling": {
                "max_examples": run.max_examples,
                "sample_seed": run.sample_seed,
                "selected_example_ids": selected_ids,
                "selected_example_ids_fingerprint": stable_json_sha256(selected_ids),
            },
            "few_shot": {
                "requested_count": config.few_shot_count,
                "seed": config.few_shot_seed,
                "match_field": config.few_shot_match_field,
                "dataset": {
                    "name": config.few_shot_dataset_name,
                    "version": config.few_shot_dataset_version,
                    "schema_version": config.few_shot_schema_version,
                    "uri": manifest_uri(config.few_shot_path),
                    "content_hash": few_shot_hash,
                    "split": config.few_shot_split,
                    "language": config.few_shot_language,
                    "declared_row_count": config.few_shot_row_count,
                },
                "selected_example_ids": few_shot_ids,
                "selected_example_ids_fingerprint": stable_json_sha256(few_shot_ids),
                "selected_example_ids_by_match_value": few_shot_ids_by_match_value,
                "selected_example_ids_by_match_value_fingerprint": stable_json_sha256(
                    few_shot_ids_by_match_value
                ),
            },
        }
        if config.task_type == "typed_decision":
            run_manifest["td_identity"] = {**runtime["td"], "benchmark_version": config.dataset_version, "data_hash": dataset_hash, "suite_seed": run.sample_seed, "sample_strategy": run.task_spec.metadata.get("td_sample_strategy", "stable_hash") if run.max_examples is not None and run.sample_seed is not None else "stable_hash"}
        (
            run_manifest["evaluation_fingerprint"],
            run_manifest["comparison_group_fingerprint"],
        ) = leaderboard_fingerprints(run_manifest)
        run_manifest["resolved_config_hash"] = stable_json_sha256(run_manifest)
        runs.append(run_manifest)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "targets": targets,
        "model": model,
        "runtime": runtime,
        "registry": registry_context,
        "code": git_state(repository_root),
        "runs": runs,
    }
    manifest["manifest_hash"] = stable_json_sha256(manifest)
    return manifest


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
