import hashlib
import json
from typing import Any
import re
import sys
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from .evaluators import (
    BatchRequestError,
    DEFAULT_GENERATED_MC_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    THINKING_DISABLED_EXTRA_BODY,
    CountsRequests,
    LogprobModelClient,
    OpenAICompatibleAnswerVerifierClient,
    OpenAICompatibleJudgeClient,
    OpenAICompatibleTextGenerationClient,
    OpenAICompatibleToolCallClient,
    create_logprob_client,
    model_requires_bos,
    probe_sequence_start,
    mean_sample_report,
    validate_completion_settings,
)
from .td_client import TDClient
from .td_lengths import prepare_runtime
from .registry import DATASETS
from .reader import (
    read_external_responses,
    read_free_response_dataset,
    read_jsonl_dataset,
)
from .preflight import PreflightError, run_preflight_checks
from .scorers import (
    RunContext,
    evaluate_run,
    generation_request_settings,
    load_livecodebench_demonstrations,
    read_examples,
)
from .scorer_table import (
    ANSWER_PARSED_EVAL_MODES,
    EVAL_MODES,
    EVAL_MODE_BY_DATASET_TYPE,
    LOGPROB_EVAL_MODE,
    dataset_type,
    scorer,
)
from .code_execution import CodeExecutor, create_code_executor
from .artifacts import (
    ApiTraceCollector,
    OUTPUT_LEVELS,
    response_artifact_filename,
    write_response_artifact,
)
from .replay import ArtifactReplayTextGenerationClient
from .manifest import build_run_manifest, sha256_file, stable_json_sha256, write_run_manifest
from .registry_bundle import (
    GenerationSpec,
    REGISTRY_SCHEMA_VERSION,
    ModelProfile,
    RegistryBundle,
    TaskSpec,
    install_registry_bundle,
    load_registry_bundle,
    registry_file_hash,
)
from .report import (
    DEFAULT_MAX_REQUEST_FAILURE_RATE,
    accumulate_totals,
    dataset_report_entry,
    finalize_report,
    metric_family,
    new_final_report,
    request_counts,
    summarize_run,
)
from .statistics import validate_statistics_config
from .serialization import profile_serialization, serialization_manifest


@dataclass(frozen=True)
class ResolvedBenchmarkRun:
    run_name: str
    requested_target: str
    task_name: str
    task_spec: TaskSpec
    max_examples: int | None = None
    sample_seed: str | None = None
    suite_name: str | None = None
    suite_version: str | None = None

    @property
    def dataset_name(self) -> str:
        return self.task_spec.dataset


@dataclass(frozen=True)
class EffectiveBenchmarkConfig:
    name: str
    dataset_name: str
    path: Path | str | None
    few_shot_path: Path | str | None
    source: str
    question_key: str
    choices_key: str
    answer_key: str
    task_type: str
    question_only_before_newline: bool
    dataset_version: str
    schema_version: str
    content_hash: str | None
    dataset_split: str | None
    language: str | None
    row_count: int | None
    few_shot_dataset_name: str | None
    few_shot_dataset_version: str | None
    few_shot_schema_version: str | None
    few_shot_content_hash: str | None
    few_shot_split: str | None
    few_shot_language: str | None
    few_shot_row_count: int | None
    generation_prompt: str
    formatter_version: str
    generation_max_tokens: int | None
    generation_temperature: float
    generation_top_p: float | None
    generation_top_k: int | None
    generation_stop: tuple[str, ...]
    generation_num_samples: int
    generation_pass_at_k: tuple[int, ...]
    generation_aggregation: str
    append: bool
    eval_mode: str
    score_target: str
    normalization: str
    pmi_reference_prompt: str | None
    length_normalization: bool
    judge_prompt: str | None
    answer_verification: str
    few_shot_count: int
    few_shot_seed: str | None
    few_shot_match_field: str | None
    request_type: str
    generation_api: str
    subgroup_fields: tuple[str, ...]
    scorer_id: str


@dataclass(frozen=True)
class PreparedRun:
    run: ResolvedBenchmarkRun
    config: EffectiveBenchmarkConfig
    examples: list
    few_shot_examples: list
    few_shot_examples_per_example: list[list]


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_REGISTRY_CONTEXT = {
    "schema_version": "builtin/legacy",
    "registry_version": "builtin",
    "path": None,
    "content_hash": None,
}
ACTIVE_MODEL_PROFILES: dict[str, ModelProfile] = {}
ACTIVE_REGISTRY_BUNDLE: RegistryBundle | None = None


def load_registry_file(registry_file: str | None) -> None:
    global ACTIVE_REGISTRY_BUNDLE
    if registry_file is None:
        registry_path = Path(__file__).resolve().with_name("builtin_registry.json")
        bundle = load_registry_bundle(registry_path)
        install_registry_bundle(bundle, registry_path)
        ACTIVE_REGISTRY_BUNDLE = bundle
        ACTIVE_MODEL_PROFILES.clear()
        ACTIVE_MODEL_PROFILES.update(bundle.model_profiles)
        ACTIVE_REGISTRY_CONTEXT.clear()
        ACTIVE_REGISTRY_CONTEXT.update(
            {
                "schema_version": bundle.schema_version,
                "registry_version": bundle.registry_version,
                "path": str(registry_path),
                "content_hash": registry_file_hash(registry_path),
            }
        )
        return
    registry_path = Path(registry_file).expanduser().resolve()
    try:
        with open(registry_path, encoding="utf-8") as file:
            registry = json.load(file)
    except OSError as exc:
        raise ValueError(f"Could not read registry file {registry_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Registry file {registry_path} is not valid JSON: {exc}") from exc
    if registry.get("schema_version") == REGISTRY_SCHEMA_VERSION:
        bundle = RegistryBundle.model_validate(registry)
        install_registry_bundle(bundle, registry_path)
        ACTIVE_REGISTRY_BUNDLE = bundle
        ACTIVE_MODEL_PROFILES.clear()
        ACTIVE_MODEL_PROFILES.update(bundle.model_profiles)
        ACTIVE_REGISTRY_CONTEXT.clear()
        ACTIVE_REGISTRY_CONTEXT.update(
            {
                "schema_version": bundle.schema_version,
                "registry_version": bundle.registry_version,
                "path": str(registry_path),
                "content_hash": registry_file_hash(registry_path),
            }
        )
        return
    if registry.get("version") == 1:
        raise ValueError(
            "Legacy version=1 registries are not supported; write a bilgeval.registry/v1 "
            "bundle (see docs/data-and-registry-v1.md)."
        )
    raise ValueError(
        f"Unsupported registry version in {registry_path}: "
        f"{registry.get('schema_version') or registry.get('version')}"
    )


load_registry_file(None)


def resolve_model_profile(
    model: str,
    requested_name: str | None = None,
) -> tuple[str | None, ModelProfile | None]:
    if requested_name is not None:
        profile = ACTIVE_MODEL_PROFILES.get(requested_name)
        if profile is None:
            available = ", ".join(sorted(ACTIVE_MODEL_PROFILES)) or "none"
            raise ValueError(
                f"Unknown model profile {requested_name!r}. Available profiles: {available}."
            )
        if re.search(profile.model_pattern, model, flags=re.IGNORECASE) is None:
            raise ValueError(
                f"Model {model!r} does not match profile {requested_name!r} pattern "
                f"{profile.model_pattern!r}."
            )
        return requested_name, profile

    matches = [
        (name, profile)
        for name, profile in ACTIVE_MODEL_PROFILES.items()
        if re.search(profile.model_pattern, model, flags=re.IGNORECASE) is not None
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Model {model!r} matches multiple profiles: "
            + ", ".join(name for name, _ in matches)
            + ". Select one with --model-profile."
        )
    return matches[0] if matches else (None, None)


def resolve_generation_api(
    configs: list[EffectiveBenchmarkConfig],
    requested_api: str | None,
    profile: ModelProfile | None,
) -> str:
    generation_configs = [config for config in configs if config.request_type == "generate"]
    configured_apis = {
        config.generation_api
        for config in generation_configs
        if config.generation_api != "auto"
    }
    if len(configured_apis) > 1:
        raise ValueError(
            "Selected generation benchmarks require conflicting generation APIs: "
            + ", ".join(sorted(configured_apis))
            + ". Run them separately or register them with a common API policy."
        )
    configured_api = next(iter(configured_apis), None)
    if requested_api is not None and configured_api is not None and requested_api != configured_api:
        raise ValueError(
            f"--generation-api={requested_api!r} conflicts with the registered benchmark "
            f"generation_api={configured_api!r}."
        )
    resolved = requested_api or configured_api or (profile.generation_api if profile else None)
    if generation_configs and resolved is None:
        names = ", ".join(config.dataset_name for config in generation_configs)
        raise ValueError(
            "Generation API could not be resolved for model-backed generation tasks "
            f"({names}). Register each benchmark with generation_api='chat' or 'completions', "
            "select a matching model profile, or pass --generation-api explicitly."
        )
    return resolved or "completions"


def available_target_names() -> list[str]:
    if ACTIVE_REGISTRY_BUNDLE is None:
        raise RuntimeError("No TaskSpec registry is active.")
    return [
        "all",
        *ACTIVE_REGISTRY_BUNDLE.tasks.keys(),
        *ACTIVE_REGISTRY_BUNDLE.suites.keys(),
    ]


def build_run_name(
    dataset_name: str,
    requested_target: str,
    configured_run_name: str | None,
    max_examples: int | None,
) -> str:
    del requested_target
    name_parts = [configured_run_name or dataset_name]

    if max_examples is not None:
        name_parts.append(f"max{max_examples}")

    return "__".join(name_parts)


def resolve_target_name(target_name: str) -> list[ResolvedBenchmarkRun]:
    if ACTIVE_REGISTRY_BUNDLE is None:
        raise RuntimeError("No TaskSpec registry is active.")
    if target_name == "all":
        return [
            run
            for task_name in ACTIVE_REGISTRY_BUNDLE.tasks
            for run in resolve_target_name(task_name)
        ]
    task = ACTIVE_REGISTRY_BUNDLE.tasks.get(target_name)
    if task is not None:
        return [
            ResolvedBenchmarkRun(
                run_name=build_run_name(
                    dataset_name=task.dataset,
                    requested_target=target_name,
                    configured_run_name=target_name,
                    max_examples=None,
                ),
                requested_target=target_name,
                task_name=target_name,
                task_spec=task,
            )
        ]
    suite = ACTIVE_REGISTRY_BUNDLE.suites.get(target_name)
    if suite is None:
        raise ValueError(
            f"Unknown task or suite: {target_name}. "
            f"Available: {', '.join(available_target_names())}"
        )
    return [
        ResolvedBenchmarkRun(
            run_name=build_run_name(
                dataset_name=ACTIVE_REGISTRY_BUNDLE.tasks[task_ref.task].dataset,
                requested_target=target_name,
                configured_run_name=task_ref.run_name or task_ref.task,
                max_examples=task_ref.max_examples,
            ),
            requested_target=target_name,
            task_name=task_ref.task,
            task_spec=ACTIVE_REGISTRY_BUNDLE.tasks[task_ref.task],
            max_examples=task_ref.max_examples,
            sample_seed=task_ref.sample_seed,
            suite_name=target_name,
            suite_version=suite.version,
        )
        for task_ref in suite.tasks
    ]


def resolve_benchmark_runs(target_names: list[str]) -> list[ResolvedBenchmarkRun]:
    if not target_names:
        raise ValueError("At least one benchmark target is required.")

    resolved_runs = []
    run_name_counts: dict[str, int] = {}

    for target_name in target_names:
        for run in resolve_target_name(target_name):
            occurrence = run_name_counts.get(run.run_name, 0) + 1
            run_name_counts[run.run_name] = occurrence

            unique_run_name = run.run_name
            if occurrence > 1:
                unique_run_name = f"{run.run_name}__repeat{occurrence}"

            resolved_runs.append(
                ResolvedBenchmarkRun(
                    run_name=unique_run_name,
                    requested_target=run.requested_target,
                    task_name=run.task_name,
                    task_spec=run.task_spec,
                    max_examples=run.max_examples,
                    sample_seed=run.sample_seed,
                    suite_name=run.suite_name,
                    suite_version=run.suite_version,
                )
            )

    return resolved_runs


def apply_max_examples_override(
    selected_runs: list[ResolvedBenchmarkRun],
    max_examples: int | None,
) -> list[ResolvedBenchmarkRun]:
    if max_examples is None:
        return selected_runs
    if max_examples < 1:
        raise ValueError("max_examples must be at least 1.")

    overridden_runs = []
    run_name_counts: dict[str, int] = {}
    for run in selected_runs:
        configured_run_name = re.sub(r"__max\d+$", "", run.run_name)
        run_name = build_run_name(
            dataset_name=run.dataset_name,
            requested_target=run.requested_target,
            configured_run_name=configured_run_name,
            max_examples=max_examples,
        )
        occurrence = run_name_counts.get(run_name, 0) + 1
        run_name_counts[run_name] = occurrence
        if occurrence > 1:
            run_name = f"{run_name}__repeat{occurrence}"
        overridden_runs.append(replace(run, run_name=run_name, max_examples=max_examples))

    return overridden_runs


def apply_generation_overrides(
    selected_runs: list[ResolvedBenchmarkRun],
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    stop: list[str] | None = None,
    num_samples: int | None = None,
    pass_at_k: list[int] | None = None,
    aggregation: str | None = None,
    max_tokens: int | None = None,
) -> list[ResolvedBenchmarkRun]:
    if all(
        value is None
        for value in (
            temperature,
            top_p,
            top_k,
            stop,
            num_samples,
            pass_at_k,
            aggregation,
            max_tokens,
        )
    ):
        return selected_runs
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens override must be at least 1.")
    overridden = []
    for run in selected_runs:
        generation = run.task_spec.generation
        generation = generation.model_copy(
            update={
                "max_tokens": generation.max_tokens if max_tokens is None else max_tokens,
                "temperature": generation.temperature if temperature is None else temperature,
                "top_p": generation.top_p if top_p is None else top_p,
                "top_k": generation.top_k if top_k is None else top_k,
                "stop": generation.stop if stop is None else list(stop),
                "num_samples": generation.num_samples if num_samples is None else num_samples,
                "pass_at_k": generation.pass_at_k if pass_at_k is None else list(pass_at_k),
                "aggregation": generation.aggregation if aggregation is None else aggregation,
            }
        )
        task_spec = run.task_spec.model_copy(update={"generation": generation})
        overridden.append(replace(run, task_spec=task_spec))
    return overridden


def example_sample_key(example, index: int) -> str:
    example_id = getattr(example, "id", "")
    if example_id:
        return str(example_id)
    raw = getattr(example, "raw", None)
    if raw is not None:
        try:
            return json.dumps(raw, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(raw)
    return str(index)


def stable_sample_examples(examples: list, run: ResolvedBenchmarkRun) -> list:
    if run.task_spec.scorer.id == "typed_decision" and run.task_spec.metadata.get("td_task_id"):
        from .td_reader import select_task
        examples = select_task(examples, run.task_spec.metadata["td_task_id"])
    if run.task_spec.scorer.id == "typed_decision" and run.max_examples is not None and run.sample_seed is not None and run.task_spec.metadata.get("td_sample_strategy") == "stratified_stable_hash":
        from .td_sampling import stratified_stable_hash
        return stratified_stable_hash(examples, run)
    sample_seed = run.sample_seed
    if not sample_seed:
        return examples[: run.max_examples] if run.max_examples is not None else examples
    ranked = sorted(
        enumerate(examples),
        key=lambda item: hashlib.sha256(
            f"{sample_seed}\0{run.dataset_name}\0{example_sample_key(item[1], item[0])}".encode("utf-8")
        ).hexdigest(),
    )
    sampled = [example for _, example in ranked]
    return sampled[: run.max_examples] if run.max_examples is not None else sampled


def ordered_examples_for_run(examples: list, run: ResolvedBenchmarkRun) -> list:
    if not run.sample_seed:
        return examples
    unbounded_run = replace(run, max_examples=None)
    return stable_sample_examples(examples, unbounded_run)


def select_few_shot_examples(examples: list, run: ResolvedBenchmarkRun) -> list:
    few_shot_count = run.task_spec.few_shot.count
    if few_shot_count < 0 or few_shot_count > 25:
        raise ValueError(f"few_shot_count must be between 0 and 25 for {run.run_name}.")
    ordering_run = run
    if run.task_spec.few_shot.seed:
        ordering_task = run.task_spec.model_copy(
            update={
                "dataset": run.task_spec.few_shot.dataset or run.dataset_name,
            }
        )
        ordering_run = replace(
            run,
            task_spec=ordering_task,
            sample_seed=run.task_spec.few_shot.seed,
        )
    ordered = ordered_examples_for_run(examples, ordering_run)
    return ordered[:few_shot_count] if few_shot_count else []


def few_shot_match_value(example, field_name: str) -> str:
    value = getattr(example, "raw", {})
    for part in field_name.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(
                f"Few-shot match field {field_name!r} is missing from example "
                f"{getattr(example, 'id', '') or '<without-id>'}."
            )
        value = value[part]
    if value is None or not str(value).strip():
        raise ValueError(
            f"Few-shot match field {field_name!r} is empty for example "
            f"{getattr(example, 'id', '') or '<without-id>'}."
        )
    return str(value)


def select_matched_few_shot_examples(
    evaluation_examples: list,
    candidate_examples: list,
    run: ResolvedBenchmarkRun,
) -> list[list]:
    match_field = run.task_spec.few_shot.match_field
    if not match_field:
        return [candidate_examples for _ in evaluation_examples]

    groups: dict[str, list] = {}
    for candidate in candidate_examples:
        groups.setdefault(few_shot_match_value(candidate, match_field), []).append(candidate)

    count = run.task_spec.few_shot.count
    selected_groups: dict[str, list] = {}
    for group_name, group_examples in groups.items():
        if run.task_spec.few_shot.seed:
            group_examples = sorted(
                enumerate(group_examples),
                key=lambda item: hashlib.sha256(
                    (
                        f"{run.task_spec.few_shot.seed}\0{run.task_spec.few_shot.dataset}\0"
                        f"{group_name}\0{example_sample_key(item[1], item[0])}"
                    ).encode("utf-8")
                ).hexdigest(),
            )
            group_examples = [example for _, example in group_examples]
        selected_groups[group_name] = group_examples[:count]

    per_example = []
    for example in evaluation_examples:
        group_name = few_shot_match_value(example, match_field)
        selected = selected_groups.get(group_name, [])
        if len(selected) != count:
            raise ValueError(
                f"Benchmark {run.run_name} requests {count} few-shot examples matching "
                f"{match_field}={group_name!r}, but its demonstration dataset provides "
                f"only {len(selected)}."
            )
        per_example.append(selected)
    return per_example


def read_few_shot_examples(config: EffectiveBenchmarkConfig, run: ResolvedBenchmarkRun) -> list:
    if not config.few_shot_count:
        return []
    if config.few_shot_path is None:
        raise ValueError(
            f"Benchmark {run.run_name} requests {config.few_shot_count} few-shot examples "
            "but does not define a few-shot dataset."
        )
    few_shot_config = replace(
        config,
        path=config.few_shot_path,
        schema_version=config.few_shot_schema_version or config.schema_version,
    )
    if config.task_type == "free_response":
        examples = read_free_response_dataset(few_shot_config)
    else:
        examples = read_jsonl_dataset(few_shot_config)
    if config.few_shot_row_count is not None and len(examples) != config.few_shot_row_count:
        raise ValueError(
            f"Few-shot dataset {config.few_shot_dataset_name or config.few_shot_path} row count "
            f"mismatch: expected {config.few_shot_row_count}, got {len(examples)}."
        )
    if (
        config.few_shot_content_hash
        and isinstance(config.few_shot_path, Path)
        and config.few_shot_path.is_file()
    ):
        actual_hash = sha256_file(config.few_shot_path)
        if actual_hash != config.few_shot_content_hash:
            raise ValueError(
                f"Few-shot dataset {config.few_shot_dataset_name or config.few_shot_path} content "
                f"hash mismatch: expected {config.few_shot_content_hash}, got {actual_hash}."
            )
    for field_name, expected in (
        ("split", config.few_shot_split),
        ("language", config.few_shot_language),
    ):
        if expected is None:
            continue
        mismatched_ids = [
            str(getattr(example, "id", ""))
            for example in examples
            if str(getattr(example, "raw", {}).get(field_name, "")) != expected
        ]
        if mismatched_ids:
            raise ValueError(
                f"Few-shot dataset {config.few_shot_dataset_name or config.few_shot_path} has "
                f"{len(mismatched_ids)} record(s) outside declared {field_name}={expected!r}."
            )
    if config.few_shot_match_field:
        if len(examples) < config.few_shot_count:
            raise ValueError(
                f"Benchmark {run.run_name} requests {config.few_shot_count} few-shot examples "
                f"per {config.few_shot_match_field!r} value, but "
                f"{config.few_shot_dataset_name or config.few_shot_path} provides only "
                f"{len(examples)} total."
            )
        return examples
    selected = select_few_shot_examples(examples, run)
    if len(selected) != config.few_shot_count:
        raise ValueError(
            f"Benchmark {run.run_name} requests {config.few_shot_count} few-shot examples "
            f"but {config.few_shot_dataset_name or config.few_shot_path} provides only "
            f"{len(selected)}."
        )
    return selected


def normalized_example_question(example) -> str:
    question = str(getattr(example, "question", getattr(example, "prompt", "")))
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def ensure_no_few_shot_leakage(evaluation_examples: list, few_shot_examples: list) -> None:
    if not few_shot_examples:
        return
    evaluation_by_question = {
        normalized_example_question(example): str(getattr(example, "id", ""))
        for example in evaluation_examples
    }
    overlaps = []
    for example in few_shot_examples:
        fingerprint = normalized_example_question(example)
        if fingerprint in evaluation_by_question:
            overlaps.append(
                {
                    "few_shot_id": str(getattr(example, "id", "")),
                    "evaluation_id": evaluation_by_question[fingerprint],
                }
            )
    if overlaps:
        preview = ", ".join(
            f"few-shot={item['few_shot_id'] or '<missing>'}/eval={item['evaluation_id'] or '<missing>'}"
            for item in overlaps[:5]
        )
        raise ValueError(
            f"Few-shot leakage detected: {len(overlaps)} demonstration question(s) also occur "
            f"in the evaluation split ({preview})."
        )


def create_run_output_dir(base_output_dir: str) -> Path:
    output_root = Path(base_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    date_prefix = datetime.now().strftime("%Y-%m-%d")
    run_number = 1

    while True:
        run_output_path = output_root / f"{date_prefix}_run{run_number:03d}"
        if not run_output_path.exists():
            run_output_path.mkdir(parents=True, exist_ok=False)
            return run_output_path
        run_number += 1


def resolve_run_config(run: ResolvedBenchmarkRun) -> EffectiveBenchmarkConfig:
    dataset_config = DATASETS[run.dataset_name]
    task = run.task_spec
    few_shot_dataset = None
    if task.few_shot.dataset:
        few_shot_dataset = DATASETS.get(task.few_shot.dataset)
        if few_shot_dataset is None:
            raise ValueError(
                f"Benchmark {run.run_name} references unknown few-shot dataset: "
                f"{task.few_shot.dataset}"
            )
        if few_shot_dataset.task_type != dataset_config.task_type:
            raise ValueError(
                f"Benchmark {run.run_name} has incompatible evaluation and few-shot dataset types."
            )
    return EffectiveBenchmarkConfig(
        name=run.run_name,
        dataset_name=run.dataset_name,
        path=dataset_config.path,
        few_shot_path=few_shot_dataset.path if few_shot_dataset else None,
        source=dataset_config.source,
        question_key=dataset_config.question_key,
        choices_key=dataset_config.choices_key,
        answer_key=dataset_config.answer_key,
        task_type=dataset_config.task_type,
        question_only_before_newline=dataset_config.question_only_before_newline,
        dataset_version=dataset_config.version,
        schema_version=dataset_config.schema_version,
        content_hash=dataset_config.content_hash,
        dataset_split=dataset_config.split,
        language=dataset_config.language,
        row_count=dataset_config.row_count,
        few_shot_dataset_name=few_shot_dataset.name if few_shot_dataset else None,
        few_shot_dataset_version=few_shot_dataset.version if few_shot_dataset else None,
        few_shot_schema_version=few_shot_dataset.schema_version if few_shot_dataset else None,
        few_shot_content_hash=few_shot_dataset.content_hash if few_shot_dataset else None,
        few_shot_split=few_shot_dataset.split if few_shot_dataset else None,
        few_shot_language=few_shot_dataset.language if few_shot_dataset else None,
        few_shot_row_count=few_shot_dataset.row_count if few_shot_dataset else None,
        generation_prompt=task.formatter.id,
        formatter_version=task.formatter.version,
        generation_max_tokens=task.generation.max_tokens,
        generation_temperature=task.generation.temperature,
        generation_top_p=task.generation.top_p,
        generation_top_k=task.generation.top_k,
        generation_stop=tuple(task.generation.stop),
        generation_num_samples=task.generation.num_samples,
        generation_pass_at_k=tuple(task.generation.pass_at_k),
        generation_aggregation=task.generation.aggregation,
        append=task.formatter.append_choices,
        eval_mode=scorer(task.scorer.id).eval_mode,
        score_target=task.scorer.score_target,
        normalization=task.scorer.normalization,
        pmi_reference_prompt=task.scorer.pmi_reference_prompt,
        length_normalization=task.scorer.length_normalization,
        judge_prompt=task.scorer.judge_prompt,
        answer_verification=task.scorer.answer_verification,
        few_shot_count=task.few_shot.count,
        few_shot_seed=task.few_shot.seed,
        few_shot_match_field=task.few_shot.match_field,
        request_type=task.resolved_request_type,
        generation_api=task.generation_api,
        subgroup_fields=tuple(task.subgroup_fields),
        scorer_id=task.scorer.id,
    )


def validate_effective_run(run: ResolvedBenchmarkRun, config: EffectiveBenchmarkConfig) -> None:
    kind = dataset_type(config.task_type)
    scorer_def = scorer(config.scorer_id)
    expected_request_type = scorer_def.request_type
    if config.request_type != expected_request_type:
        raise ValueError(
            f"Benchmark {run.run_name} request_type={config.request_type!r} is incompatible "
            f"with eval_mode={config.eval_mode!r}; expected {expected_request_type!r}."
        )
    if config.request_type == "logprob" and config.generation_api != "auto":
        raise ValueError(
            f"Benchmark {run.run_name} is a logprob task and must use generation_api='auto'."
        )
    if config.few_shot_count and not kind.few_shot:
        raise ValueError(
            f"Benchmark {run.run_name} cannot use few-shot examples with task_type={config.task_type}."
        )
    if scorer_def.dataset_type != kind.public:
        expected = EVAL_MODE_BY_DATASET_TYPE.get(kind.public)
        requirement = (
            f"eval_mode={expected!r}" if expected else "a scorer of that data type"
        )
        raise ValueError(
            f"Benchmark {run.run_name} uses task_type={config.task_type!r}, which requires "
            f"{requirement}, not scorer {config.scorer_id!r}."
        )
    if scorer_def.generation_api and config.generation_api != scorer_def.generation_api:
        raise ValueError(
            f"Benchmark {run.run_name} ({config.scorer_id}) requires "
            f"generation_api={scorer_def.generation_api!r}: {scorer_def.generation_api_reason}."
        )
    if config.eval_mode in ANSWER_PARSED_EVAL_MODES and config.length_normalization:
        raise ValueError(
            f"Benchmark {run.run_name} uses eval_mode='{config.eval_mode}', which does not support length_normalization."
        )
    if config.eval_mode != LOGPROB_EVAL_MODE and config.normalization != "none":
        raise ValueError(
            f"Benchmark {run.run_name} uses eval_mode='{config.eval_mode}', which does not support normalization."
        )
    if (
        config.task_type != "graphwalks"
        and config.eval_mode == "logprob"
        and config.score_target == "label"
        and not config.append
    ):
        raise ValueError(
            f"Benchmark {run.run_name} uses logprob label scoring, which requires append=True."
        )
    if config.eval_mode == "logprob" and config.score_target != "text" and (
        config.normalization not in {"none", "pmi"} or config.length_normalization
    ):
        raise ValueError(
            f"Benchmark {run.run_name} uses logprob normalization, which requires score_target='text'."
        )
    if config.eval_mode not in EVAL_MODES:
        raise ValueError(f"Unsupported eval_mode for {run.run_name}: {config.eval_mode}")
    if config.score_target not in {"text", "label"}:
        raise ValueError(f"Unsupported score_target for {run.run_name}: {config.score_target}")
    if config.normalization not in {"none", "token", "char", "word", "pmi"}:
        raise ValueError(f"Unsupported normalization for {run.run_name}: {config.normalization}")
    if config.normalization == "pmi" and not config.pmi_reference_prompt:
        raise ValueError(
            f"Benchmark {run.run_name} uses PMI normalization without pmi_reference_prompt."
        )
    if config.normalization != "pmi" and config.pmi_reference_prompt is not None:
        raise ValueError(
            f"Benchmark {run.run_name} defines pmi_reference_prompt without PMI normalization."
        )
    if config.normalization == "pmi" and config.length_normalization:
        raise ValueError(f"Benchmark {run.run_name} cannot combine PMI and length normalization.")
    if config.generation_temperature < 0:
        raise ValueError(f"Benchmark {run.run_name} generation temperature must be at least 0.")
    if config.generation_top_p is not None and not 0 < config.generation_top_p <= 1:
        raise ValueError(f"Benchmark {run.run_name} generation top_p must be in (0, 1].")
    if config.generation_top_k is not None and config.generation_top_k < 1:
        raise ValueError(f"Benchmark {run.run_name} generation top_k must be at least 1.")
    if any(not value for value in config.generation_stop):
        raise ValueError(f"Benchmark {run.run_name} generation stop sequences must not be empty.")
    if len(config.generation_stop) != len(set(config.generation_stop)):
        raise ValueError(f"Benchmark {run.run_name} generation stop sequences must be unique.")
    if not 1 <= config.generation_num_samples <= 64:
        raise ValueError(f"Benchmark {run.run_name} generation num_samples must be in [1, 64].")
    if len(config.generation_pass_at_k) != len(set(config.generation_pass_at_k)):
        raise ValueError(f"Benchmark {run.run_name} generation pass_at_k values must be unique.")
    if any(
        value < 1 or value > config.generation_num_samples
        for value in config.generation_pass_at_k
    ):
        raise ValueError(
            f"Benchmark {run.run_name} generation pass_at_k values must be between 1 and num_samples."
        )
    if config.generation_num_samples > 1:
        if config.generation_temperature <= 0:
            raise ValueError(
                f"Benchmark {run.run_name} multi-sample generation requires temperature greater than zero."
            )
        if not scorer_def.multi_sample:
            raise ValueError(
                f"Benchmark {run.run_name} multi-sample generation only supports deterministic "
                "multiple-choice, free-response and code scorers."
            )
    if config.generation_aggregation not in {"first", "consensus", "mean"}:
        raise ValueError(
            f"Benchmark {run.run_name} has unsupported generation aggregation "
            f"{config.generation_aggregation!r}."
        )
    if config.generation_aggregation == "consensus" and config.generation_num_samples < 2:
        raise ValueError(
            f"Benchmark {run.run_name} consensus aggregation requires at least two samples."
        )
    if config.generation_aggregation not in scorer_def.multi_sample_aggregations:
        raise ValueError(
            f"Benchmark {run.run_name}: {config.scorer_id} supports "
            + " or ".join(f"'{item}'" for item in sorted(scorer_def.multi_sample_aggregations))
            + " aggregation."
        )


def prepare_run(run: ResolvedBenchmarkRun, config: EffectiveBenchmarkConfig) -> PreparedRun:
    validate_effective_run(run, config)
    reader_max_examples = None if run.sample_seed else run.max_examples
    examples = read_examples(config, reader_max_examples)
    few_shot_examples = read_few_shot_examples(config, run)
    ensure_no_few_shot_leakage(examples, few_shot_examples)
    examples = stable_sample_examples(examples, run)
    few_shot_examples_per_example = select_matched_few_shot_examples(
        examples,
        few_shot_examples,
        run,
    )
    return PreparedRun(
        run=run,
        config=config,
        examples=examples,
        few_shot_examples=few_shot_examples,
        few_shot_examples_per_example=few_shot_examples_per_example,
    )


def run_benchmark(
    dataset_names: list[str],
    output_dir: str,
    base_url: str = "http://127.0.0.1:8181/v1",
    model: str = "google/gemma-4-31B-it",
    api_key: str = "EMPTY",
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_insecure_skip_verify: bool | None = None,
    judge_generation_api: str = "chat",
    judge_disable_thinking: bool = False,
    judge_mtbench_guided_rating: bool = False,
    verifier_model: str | None = None,
    verifier_base_url: str | None = None,
    verifier_api_key: str | None = None,
    verifier_insecure_skip_verify: bool | None = None,
    verifier_default_headers: dict[str, str] | None = None,
    verifier_disable_thinking: bool = False,
    verifier_json_schema: bool = False,
    request_batch_size: int = 64,
    generation_concurrency: int = 16,
    judge_concurrency: int = 16,
    verifier_concurrency: int = 16,
    show_progress: bool = False,
    progress_every: int = 10,
    silent: bool = False,
    skip_preflight: bool = False,
    external_responses_file: str | None = None,
    replay_responses_file: str | None = None,
    minimum_coverage: float | None = None,
    max_request_failure_rate: float | None = DEFAULT_MAX_REQUEST_FAILURE_RATE,
    max_examples: int | None = None,
    insecure_skip_verify: bool = False,
    output_level: str = "responses",
    td_config: dict | None = None,
    add_bos_token: bool | None = None,
    bos_mode: str | None = None,
    bos_token: str | None = None,
    logprob_api: str | None = None,
    generation_api: str | None = None,
    model_profile: str | None = None,
    generation_temperature: float | None = None,
    generation_top_p: float | None = None,
    generation_top_k: int | None = None,
    generation_stop: list[str] | None = None,
    generation_num_samples: int | None = None,
    generation_pass_at_k: list[int] | None = None,
    generation_aggregation: str | None = None,
    generation_max_tokens: int | None = None,
    candidate_disable_thinking: bool = False,
    generation_extra_body: dict[str, Any] | None = None,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    stream_generation: bool = False,
    code_executor_mode: str = "auto",
    code_executor_socket: str | None = None,
    code_execution_timeout_seconds: float = 3.0,
    bootstrap_resamples: int = 10_000,
    confidence_level: float = 0.95,
    statistics_seed: str = "bilgeval-statistics-v1",
    collect_sample_outcomes: bool = False,
    dry_run: bool = False,
    completion_concurrency: int = 1,
    completion_batch_size: int | None = None,
    generation_client: str = "sync",
) -> dict:
    if generation_client not in ("sync", "async"):
        raise ValueError("generation_client must be 'sync' or 'async'.")
    if request_batch_size < 1:
        raise ValueError("request_batch_size must be at least 1.")
    validate_completion_settings(completion_concurrency, completion_batch_size)
    if generation_concurrency < 1:
        raise ValueError("generation_concurrency must be at least 1.")
    if judge_concurrency < 1:
        raise ValueError("judge_concurrency must be at least 1.")
    if verifier_concurrency < 1:
        raise ValueError("verifier_concurrency must be at least 1.")
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1.")
    if minimum_coverage is not None and not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between 0 and 1.")
    if max_request_failure_rate is not None and not 0.0 <= max_request_failure_rate <= 1.0:
        raise ValueError("max_request_failure_rate must be between 0 and 1.")
    if request_timeout <= 0:
        raise ValueError("request_timeout must be positive.")
    candidate_extra_body: dict[str, Any] = {}
    if generation_extra_body:
        if not isinstance(generation_extra_body, dict):
            raise ValueError("generation_extra_body must be a JSON object.")
        candidate_extra_body.update(json.loads(json.dumps(generation_extra_body)))
    if candidate_disable_thinking:
        merged_kwargs = dict(candidate_extra_body.get("chat_template_kwargs") or {})
        merged_kwargs.update(THINKING_DISABLED_EXTRA_BODY["chat_template_kwargs"])
        candidate_extra_body["chat_template_kwargs"] = merged_kwargs
    if not 0.1 <= code_execution_timeout_seconds <= 30:
        raise ValueError("Code execution timeout must be between 0.1 and 30 seconds.")
    if output_level not in OUTPUT_LEVELS:
        available = ", ".join(sorted(OUTPUT_LEVELS))
        raise ValueError(f"Unsupported output level: {output_level}. Available: {available}")
    validate_statistics_config(bootstrap_resamples, confidence_level)
    if not statistics_seed:
        raise ValueError("statistics_seed must not be empty.")
    profile_name, profile = resolve_model_profile(model, model_profile)
    serialization = profile_serialization(profile)
    bos_mode = bos_mode or (profile.resolved_bos_mode if profile else "auto")
    bos_token = bos_token or (profile.bos_token if profile else "<bos>")
    logprob_api = logprob_api or (profile.logprob_api if profile else "auto")
    if add_bos_token:
        bos_mode = "always"

    selected_runs = apply_max_examples_override(resolve_benchmark_runs(dataset_names), max_examples)
    if replay_responses_file is not None and len(selected_runs) != 1:
        raise ValueError("--replay-responses-file requires exactly one selected task run.")
    selected_runs = apply_generation_overrides(
        selected_runs,
        temperature=generation_temperature,
        top_p=generation_top_p,
        top_k=generation_top_k,
        stop=generation_stop,
        num_samples=generation_num_samples,
        pass_at_k=generation_pass_at_k,
        aggregation=generation_aggregation,
        max_tokens=generation_max_tokens,
    )
    resolved_run_configs = {run.run_name: resolve_run_config(run) for run in selected_runs}
    if judge_mtbench_guided_rating and any(
        config.eval_mode != "mtbench_judge"
        for config in resolved_run_configs.values()
    ):
        raise ValueError(
            "--judge-mtbench-guided-rating can only be used when every selected "
            "task uses eval_mode='mtbench_judge'."
        )
    generation_api = resolve_generation_api(
        list(resolved_run_configs.values()),
        generation_api,
        profile,
    )
    if serialization.generation.mode == "server_chat" and generation_api != "chat":
        raise ValueError(
            "The selected model profile requires generation_api='chat' for server_chat serialization."
        )
    if serialization.generation.mode == "template" and generation_api != "completions":
        raise ValueError(
            "The selected model profile uses an inline generation template and requires "
            "generation_api='completions'."
        )
    prepared_runs = [
        prepare_run(run, resolved_run_configs[run.run_name])
        for run in selected_runs
    ]
    external_responses_enabled = external_responses_file is not None
    def requires(role: str) -> bool:
        return any(
            role in scorer(config.scorer_id).requires
            for config in resolved_run_configs.values()
        )

    requires_judge_model = requires("judge")
    requires_verifier_model = any(
        config.answer_verification != "regex"
        for config in resolved_run_configs.values()
    )
    requires_logprob_client = requires("logprob")
    requires_code_executor = requires("executor")
    requires_tool_call_client = requires("tool")
    requires_generation_client = requires_judge_model or requires("generation")
    if external_responses_enabled:
        non_judge_runs = [
            run.run_name
            for run in selected_runs
            if resolved_run_configs[run.run_name].eval_mode != "judge"
        ]
        if non_judge_runs:
            raise ValueError(
                "--external-responses-file can only be used with mc_judge tasks. "
                f"Non-judge runs: {', '.join(non_judge_runs)}"
            )
        requires_generation_client = False
    if requires_judge_model and not judge_model and not dry_run:
        raise ValueError(
            "A judge model is required because at least one selected benchmark uses eval_mode='judge'. "
            "Pass --judge-model."
        )
    if requires_verifier_model and not verifier_model and not dry_run:
        raise ValueError(
            "A verifier model is required because at least one selected benchmark uses "
            "LLM answer verification. Pass --verifier-model."
        )
    effective_verifier_base_url = verifier_base_url or base_url
    effective_verifier_api_key = (
        verifier_api_key if verifier_api_key is not None else api_key
    )
    effective_verifier_insecure = (
        insecure_skip_verify
        if verifier_insecure_skip_verify is None
        else verifier_insecure_skip_verify
    )
    effective_judge_base_url = judge_base_url or base_url
    effective_judge_api_key = judge_api_key if judge_api_key is not None else api_key
    effective_judge_insecure = (
        insecure_skip_verify
        if judge_insecure_skip_verify is None
        else judge_insecure_skip_verify
    )
    effective_show_progress = show_progress and not silent

    output_path = create_run_output_dir(output_dir)
    trace_collector = (
        ApiTraceCollector(output_path / "api_traces.jsonl.gz")
        if output_level == "debug" and not dry_run
        else None
    )
    runtime = {
        "base_url": base_url,
        "judge_model": judge_model,
        "judge": {
            "model": judge_model,
            "base_url": effective_judge_base_url if judge_model else None,
            "generation_api": judge_generation_api,
            "insecure_skip_verify": effective_judge_insecure,
            "disable_thinking": judge_disable_thinking,
            "mtbench_guided_rating": judge_mtbench_guided_rating,
        },
        "answer_verifier": {
            "model": verifier_model,
            "base_url": effective_verifier_base_url if verifier_model else None,
            "generation_api": "chat",
            "insecure_skip_verify": effective_verifier_insecure,
            "disable_thinking": verifier_disable_thinking,
            "json_schema": verifier_json_schema,
        },
        "request_batch_size": request_batch_size,
        "completion_concurrency": completion_concurrency,
        "completion_batch_size": completion_batch_size,
        "generation_client": generation_client,
        "generation_concurrency": generation_concurrency,
        "judge_concurrency": judge_concurrency,
        "verifier_concurrency": verifier_concurrency,
        "candidate_disable_thinking": candidate_disable_thinking,
        "candidate_extra_body": candidate_extra_body or None,
        "request_timeout_seconds": request_timeout,
        "stream_generation": stream_generation,
        "generation_max_tokens_override": generation_max_tokens,
        "max_examples_override": max_examples,
        "output_level": output_level,
        "replay_responses_file": replay_responses_file,
        "replay_responses_hash": (
            sha256_file(Path(replay_responses_file)) if replay_responses_file else None
        ),
        "minimum_coverage": minimum_coverage,
        "max_request_failure_rate": max_request_failure_rate,
        "preflight_enabled": not skip_preflight and replay_responses_file is None,
        "insecure_skip_verify": insecure_skip_verify,
        "bos_mode": bos_mode,
        "bos_token": bos_token,
        "logprob_api": logprob_api,
        "generation_api": generation_api,
        "native_tool_calling": {
            "required": requires_tool_call_client,
            "transport": "openai_chat_tools" if requires_tool_call_client else None,
        },
        "code_execution": {
            "required": requires_code_executor,
            "mode": code_executor_mode,
            "socket": code_executor_socket,
            "timeout_seconds": code_execution_timeout_seconds,
        },
        "generation_overrides": {
            "temperature": generation_temperature,
            "top_p": generation_top_p,
            "top_k": generation_top_k,
            "stop": generation_stop,
            "num_samples": generation_num_samples,
            "pass_at_k": generation_pass_at_k,
            "aggregation": generation_aggregation,
        },
        "model_profile": {
            "name": profile_name,
            "version": profile.version if profile else None,
            "model_pattern": profile.model_pattern if profile else None,
            "fingerprint": stable_json_sha256(profile.model_dump(mode="json"))
            if profile
            else None,
        },
        "serialization": serialization_manifest(
            serialization,
            generation_api=generation_api,
        ),
        "statistics": {
            "bootstrap_resamples": bootstrap_resamples,
            "confidence_level": confidence_level,
            "seed": statistics_seed,
        },
    }
    td_client = None
    td_length_check = None
    if requires("td"):
        td_length_check, runtime["td"] = prepare_runtime(td_config, model)
        if not dry_run:
            td_client = TDClient(base_url, model, api_key, request_timeout)
    model_client = None
    if not dry_run and requires_logprob_client:
        model_client = create_logprob_client(
            base_url=base_url,
            model=model,
            api_key=api_key,
            insecure_skip_verify=insecure_skip_verify,
            bos_mode=bos_mode,
            bos_token=bos_token,
            logprob_api=logprob_api,
            prompt_renderer=serialization.logprob,
            trace_collector=trace_collector,
        )
        runtime["resolved_logprob_transport"] = model_client.transport_name
        runtime["resolved_bos"] = model_client.bos_diagnostics
    if model_client is not None:
        effective_add_bos_token = model_client.effective_add_bos_token
        effective_bos_diagnostics = model_client.bos_diagnostics
        bos_token = getattr(model_client, "bos_token", bos_token)
    elif (
        not dry_run
        and requires_generation_client
        and generation_api == "completions"
        and bos_mode == "auto"
    ):
        sequence_start = probe_sequence_start(
            base_url=base_url,
            model=model,
            api_key=api_key,
            insecure_skip_verify=insecure_skip_verify,
        )
        if sequence_start is not None:
            effective_add_bos_token = (
                sequence_start["bos_token"] is not None and not sequence_start["server_adds_bos"]
            )
            bos_token = sequence_start["bos_token"] or bos_token
            source = "vllm_tokenize_chat_template"
        else:
            effective_add_bos_token = model_requires_bos(model)
            source = "model_name"
        effective_bos_diagnostics = {
            "requested_mode": bos_mode,
            "effective_add_bos_token": effective_add_bos_token,
            "bos_token": bos_token,
            "source": source,
            "sequence_start_probe": sequence_start,
        }
    else:
        effective_add_bos_token = bos_mode == "always"
        effective_bos_diagnostics = {
            "requested_mode": bos_mode,
            "effective_add_bos_token": effective_add_bos_token,
            "source": "generation_only_no_logprob_probe",
        }
    probe_failed = (
        "sequence_start_probe" in effective_bos_diagnostics
        and effective_bos_diagnostics["sequence_start_probe"] is None
    )
    if bos_mode == "auto" and probe_failed:
        print(
            "bilgeval: warning: the server did not give the model's sequence-start token "
            f"(vLLM /tokenize with a chat template); BOS {'added' if effective_add_bos_token else 'not added'} "
            f"by the model-name rule for {model!r}. Pass --bos-mode always or never if that is wrong.",
            file=sys.stderr,
            flush=True,
        )
    runtime["model_runtime_fingerprint"] = stable_json_sha256(
        {
            "model": model,
            "model_profile": runtime["model_profile"],
            "serialization": runtime["serialization"],
            "base_url": base_url,
            "logprob_transport": model_client.transport_name if model_client else None,
            "bos": effective_bos_diagnostics,
        }
    )
    code_executor = None
    if not dry_run and requires_code_executor:
        code_executor = create_code_executor(
            mode=code_executor_mode,
            socket_path=code_executor_socket,
        )
        runtime["code_execution"]["resolved_executor"] = code_executor.name

    manifest = build_run_manifest(
        targets=dataset_names,
        model=model,
        runtime=runtime,
        prepared_runs=prepared_runs,
        registry_context=dict(ACTIVE_REGISTRY_CONTEXT),
        repository_root=REPOSITORY_ROOT,
    )
    manifest_path = output_path / "run_manifest.json"
    write_run_manifest(manifest_path, manifest)
    if dry_run:
        return {
            "dry_run": True,
            "targets": dataset_names,
            "model": model,
            "report_dir": str(output_path.resolve()),
            "manifest_file": str(manifest_path.resolve()),
            "runs": [
                {
                    "run_name": prepared.run.run_name,
                    "dataset_name": prepared.config.dataset_name,
                    "num_examples": len(prepared.examples),
                    "num_few_shot_examples": len(prepared.few_shot_examples),
                }
                for prepared in prepared_runs
            ],
        }

    text_client_type = OpenAICompatibleTextGenerationClient
    judge_client_type = OpenAICompatibleJudgeClient
    verifier_client_type = OpenAICompatibleAnswerVerifierClient
    if generation_client == "async" and (
        requires_generation_client or requires_judge_model or requires_verifier_model
    ):
        from .async_generation import (
            AsyncOpenAICompatibleAnswerVerifierClient,
            AsyncOpenAICompatibleJudgeClient,
            AsyncOpenAICompatibleTextGenerationClient,
        )

        text_client_type = AsyncOpenAICompatibleTextGenerationClient
        judge_client_type = AsyncOpenAICompatibleJudgeClient
        verifier_client_type = AsyncOpenAICompatibleAnswerVerifierClient

    candidate_generation_client = None
    live_candidate_client = None
    replay_client = None
    if requires_generation_client:
        live_candidate_client = text_client_type(
            base_url=base_url,
            model=model,
            api_key=api_key,
            insecure_skip_verify=insecure_skip_verify,
            add_bos_token=effective_add_bos_token,
            bos_token=bos_token,
            generation_api=generation_api,
            prompt_renderer=serialization.generation,
            trace_collector=trace_collector,
            max_concurrency=generation_concurrency,
            completion_concurrency=completion_concurrency,
            completion_batch_size=completion_batch_size,
            default_extra_body=candidate_extra_body or None,
            request_timeout=request_timeout,
            stream=stream_generation and generation_api == "chat",
        )
        if replay_responses_file is not None:
            replay_client = ArtifactReplayTextGenerationClient.from_path(
                replay_responses_file,
                expected_run_name=selected_runs[0].run_name,
                fallback_client=live_candidate_client,
            )
            candidate_generation_client = replay_client
        else:
            candidate_generation_client = live_candidate_client
    tool_call_client = None
    if requires_tool_call_client:
        tool_call_client = OpenAICompatibleToolCallClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            insecure_skip_verify=insecure_skip_verify,
            trace_collector=trace_collector,
            max_concurrency=generation_concurrency,
            request_timeout=request_timeout,
        )
    judge_client = None
    if requires_judge_model:
        judge_extra_body: dict[str, Any] = {}
        if judge_disable_thinking:
            judge_extra_body["chat_template_kwargs"] = {
                "thinking": False,
                "enable_thinking": False,
            }
        if judge_mtbench_guided_rating:
            judge_extra_body["regex"] = (
                r"\[\[(10(\.0+)?|[1-9](\.[0-9]+)?)\]\]"
            )
        judge_client = judge_client_type(
            base_url=effective_judge_base_url,
            model=judge_model or "",
            api_key=effective_judge_api_key,
            insecure_skip_verify=effective_judge_insecure,
            add_bos_token=False,
            bos_token=bos_token,
            generation_api=judge_generation_api,
            trace_collector=trace_collector,
            request_role="judge",
            default_extra_body=judge_extra_body or None,
            max_concurrency=judge_concurrency,
            completion_concurrency=completion_concurrency,
            completion_batch_size=completion_batch_size,
            request_timeout=request_timeout,
        )
    verifier_client = None
    if requires_verifier_model:
        verifier_extra_body: dict[str, Any] = {}
        if verifier_disable_thinking:
            verifier_extra_body["chat_template_kwargs"] = dict(
                THINKING_DISABLED_EXTRA_BODY["chat_template_kwargs"]
            )
        if verifier_json_schema:
            verifier_extra_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_verification",
                    "schema": {
                        "type": "object",
                        "properties": {"match": {"type": "boolean"}},
                        "required": ["match"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        verifier_client = verifier_client_type(
            base_url=effective_verifier_base_url,
            model=verifier_model or "",
            api_key=effective_verifier_api_key,
            insecure_skip_verify=effective_verifier_insecure,
            generation_api="chat",
            trace_collector=trace_collector,
            request_role="answer_verifier",
            default_headers=verifier_default_headers,
            default_extra_body=verifier_extra_body or None,
            max_concurrency=verifier_concurrency,
            request_timeout=request_timeout,
        )
    external_responses = None
    if external_responses_file is not None:
        external_responses = read_external_responses(external_responses_file)

    if not skip_preflight and replay_client is None:
        run_preflight_checks(
            selected_runs=selected_runs,
            td_client=td_client,
            resolved_run_configs=resolved_run_configs,
            model_client=model_client,
            candidate_generation_client=candidate_generation_client,
            tool_call_client=tool_call_client,
            judge_client=judge_client,
            verifier_client=verifier_client,
            code_executor=code_executor,
            code_execution_timeout_seconds=code_execution_timeout_seconds,
            external_responses_enabled=external_responses_enabled,
            show_progress=effective_show_progress,
        )
    if trace_collector is not None:
        trace_collector.start()

    final_report = new_final_report(
        targets=dataset_names,
        output_path=output_path,
        manifest_path=manifest_path,
        model=model,
        judge_model=judge_model,
        verifier_model=verifier_model,
        judge_disable_thinking=judge_disable_thinking,
        judge_mtbench_guided_rating=judge_mtbench_guided_rating,
        verifier_disable_thinking=verifier_disable_thinking,
        verifier_json_schema=verifier_json_schema,
        candidate_disable_thinking=candidate_disable_thinking,
        candidate_extra_body=candidate_extra_body or None,
        request_timeout=request_timeout,
        stream_generation=stream_generation,
        generation_max_tokens=generation_max_tokens,
        request_batch_size=request_batch_size,
        generation_client=generation_client,
        generation_concurrency=generation_concurrency,
        judge_concurrency=judge_concurrency,
        verifier_concurrency=verifier_concurrency,
        output_level=output_level,
        preflight_enabled=not skip_preflight and replay_responses_file is None,
        external_responses_file=external_responses_file,
        replay_responses_file=replay_responses_file,
        minimum_coverage=minimum_coverage,
        max_request_failure_rate=max_request_failure_rate,
        max_examples=max_examples,
        bos_mode=bos_mode,
        bos_token=bos_token,
        effective_add_bos_token=effective_add_bos_token,
        bos_diagnostics=effective_bos_diagnostics,
        logprob_api=logprob_api,
        logprob_transport=model_client.transport_name if model_client else None,
        generation_api=generation_api,
        code_execution=runtime["code_execution"],
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        statistics_seed=statistics_seed,
    )
    accuracies = []
    family_accuracies: dict[str, list[float]] = {}
    sample_outcomes_by_run: dict[str, list[dict]] = {}
    request_clients = {
        role: client
        for role, client in (
            ("logprob", model_client),
            ("candidate", live_candidate_client),
            ("tool_call", tool_call_client),
            ("judge", judge_client),
            ("verifier", verifier_client),
        )
        if isinstance(client, CountsRequests)
    }
    manifest_runs = {
        item["run_name"]: item
        for item in manifest["runs"]
    }

    for prepared in prepared_runs:
        run = prepared.run
        config = prepared.config
        examples = prepared.examples
        few_shot_examples = prepared.few_shot_examples
        few_shot_examples_per_example = prepared.few_shot_examples_per_example
        task_generation_settings = generation_request_settings(config)

        if effective_show_progress:
            print(
                (
                    f"[{run.run_name}] start "
                    f"examples={len(examples)} "
                    f"eval_mode={config.eval_mode} "
                    f"generation_prompt={config.generation_prompt} "
                    f"judge_prompt={config.judge_prompt} "
                    f"append={config.append} "
                    f"score_target={config.score_target} "
                    f"normalization={config.normalization} "
                    f"length_normalization={config.length_normalization} "
                    f"few_shot_count={config.few_shot_count} "
                    f"few_shot_candidate_pool={len(few_shot_examples)}"
                ),
                file=sys.stderr,
                flush=True,
            )

        def dataset_progress_callback(
            completed_units: int,
            total_units: int,
            request_count: int,
        ) -> None:
            if not effective_show_progress:
                return
            if (
                request_count == 1
                or request_count % progress_every == 0
                or completed_units == total_units
            ):
                percent = (completed_units / total_units * 100) if total_units else 100.0
                print(
                    (
                        f"[{run.run_name}] progress "
                        f"units={completed_units}/{total_units} "
                        f"requests={request_count} "
                        f"percent={percent:.1f}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )

        requests_before = {role: client.request_tally.snapshot() for role, client in request_clients.items()}
        report = evaluate_run(
            RunContext(
                run=run,
                config=config,
                examples=examples,
                few_shot_examples=few_shot_examples,
                few_shot_examples_per_example=few_shot_examples_per_example,
                generation_settings=task_generation_settings,
                request_batch_size=request_batch_size,
                progress_callback=dataset_progress_callback,
                output_level=output_level,
                code_execution_timeout_seconds=code_execution_timeout_seconds,
                td_client=td_client,
                td_length_check=td_length_check,
                logprob_client=model_client,
                generation_client=candidate_generation_client,
                judge_client=judge_client,
                verifier_client=verifier_client,
                tool_call_client=tool_call_client,
                code_executor=code_executor,
                external_responses=external_responses,
            )
        )

        if (
            config.generation_aggregation == "mean"
            and config.generation_num_samples > 1
            and "multi_sample_metrics" in report
        ):
            report = mean_sample_report(
                report,
                metric_name=scorer(config.scorer_id).mean_metric_name,
            )
        summary = summarize_run(
            report,
            run_name=run.run_name,
            examples=examples,
            subgroup_fields=config.subgroup_fields,
            seed_context=(
                f"{model}\0{run.run_name}\0"
                f"{manifest_runs[run.run_name]['resolved_config_hash']}"
            ),
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            statistics_seed=statistics_seed,
        )
        if summary.is_binary:
            sample_outcomes_by_run[run.run_name] = summary.sample_outcomes
        entry, primary_metric = dataset_report_entry(
            run,
            config,
            summary,
            manifest_runs[run.run_name],
            num_few_shot_examples=len(few_shot_examples),
            output_level=output_level,
        )
        entry["requests"] = request_counts(request_clients, requests_before)
        if config.task_type == "typed_decision":
            if report["num_requests"]:
                entry["requests"]["td"] = {"sent": report["num_requests"], "failed": report["td_failed_requests"]}
            entry["td_identity"] = manifest_runs[run.run_name]["td_identity"]
        final_report["datasets"][run.run_name] = entry
        if primary_metric is not None:
            final_report["primary_metrics"][run.run_name] = primary_metric
        if summary.is_binary:
            accuracies.append(report["accuracy"])
            family_accuracies.setdefault(metric_family(config), []).append(report["accuracy"])
        accumulate_totals(final_report, report)

        if output_level != "summary":
            response_file = output_path / response_artifact_filename(run.run_name)
            final_report["artifacts"]["responses"][run.run_name] = write_response_artifact(
                response_file,
                run_name=run.run_name,
                results=report["results"],
            )

        if effective_show_progress:
            metric_text = (
                f"accuracy={report['accuracy']:.4f}"
                if summary.is_binary
                else (f"{report['metric_name']}={report['score']:.4f}" if report["score"] is not None else "TD: ortak küme boş")
            )
            print(
                (
                    f"[{run.run_name}] done "
                    f"{metric_text} "
                    f"requests={report['num_requests']}"
                ),
                file=sys.stderr,
                flush=True,
            )

    gate_failures = finalize_report(
        final_report,
        accuracies=accuracies,
        family_accuracies=family_accuracies,
        replay_client=replay_client,
        replay_responses_file=replay_responses_file,
        minimum_coverage=minimum_coverage,
        max_request_failure_rate=max_request_failure_rate,
    )

    if trace_collector is not None:
        final_report["artifacts"]["api_traces"] = trace_collector.close()

    if td_client is not None:
        td_client.close()
    final_file = output_path / "final_report.json"
    with open(final_file, "w", encoding="utf-8") as file:
        json.dump(final_report, file, ensure_ascii=False, indent=2)

    if gate_failures:
        raise ValueError(" ".join(gate_failures) + f" Report was written to {final_file}.")

    if collect_sample_outcomes:
        final_report["_sample_outcomes"] = sample_outcomes_by_run
    return final_report


def main() -> None:
    """The console entry point (setup.py: bilgeval = bilgeval.main:main)."""
    from .cli import main as cli_main

    cli_main()
