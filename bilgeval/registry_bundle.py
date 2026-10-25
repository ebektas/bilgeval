import hashlib
import json
import re
import string
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .generation_prompts import (
    BUILTIN_GENERATION_PROMPTS,
    CHOICE_FIELDS,
    GENERATION_PROMPTS,
    PROMPT_FIELDS,
)
from .judge_prompts import (
    BUILTIN_JUDGE_PROMPTS,
    JUDGE_PROMPTS,
    JudgePromptConfig,
)
from .registry import DATASETS, DatasetConfig, _resolve_registry_path
from .scorer_table import DATASET_TYPES, PUBLIC_TASK_TYPES, SCORER_IDS, SCORERS


REGISTRY_SCHEMA_VERSION = "bilgeval.registry/v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetReaderSpec(StrictModel):
    source: Literal["jsonl", "local", "s3"] = "jsonl"
    question_key: str = Field(default="question", min_length=1)
    choices_key: str = Field(default="choices", min_length=1)
    answer_key: str = Field(default="answer", min_length=1)
    question_only_before_newline: bool = False


class DatasetSpec(StrictModel):
    version: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    schema_version: str = Field(min_length=1)
    task_type: Literal[PUBLIC_TASK_TYPES]
    split: str = Field(min_length=1)
    language: str = Field(min_length=1)
    row_count: int | None = Field(default=None, ge=1)
    reader: DatasetReaderSpec = Field(default_factory=DatasetReaderSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FormatterSpec(StrictModel):
    id: str = Field(min_length=1)
    version: str = Field(default="1", min_length=1)
    append_choices: bool = False


class ScorerSpec(StrictModel):
    id: Literal[SCORER_IDS]
    score_target: Literal["text", "label"] = "text"
    normalization: Literal["none", "token", "char", "word", "pmi"] = "none"
    length_normalization: bool = False
    pmi_reference_prompt: str | None = None
    judge_prompt: str | None = None
    answer_verification: Literal["regex", "llm_as_regex"] = "regex"

    @field_validator("answer_verification", mode="before")
    @classmethod
    def _normalize_answer_verification(cls, value):
        if value == "llm":
            return "llm_as_regex"
        return value


class JudgePromptSpec(StrictModel):
    template: str = Field(min_length=1)
    output_type: Literal["binary_correctness"] = "binary_correctness"


class GenerationSpec(StrictModel):
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    stop: list[str] = Field(default_factory=list)
    num_samples: int = Field(default=1, ge=1, le=64)
    pass_at_k: list[int] = Field(default_factory=lambda: [1])
    aggregation: Literal["first", "consensus", "mean"] = "first"

    @model_validator(mode="after")
    def validate_stop_sequences(self):
        if any(not value for value in self.stop):
            raise ValueError("Generation stop sequences must not be empty.")
        if len(self.stop) != len(set(self.stop)):
            raise ValueError("Generation stop sequences must be unique.")
        if len(self.pass_at_k) != len(set(self.pass_at_k)):
            raise ValueError("Generation pass_at_k values must be unique.")
        if any(value < 1 or value > self.num_samples for value in self.pass_at_k):
            raise ValueError(
                "Generation pass_at_k values must be between 1 and num_samples."
            )
        if self.num_samples > 1 and self.temperature <= 0:
            raise ValueError("Multi-sample generation requires temperature greater than zero.")
        if self.aggregation == "consensus" and self.num_samples < 2:
            raise ValueError("Consensus aggregation requires at least two samples.")
        return self


class PromptRendererSpec(StrictModel):
    mode: Literal["plain", "template", "server_chat"] = "plain"
    template_id: str = Field(default="plain", min_length=1)
    template_version: str = Field(default="1", min_length=1)
    template: str | None = None
    template_hash: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    @model_validator(mode="after")
    def validate_renderer(self):
        if self.mode == "template":
            if self.template is None:
                raise ValueError("A template renderer requires an inline template.")
            try:
                fields = [
                    field_name
                    for _, field_name, _, _ in string.Formatter().parse(self.template)
                    if field_name is not None
                ]
            except ValueError as exc:
                raise ValueError(f"Invalid model prompt template: {exc}") from exc
            if fields != ["prompt"]:
                raise ValueError(
                    "A model prompt template must contain exactly one {prompt} field."
                )
            actual_hash = "sha256:" + hashlib.sha256(
                self.template.encode("utf-8")
            ).hexdigest()
            if self.template_hash is not None and self.template_hash != actual_hash:
                raise ValueError(
                    "Model prompt template_hash does not match the inline template."
                )
        elif self.template is not None or self.template_hash is not None:
            raise ValueError(
                "Only a template renderer may define template or template_hash."
            )
        return self


class ModelSerializationSpec(StrictModel):
    tokenizer_id: str | None = Field(default=None, min_length=1)
    tokenizer_revision: str | None = Field(default=None, min_length=1)
    logprob: PromptRendererSpec = Field(default_factory=PromptRendererSpec)
    generation: PromptRendererSpec = Field(default_factory=PromptRendererSpec)


class FewShotSpec(StrictModel):
    count: int = Field(default=0, ge=0, le=25)
    dataset: str | None = None
    seed: str | None = None
    match_field: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_dataset_for_examples(self):
        if self.count and not self.dataset:
            raise ValueError("few_shot.dataset is required when few_shot.count is greater than zero.")
        if self.match_field and not self.count:
            raise ValueError("few_shot.match_field requires few_shot.count to be greater than zero.")
        return self


class ComponentSpec(StrictModel):
    id: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(StrictModel):
    version: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    request_type: Literal["logprob", "generate", "decide"] | None = None
    generation_api: Literal["auto", "completions", "chat"] = "auto"
    formatter: FormatterSpec
    scorer: ScorerSpec
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    few_shot: FewShotSpec = Field(default_factory=FewShotSpec)
    filters: list[ComponentSpec] = Field(default_factory=list)
    metrics: list[ComponentSpec] = Field(
        min_length=1,
        default_factory=lambda: [ComponentSpec(id="accuracy")]
    )
    subgroup_fields: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def resolved_request_type(self) -> str:
        return self.request_type or SCORERS[self.scorer.id].request_type


class SuiteTaskSpec(StrictModel):
    task: str = Field(min_length=1)
    run_name: str | None = None
    max_examples: int | None = Field(default=None, ge=1)
    sample_seed: str | None = None


class SuiteSpec(StrictModel):
    version: str = Field(min_length=1)
    tasks: list[SuiteTaskSpec] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelProfile(StrictModel):
    version: str = Field(min_length=1)
    model_pattern: str = Field(min_length=1)
    bos_mode: Literal["auto", "always", "never"] = "auto"
    add_bos_token: bool | None = None
    bos_token: str = "<bos>"
    logprob_api: Literal["auto", "openai", "sglang"] = "auto"
    generation_api: Literal["completions", "chat"] = "completions"
    serialization: ModelSerializationSpec = Field(default_factory=ModelSerializationSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_pattern(self):
        try:
            re.compile(self.model_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid model_pattern regular expression: {exc}") from exc
        if self.add_bos_token is not None and self.bos_mode != "auto":
            legacy_mode = "always" if self.add_bos_token else "never"
            if self.bos_mode != legacy_mode:
                raise ValueError(
                    "ModelProfile add_bos_token conflicts with bos_mode; remove the deprecated field."
                )
        if self.serialization.logprob.mode == "server_chat":
            raise ValueError(
                "Logprob serialization cannot use server_chat because continuation scores "
                "require a fully materialized prompt."
            )
        if (
            self.serialization.generation.mode == "server_chat"
            and self.generation_api != "chat"
        ):
            raise ValueError(
                "Generation serialization mode server_chat requires generation_api='chat'."
            )
        if (
            self.serialization.generation.mode == "template"
            and self.generation_api != "completions"
        ):
            raise ValueError(
                "An inline generation template requires generation_api='completions' to avoid "
                "applying a second server-side chat template."
            )
        return self

    @property
    def resolved_bos_mode(self) -> str:
        if self.add_bos_token is None:
            return self.bos_mode
        return "always" if self.add_bos_token else "never"


class RegistryBundle(StrictModel):
    schema_version: Literal[REGISTRY_SCHEMA_VERSION]
    registry_version: str = Field(min_length=1)
    datasets: dict[str, DatasetSpec]
    tasks: dict[str, TaskSpec]
    suites: dict[str, SuiteSpec] = Field(default_factory=dict)
    generation_prompts: dict[str, str] = Field(default_factory=dict)
    judge_prompts: dict[str, JudgePromptSpec] = Field(default_factory=dict)
    model_profiles: dict[str, ModelProfile] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references_and_components(self):
        for collection_name, values in (
            ("dataset", self.datasets),
            ("task", self.tasks),
            ("suite", self.suites),
            ("judge prompt", self.judge_prompts),
            ("model profile", self.model_profiles),
        ):
            empty_names = [name for name in values if not name.strip()]
            if empty_names:
                raise ValueError(f"Registry {collection_name} names must not be empty.")
        collisions = set(self.tasks) & set(self.suites)
        if collisions:
            raise ValueError(
                "Task and suite names share one CLI namespace and must be unique: "
                + ", ".join(sorted(collisions))
            )

        prompt_templates = {**BUILTIN_GENERATION_PROMPTS, **self.generation_prompts}
        available_prompts = set(prompt_templates)
        for prompt_name, template in self.generation_prompts.items():
            try:
                fields = {
                    field_name
                    for _, field_name, _, _ in string.Formatter().parse(template)
                    if field_name is not None
                }
            except ValueError as exc:
                raise ValueError(f"Formatter {prompt_name!r} has invalid braces: {exc}") from exc
            unsupported_fields = fields - PROMPT_FIELDS
            if unsupported_fields:
                raise ValueError(
                    f"Formatter {prompt_name!r} uses unsupported fields: "
                    + ", ".join(sorted(unsupported_fields))
                )
            if "question" not in fields:
                raise ValueError(f"Formatter {prompt_name!r} must include {{question}}.")
        available_judge_prompts = set(BUILTIN_JUDGE_PROMPTS) | set(self.judge_prompts)
        required_judge_fields = {
            "question",
            "choices",
            "gold_label",
            "gold_text",
            "candidate_answer",
        }
        for prompt_name, prompt in self.judge_prompts.items():
            try:
                fields = {
                    field_name
                    for _, field_name, _, _ in string.Formatter().parse(prompt.template)
                    if field_name is not None
                }
            except ValueError as exc:
                raise ValueError(
                    f"Judge prompt {prompt_name!r} has invalid braces: {exc}"
                ) from exc
            unsupported_fields = fields - required_judge_fields
            if unsupported_fields:
                raise ValueError(
                    f"Judge prompt {prompt_name!r} uses unsupported fields: "
                    + ", ".join(sorted(unsupported_fields))
                )
            missing_fields = required_judge_fields - fields
            if missing_fields:
                raise ValueError(
                    f"Judge prompt {prompt_name!r} is missing fields: "
                    + ", ".join(sorted(missing_fields))
                )
        for task_name, task in self.tasks.items():
            dataset = self.datasets.get(task.dataset)
            if dataset is None:
                raise ValueError(f"Task {task_name!r} references unknown dataset {task.dataset!r}.")
            scorer = SCORERS[task.scorer.id]
            if dataset.task_type != scorer.dataset_type:
                raise ValueError(
                    f"Task {task_name!r} uses scorer {task.scorer.id!r}, which requires "
                    f"{scorer.dataset_type!r} data, not {dataset.task_type!r}."
                )
            expected_request_type = scorer.request_type
            if task.resolved_request_type != expected_request_type:
                raise ValueError(
                    f"Task {task_name!r} request_type={task.resolved_request_type!r} is incompatible "
                    f"with scorer {task.scorer.id!r}; expected {expected_request_type!r}."
                )
            if task.resolved_request_type == "logprob" and task.generation_api != "auto":
                raise ValueError(
                    f"Task {task_name!r} is a logprob task and must use generation_api='auto'; "
                    "chat/completions generation is not used for candidate scoring."
                )
            if scorer.generation_api and task.generation_api != scorer.generation_api:
                raise ValueError(
                    f"Task {task_name!r} ({task.scorer.id}) must use "
                    f"generation_api={scorer.generation_api!r} because "
                    f"{scorer.generation_api_reason}."
                )
            if task.generation.num_samples > 1 and task.resolved_request_type != "generate":
                raise ValueError(
                    f"Task {task_name!r} configures multi-sample generation for a "
                    "non-generation scorer."
                )
            if any(not field.strip() for field in task.subgroup_fields):
                raise ValueError(f"Task {task_name!r} subgroup_fields must not contain empty names.")
            if len(task.subgroup_fields) != len(set(task.subgroup_fields)):
                raise ValueError(f"Task {task_name!r} subgroup_fields must be unique.")
            if dataset.task_type == "typed_decision":
                if task.formatter.id != "typed_decision_v1":
                    raise ValueError("TD requires typed_decision_v1 formatter")
                if task.few_shot.count or task.formatter.append_choices or task.generation.num_samples != 1:
                    raise ValueError("TD sends exactly one structured request per state")
            elif task.formatter.id not in available_prompts:
                raise ValueError(
                    f"Task {task_name!r} references unknown formatter {task.formatter.id!r}."
                )
            formatter_template = prompt_templates.get(task.formatter.id, "")
            if dataset.task_type == "multiple_choice":
                if task.scorer.score_target == "label" and not task.formatter.append_choices:
                    raise ValueError(
                        f"Multiple-choice label task {task_name!r} must append choices."
                    )
                if task.formatter.append_choices and not any(
                    "{" + field + "}" in formatter_template for field in CHOICE_FIELDS
                ):
                    raise ValueError(
                        f"Multiple-choice formatter {task.formatter.id!r} must include "
                        "{choices_block}, {choices_lines} or {choices_paren}."
                    )
            if dataset.task_type in {"free_response", "code_generation"} and task.formatter.append_choices:
                raise ValueError(
                    f"{dataset.task_type} task {task_name!r} cannot append choices."
                )
            if not DATASET_TYPES[dataset.task_type].few_shot:
                if task.formatter.append_choices:
                    raise ValueError(
                        f"Specialized task {task_name!r} cannot append choices."
                    )
                if task.few_shot.count:
                    raise ValueError(
                        f"Specialized task {task_name!r} does not support few-shot examples."
                    )
            if scorer.request_type == "logprob":
                if task.scorer.normalization == "pmi" and not task.scorer.pmi_reference_prompt:
                    raise ValueError(
                        f"Task {task_name!r} uses PMI normalization and must define "
                        "scorer.pmi_reference_prompt."
                    )
                if task.scorer.normalization == "pmi" and task.scorer.length_normalization:
                    raise ValueError(
                        f"Task {task_name!r} cannot combine PMI with length_normalization."
                    )
                if task.scorer.normalization != "pmi" and task.scorer.pmi_reference_prompt is not None:
                    raise ValueError(
                        f"Task {task_name!r} defines pmi_reference_prompt without PMI normalization."
                    )
                if task.scorer.score_target == "label" and (
                    task.scorer.normalization not in {"none", "pmi"}
                    or task.scorer.length_normalization
                ):
                    raise ValueError(
                        f"Task {task_name!r} cannot normalize label logprob scores."
                    )
            elif task.scorer.normalization != "none" or task.scorer.length_normalization:
                raise ValueError(
                    f"Task {task_name!r} uses normalization with a non-logprob scorer."
                )
            if scorer.dataset_type == "free_response" and task.scorer.score_target != "text":
                raise ValueError(f"Free-response task {task_name!r} must use score_target='text'.")
            if task.scorer.answer_verification != "regex" and not scorer.answer_verification:
                raise ValueError(
                    f"Task {task_name!r} can only use LLM answer verification with "
                    "a regex-match scorer."
                )
            if scorer.judge_prompt == "required" and not task.scorer.judge_prompt:
                raise ValueError(f"Judge task {task_name!r} must define scorer.judge_prompt.")
            if (
                scorer.judge_prompt == "required"
                and task.scorer.judge_prompt not in available_judge_prompts
            ):
                raise ValueError(
                    f"Judge task {task_name!r} references unknown judge prompt "
                    f"{task.scorer.judge_prompt!r}."
                )
            if scorer.judge_prompt != "required" and task.scorer.judge_prompt is not None:
                raise ValueError(
                    f"Task {task_name!r} defines judge_prompt without the mc_judge scorer."
                )
            if task.filters:
                raise ValueError(
                    f"Task {task_name!r} configures filters, which are reserved but not implemented in registry v1."
                )
            expected_metric = scorer.expected_metric
            unsupported_metrics = [
                metric.id for metric in task.metrics if metric.id != expected_metric
            ]
            if unsupported_metrics:
                raise ValueError(
                    f"Task {task_name!r} uses unsupported metrics: {', '.join(unsupported_metrics)}."
                )
            if any(metric.parameters for metric in task.metrics):
                raise ValueError(
                    f"Task {task_name!r} configures metric parameters, which are not implemented in registry v1."
                )
            if len(task.metrics) != 1:
                raise ValueError(
                    f"Task {task_name!r} must configure {expected_metric} exactly once "
                    "in registry v1."
                )
            few_shot_dataset_name = task.few_shot.dataset
            if few_shot_dataset_name:
                few_shot_dataset = self.datasets.get(few_shot_dataset_name)
                if few_shot_dataset is None:
                    raise ValueError(
                        f"Task {task_name!r} references unknown few-shot dataset "
                        f"{few_shot_dataset_name!r}."
                    )
                if few_shot_dataset.task_type != dataset.task_type:
                    raise ValueError(
                        f"Task {task_name!r} has incompatible evaluation and few-shot dataset types."
                    )

        for suite_name, suite in self.suites.items():
            for task_ref in suite.tasks:
                if task_ref.task not in self.tasks:
                    raise ValueError(
                        f"Suite {suite_name!r} references unknown task {task_ref.task!r}."
                    )
        return self


def load_registry_bundle(registry_path: Path) -> RegistryBundle:
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read registry file {registry_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Registry file {registry_path} is not valid JSON: {exc}") from exc
    return RegistryBundle.model_validate(raw)


def registry_file_hash(registry_path: Path) -> str:
    return "sha256:" + hashlib.sha256(registry_path.read_bytes()).hexdigest()


def install_registry_bundle(bundle: RegistryBundle, registry_path: Path) -> None:
    datasets = {}
    for name, spec in bundle.datasets.items():
        source = spec.reader.source
        if spec.uri.startswith("s3://"):
            source = "s3"
        datasets[name] = DatasetConfig(
            name=name,
            path=_resolve_registry_path(spec.uri, registry_path),
            source=source,
            question_key=spec.reader.question_key,
            choices_key=spec.reader.choices_key,
            answer_key=spec.reader.answer_key,
            task_type=DATASET_TYPES[spec.task_type].internal,
            version=spec.version,
            schema_version=spec.schema_version,
            content_hash=spec.content_hash,
            split=spec.split,
            language=spec.language,
            row_count=spec.row_count,
            question_only_before_newline=spec.reader.question_only_before_newline,
        )

    DATASETS.clear()
    DATASETS.update(datasets)
    GENERATION_PROMPTS.clear()
    GENERATION_PROMPTS.update(BUILTIN_GENERATION_PROMPTS)
    GENERATION_PROMPTS.update(bundle.generation_prompts)
    JUDGE_PROMPTS.clear()
    JUDGE_PROMPTS.update(BUILTIN_JUDGE_PROMPTS)
    JUDGE_PROMPTS.update(
        {
            name: JudgePromptConfig(
                template=spec.template,
                output_type=spec.output_type,
            )
            for name, spec in bundle.judge_prompts.items()
        }
    )
