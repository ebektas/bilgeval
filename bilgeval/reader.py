import json
import csv
import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .registry import DatasetConfig
from .scorer_table import dataset_type
from .schema import (
    CodeGenerationInstance,
    INSTANCE_SCHEMA_VERSION,
    FreeResponseInstance,
    FunctionCallInstance,
    MultipleChoiceInstance,
    parse_canonical_instance,
)


@dataclass
class Choice:
    label: str
    text: str


@dataclass
class Example:
    id: str
    question: str
    choices: list[Choice]
    answer_label: str
    answer_text: str
    raw: dict[str, Any]


@dataclass
class GraphWalksExample:
    id: str
    prompt: str
    answer_nodes: list[str]
    problem_type: str
    raw: dict[str, Any]


@dataclass
class OneRulerExample:
    id: str
    prompt: str
    outputs: list[str]
    task: str
    raw: dict[str, Any]


ONERULER_TASKS = frozenset(
    {"niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery", "niah_none", "cwe", "cwe_easy"}
)


@dataclass
class IFEvalExample:
    id: str
    prompt: str
    instruction_ids: list[int]
    verifier_kwargs: dict[str, Any]
    raw: dict[str, Any]


@dataclass
class OfficialIFEvalExample:
    """A row of Google IFEval or IFBench in the upstream format."""

    id: str
    prompt: str
    instruction_ids: list[str]
    kwargs: list[dict[str, Any]]
    raw: dict[str, Any]


@dataclass
class FreeResponseExample:
    id: str
    question: str
    answer: str
    answer_text: str
    raw: dict[str, Any]


@dataclass
class HumanEvalExample:
    id: str
    prompt: str
    test: str
    entry_point: str
    canonical_solution: str | None
    raw: dict[str, Any]


@dataclass
class LiveCodeBenchExample:
    id: str
    question_title: str
    question_content: str
    starter_code: str
    fn_name: str | None
    num_tests: int
    tests: str
    raw: dict[str, Any]

    @property
    def question(self) -> str:
        return self.question_content


@dataclass
class FunctionCallExample:
    id: str
    messages: list[dict[str, str]]
    tools: list[dict[str, Any]]
    expected_calls: list[dict[str, Any]]
    should_call: bool
    call_order: str
    scoring_mode: str
    category: str
    raw: dict[str, Any]


@dataclass
class SafetyBehaviorExample:
    id: str
    prompt: str
    subcategory: str
    expected: str
    raw: dict[str, Any]


@dataclass
class DocumentQAExample:
    id: str
    document: str
    question: str
    answer: Any
    answer_format: str
    task_type: str
    evidence_spans: list[str]
    raw: dict[str, Any]


@dataclass
class MultiTurnJudgeExample:
    id: str
    task_id: str
    task_name: str
    topic: str
    turns: list[dict[str, str]]
    eval_turn_positions: list[int]
    evaluation_initial: str
    evaluation_prompt: str
    scoring_format: str
    raw: dict[str, Any]


@dataclass
class ExternalResponse:
    dataset_name: str
    id: str
    candidate_response: str
    generation_prompt_text: str | None
    raw: dict[str, Any]


def normalize_question(question: Any, config: DatasetConfig) -> str:
    normalized_question = str(question)

    if config.question_only_before_newline:
        normalized_question = normalized_question.splitlines()[0]

    return normalized_question


def normalize_choices(raw_choices: list[dict | str] | dict) -> list[Choice]:
    if isinstance(raw_choices, dict):
        texts = raw_choices.get("text") or raw_choices.get("content") or raw_choices.get("answer")
        labels = raw_choices.get("label")
        if isinstance(texts, list):
            if labels is None:
                labels = [chr(ord("A") + i) for i in range(len(texts))]
            if not isinstance(labels, list):
                raise TypeError(f"Choice labels must be a list: {labels}")
            if len(labels) != len(texts):
                raise ValueError("Choice labels and texts must have the same length.")
            raw_choices = [{"label": label, "text": text} for label, text in zip(labels, texts)]
        else:
            raise TypeError(f"Unsupported choice dict shape: {raw_choices}")

    choices = []

    for i, choice in enumerate(raw_choices):
        if isinstance(choice, str):
            label = chr(ord("A") + i)
            text = choice
        elif isinstance(choice, dict):
            label = choice.get("label", chr(ord("A") + i))
            text = choice.get("text") or choice.get("content") or choice.get("answer")
            if text is None:
                raise ValueError(f"Choice has no text field: {choice}")
        else:
            raise TypeError(f"Unsupported choice type: {type(choice)}")

        choices.append(Choice(label=str(label), text=str(text)))

    if len(choices) < 2:
        raise ValueError("Multiple-choice examples must define at least two choices.")
    empty_labels = [choice.label for choice in choices if not choice.label.strip()]
    if empty_labels:
        raise ValueError("Choice labels must not be empty.")
    labels = [choice.label for choice in choices]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Choice labels must be unique: {labels}")
    if any(not choice.text.strip() for choice in choices):
        raise ValueError("Choice text must not be empty.")
    return choices


LEGACY_ID_KEYS = (
    "id",
    "Record ID",
    "question_id",
    "unique_id",
    "uuid",
    "task_id",
    "prompt_id",
    "__translation_record_key",
)


def legacy_record_id(obj: dict[str, Any], dataset_name: str) -> str:
    for key in LEGACY_ID_KEYS:
        value = obj.get(key)
        if value is not None and str(value).strip():
            return str(value)
    canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"{dataset_name}-{digest}"


def normalize_example(obj: dict[str, Any], config: DatasetConfig) -> Example:
    if config.schema_version == INSTANCE_SCHEMA_VERSION:
        instance = parse_canonical_instance(obj)
        if not isinstance(instance, MultipleChoiceInstance):
            raise TypeError(
                f"Expected a multiple_choice instance, got {instance.task_type!r}."
            )
        choices = [Choice(label=choice.label, text=choice.text) for choice in instance.input.choices]
        answer_text = instance.target.answer_text or next(
            choice.text
            for choice in instance.input.choices
            if choice.label == instance.target.answer_label
        )
        return Example(
            id=instance.id,
            question=instance.input.question,
            choices=choices,
            answer_label=instance.target.answer_label,
            answer_text=answer_text,
            raw=obj,
        )

    if config.schema_version != "legacy":
        raise ValueError(f"Unsupported dataset schema_version: {config.schema_version}")

    choices = normalize_choices(obj[config.choices_key])
    answer = obj[config.answer_key]

    if isinstance(answer, str) and any(choice.label == answer for choice in choices):
        answer_label = answer
        answer_text = next(choice.text for choice in choices if choice.label == answer_label)
    elif isinstance(answer, int):
        answer_label = choices[answer].label
        answer_text = choices[answer].text
    else:
        answer_text = str(answer)
        matching = [choice for choice in choices if choice.text == answer_text]
        if len(matching) != 1:
            raise ValueError(
                "A text-valued gold answer must match exactly one choice; "
                f"matched {len(matching)} choices."
            )
        answer_label = matching[0].label

    return Example(
        id=legacy_record_id(obj, config.name),
        question=normalize_question(obj[config.question_key], config),
        choices=choices,
        answer_label=answer_label,
        answer_text=str(obj.get("answer_text", answer_text)),
        raw=obj,
    )


def read_jsonl_dataset(config: DatasetConfig) -> list[Example]:
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a JSONL path.")

    examples = []

    for line_no, obj in iter_jsonl_objects(config.path, config.name):
        try:
            examples.append(normalize_example(obj, config))
        except Exception as exc:
            raise ValueError(f"Failed to parse {config.name}:{line_no}: {exc}") from exc

    return examples


def is_s3_path(path: Path | str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def parse_s3_uri(path: str) -> tuple[str, str]:
    parsed = urlparse(path)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 dataset path: {path}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 dataset path must include an object key or prefix: {path}")
    return parsed.netloc, key


def s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("S3-backed datasets require boto3. Install bilgeval with boto3 available.") from exc

    endpoint_url = os.getenv("BILGEVAL_S3_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL")
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


@contextmanager
def open_s3_text(path: str):
    import io

    bucket, key = parse_s3_uri(path)
    response = s3_client().get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        with io.TextIOWrapper(body, encoding="utf-8") as file:
            yield file
    finally:
        body.close()


@contextmanager
def open_dataset_text(path: Path | str):
    if is_s3_path(path):
        with open_s3_text(str(path)) as file:
            yield file
        return
    with open(path, "r", encoding="utf-8") as file:
        yield file


def iter_jsonl_objects(path: Path | str, label: str):
    with open_dataset_text(path) as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except Exception as exc:
                raise ValueError(f"Failed to parse {label}:{line_no}: {exc}") from exc


def normalize_free_response_example(obj: dict[str, Any], config: DatasetConfig) -> FreeResponseExample:
    if config.schema_version == INSTANCE_SCHEMA_VERSION:
        instance = parse_canonical_instance(obj)
        if not isinstance(instance, FreeResponseInstance):
            raise TypeError(
                f"Expected a free_response instance, got {instance.task_type!r}."
            )
        primary_reference = instance.target.references[0]
        return FreeResponseExample(
            id=instance.id,
            question=instance.input.question,
            answer=primary_reference,
            answer_text=primary_reference,
            raw=obj,
        )

    if config.schema_version != "legacy":
        raise ValueError(f"Unsupported dataset schema_version: {config.schema_version}")

    answer = obj.get(config.answer_key, obj.get("answer"))
    if answer is None:
        raise ValueError("Free-response example is missing an answer.")
    answer_text = obj.get("answer_text", answer)
    if isinstance(answer_text, str) and "####" in answer_text:
        answer_text = answer_text.rsplit("####", 1)[1].strip()

    return FreeResponseExample(
        id=legacy_record_id(obj, config.name),
        question=normalize_question(obj[config.question_key], config),
        answer=str(answer),
        answer_text=str(answer_text),
        raw=obj,
    )


def read_free_response_dataset(config: DatasetConfig) -> list[FreeResponseExample]:
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a JSONL path.")

    examples = []

    for line_no, obj in iter_jsonl_objects(config.path, config.name):
        try:
            examples.append(normalize_free_response_example(obj, config))
        except Exception as exc:
            raise ValueError(f"Failed to parse {config.name}:{line_no}: {exc}") from exc

    return examples


def normalize_humaneval_example(
    obj: dict[str, Any],
    config: DatasetConfig,
) -> HumanEvalExample:
    if config.schema_version == INSTANCE_SCHEMA_VERSION:
        instance = parse_canonical_instance(obj)
        if not isinstance(instance, CodeGenerationInstance):
            raise TypeError(
                f"Expected a code_generation instance, got {instance.task_type!r}."
            )
        return HumanEvalExample(
            id=instance.id,
            prompt=instance.input.prompt,
            test=instance.target.test,
            entry_point=instance.target.entry_point,
            canonical_solution=instance.target.canonical_solution,
            raw=obj,
        )

    if config.schema_version != "legacy":
        raise ValueError(f"Unsupported dataset schema_version: {config.schema_version}")

    prompt = obj.get(config.question_key)
    test = obj.get("test")
    entry_point = obj.get("entry_point")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("HumanEval example is missing its prompt.")
    if not isinstance(test, str) or "def check(" not in test:
        raise ValueError("HumanEval example is missing its unit-test check function.")
    if not isinstance(entry_point, str) or not entry_point.strip():
        raise ValueError("HumanEval example is missing its entry_point.")
    return HumanEvalExample(
        id=str(obj.get("task_id") or obj.get("id") or ""),
        prompt=prompt,
        test=test,
        entry_point=entry_point,
        canonical_solution=(
            str(obj["canonical_solution"])
            if obj.get("canonical_solution") is not None
            else None
        ),
        raw=obj,
    )


def normalize_livecodebench_example(
    obj: dict[str, Any],
    config: DatasetConfig,
) -> LiveCodeBenchExample:
    if obj.get("schema_version") != "livecodebench.codegen/v1":
        raise ValueError(
            f"Unsupported LiveCodeBench schema_version: {obj.get('schema_version')!r}"
        )
    for key in ("id", "question_content", "tests"):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise ValueError(f"LiveCodeBench example is missing {key}.")
    test_type = obj.get("test_type")
    fn_name = obj.get("fn_name")
    if test_type not in {"stdin", "functional"}:
        raise ValueError(f"Unsupported LiveCodeBench test_type: {test_type!r}")
    if (test_type == "functional") != bool(fn_name):
        raise ValueError("LiveCodeBench test_type and fn_name disagree.")
    num_tests = obj.get("num_tests")
    if not isinstance(num_tests, int) or num_tests < 1:
        raise ValueError("LiveCodeBench example needs at least one test.")
    return LiveCodeBenchExample(
        id=obj["id"],
        question_title=str(obj.get("question_title") or ""),
        question_content=obj["question_content"],
        starter_code=str(obj.get("starter_code") or ""),
        fn_name=fn_name or None,
        num_tests=num_tests,
        tests=obj["tests"],
        raw={key: value for key, value in obj.items() if key != "tests"},
    )


def read_livecodebench_dataset(config: DatasetConfig) -> list[LiveCodeBenchExample]:
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a JSONL path.")
    examples = []
    for line_no, obj in iter_jsonl_objects(config.path, config.name):
        try:
            examples.append(normalize_livecodebench_example(obj, config))
        except Exception as exc:
            raise ValueError(f"Failed to parse {config.name}:{line_no}: {exc}") from exc
    return examples


def read_humaneval_dataset(config: DatasetConfig) -> list[HumanEvalExample]:
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a JSONL path.")
    examples = []
    for line_no, obj in iter_jsonl_objects(config.path, config.name):
        try:
            examples.append(normalize_humaneval_example(obj, config))
        except Exception as exc:
            raise ValueError(f"Failed to parse {config.name}:{line_no}: {exc}") from exc
    return examples


def normalize_function_call_example(
    obj: dict[str, Any],
    config: DatasetConfig,
) -> FunctionCallExample:
    if config.schema_version != INSTANCE_SCHEMA_VERSION:
        raise ValueError("Function-call datasets must use the canonical instance schema.")
    instance = parse_canonical_instance(obj)
    if not isinstance(instance, FunctionCallInstance):
        raise TypeError(
            f"Expected a function_call instance, got {instance.task_type!r}."
        )
    category = instance.metadata.get("bfcl_category", "")
    if not isinstance(category, str) or not category.strip():
        raise ValueError("Function-call examples must define metadata.bfcl_category.")
    return FunctionCallExample(
        id=instance.id,
        messages=[
            message.model_dump(mode="json")
            for message in instance.input.messages
        ],
        tools=[
            tool.model_dump(mode="json")
            for tool in instance.input.tools
        ],
        expected_calls=[
            call.model_dump(mode="json")
            for call in instance.target.calls
        ],
        should_call=instance.target.should_call,
        call_order=instance.target.call_order,
        scoring_mode=instance.target.scoring_mode,
        category=category,
        raw=obj,
    )


def read_function_call_dataset(
    config: DatasetConfig,
) -> list[FunctionCallExample]:
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a JSONL path.")
    examples = []
    for line_no, obj in iter_jsonl_objects(config.path, config.name):
        try:
            examples.append(normalize_function_call_example(obj, config))
        except Exception as exc:
            raise ValueError(f"Failed to parse {config.name}:{line_no}: {exc}") from exc
    return examples


def normalize_safety_behavior_example(
    obj: dict[str, Any],
    config: DatasetConfig,
) -> SafetyBehaviorExample:
    prompt = obj.get(config.question_key)
    subcategory = obj.get("subcategory")
    expected = obj.get(config.answer_key)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Safety behavior example is missing its prompt.")
    if not isinstance(subcategory, str) or not subcategory.strip():
        raise ValueError("Safety behavior example is missing its subcategory.")
    if expected not in {"must_refuse", "must_engage"}:
        raise ValueError(
            "Safety behavior example expected value must be 'must_refuse' or 'must_engage'."
        )
    return SafetyBehaviorExample(
        id=str(obj.get("prompt_id") or obj.get("id") or ""),
        prompt=prompt,
        subcategory=subcategory,
        expected=expected,
        raw=obj,
    )


def read_safety_behavior_dataset(config: DatasetConfig) -> list[SafetyBehaviorExample]:
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a JSONL path.")
    examples = []
    for line_no, obj in iter_jsonl_objects(config.path, config.name):
        try:
            examples.append(normalize_safety_behavior_example(obj, config))
        except Exception as exc:
            raise ValueError(f"Failed to parse {config.name}:{line_no}: {exc}") from exc
    return examples


def normalize_document_qa_example(
    obj: dict[str, Any],
    config: DatasetConfig,
) -> DocumentQAExample:
    document = obj.get("document")
    question = obj.get(config.question_key)
    answer = obj.get(config.answer_key)
    answer_format = obj.get("answer_format")
    task_type = obj.get("task_type")
    evidence_spans = obj.get("evidence_spans") or []
    if not isinstance(document, str) or not document.strip():
        raise ValueError("Document QA example is missing its document.")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Document QA example is missing its question.")
    if answer is None:
        raise ValueError("Document QA example is missing its answer.")
    if answer_format not in {
        "span",
        "boolean",
        "multiple_choice",
        "list",
        "free_form",
        "unanswerable",
    }:
        raise ValueError(f"Unsupported document QA answer_format: {answer_format!r}.")
    if not isinstance(task_type, str) or not task_type.strip():
        raise ValueError("Document QA example is missing its task_type.")
    if not isinstance(evidence_spans, list) or any(
        not isinstance(value, str) for value in evidence_spans
    ):
        raise ValueError("Document QA evidence_spans must be a list of strings.")
    return DocumentQAExample(
        id=str(obj.get("id") or ""),
        document=document,
        question=question,
        answer=answer,
        answer_format=answer_format,
        task_type=task_type,
        evidence_spans=evidence_spans,
        raw=obj,
    )


def read_document_qa_dataset(config: DatasetConfig) -> list[DocumentQAExample]:
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a JSONL path.")
    examples = []
    for line_no, obj in iter_jsonl_objects(config.path, config.name):
        try:
            examples.append(normalize_document_qa_example(obj, config))
        except Exception as exc:
            raise ValueError(f"Failed to parse {config.name}:{line_no}: {exc}") from exc
    return examples


def normalize_multi_turn_judge_example(
    obj: dict[str, Any],
    config: DatasetConfig,
) -> MultiTurnJudgeExample:
    dialogue_id = str(obj.get("id", "")).strip()
    task_id = str(obj.get("task_id", "")).strip()
    task_name = str(obj.get("task_name", "")).strip()
    topic = str(obj.get("topic", "")).strip()
    turns = obj.get("turns")
    if not dialogue_id or not task_id or not task_name or not topic:
        raise ValueError("MTBench dialogue requires id, task_id, task_name, and topic.")
    if not isinstance(turns, list) or not turns:
        raise ValueError("MTBench dialogue turns must be a non-empty list.")
    if obj.get("num_turns") != len(turns):
        raise ValueError("MTBench num_turns does not match the turns list.")

    normalized_turns = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"MTBench turn {index} must be an object.")
        role = turn.get("role")
        expected_role = "human" if index % 2 == 0 else "assistant"
        content = turn.get("content")
        if role != expected_role:
            raise ValueError(
                f"MTBench turn {index} must use role {expected_role!r}, got {role!r}."
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"MTBench turn {index} has empty content.")
        normalized_turns.append({"role": role, "content": content})

    assistant_count = sum(turn["role"] == "assistant" for turn in normalized_turns)
    eval_turn_positions = list(range(assistant_count))
    if bool(obj.get("skip_first_assistant")):
        eval_turn_positions = eval_turn_positions[1:]
    if not eval_turn_positions:
        raise ValueError("MTBench dialogue has no assistant turns selected for evaluation.")

    prompt_fields = {
        name: obj.get(name)
        for name in ("evaluation_initial", "evaluation_prompt", "scoring_format")
    }
    empty_prompts = [
        name
        for name, value in prompt_fields.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if empty_prompts:
        raise ValueError(
            "MTBench dialogue has missing judge prompt fields: "
            + ", ".join(empty_prompts)
        )

    return MultiTurnJudgeExample(
        id=dialogue_id,
        task_id=task_id,
        task_name=task_name,
        topic=topic,
        turns=normalized_turns,
        eval_turn_positions=eval_turn_positions,
        evaluation_initial=prompt_fields["evaluation_initial"],
        evaluation_prompt=prompt_fields["evaluation_prompt"],
        scoring_format=prompt_fields["scoring_format"],
        raw=obj,
    )


def read_multi_turn_judge_dataset(
    config: DatasetConfig,
) -> list[MultiTurnJudgeExample]:
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a JSONL path.")
    examples = []
    for line_no, obj in iter_jsonl_objects(config.path, config.name):
        try:
            examples.append(normalize_multi_turn_judge_example(obj, config))
        except Exception as exc:
            raise ValueError(f"Failed to parse {config.name}:{line_no}: {exc}") from exc
    return examples


def validate_dataset_records(
    config: DatasetConfig,
    max_records: int | None = None,
) -> dict[str, Any]:
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a JSONL path.")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be at least 1.")
    kind = dataset_type(config.task_type)
    if not kind.validate_records:
        return {
            "dataset": config.name,
            "status": "deferred",
            "task_type": config.task_type,
            "records_validated": 0,
        }

    count = 0
    seen_ids: set[str] = set()
    for line_no, obj in iter_jsonl_objects(config.path, config.name):
        if max_records is not None and count >= max_records:
            break
        try:
            example = RECORD_NORMALIZERS[kind.internal](obj, config)
        except Exception as exc:
            raise ValueError(f"Failed to validate {config.name}:{line_no}: {exc}") from exc
        if config.schema_version == INSTANCE_SCHEMA_VERSION or kind.id_required:
            if not example.id:
                raise ValueError(f"Failed to validate {config.name}:{line_no}: id must not be empty.")
            if example.id in seen_ids:
                raise ValueError(
                    f"Failed to validate {config.name}:{line_no}: duplicate id {example.id!r}."
                )
            seen_ids.add(example.id)
            if config.schema_version == INSTANCE_SCHEMA_VERSION:
                instance = parse_canonical_instance(obj)
                if config.split and instance.split != config.split:
                    raise ValueError(
                        f"Failed to validate {config.name}:{line_no}: record split "
                        f"{instance.split!r} does not match DatasetSpec split {config.split!r}."
                    )
                if config.language and instance.language != config.language:
                    raise ValueError(
                        f"Failed to validate {config.name}:{line_no}: record language "
                        f"{instance.language!r} does not match DatasetSpec language {config.language!r}."
                    )
        count += 1

    if count == 0:
        raise ValueError(f"Dataset {config.name} contains no records.")

    return {
        "dataset": config.name,
        "status": "valid",
        "task_type": config.task_type,
        "schema_version": config.schema_version,
        "records_validated": count,
        "limited": max_records is not None,
    }


def normalize_graphwalks_example(obj: dict[str, Any], idx: int, config: DatasetConfig) -> GraphWalksExample:
    answer_nodes = obj.get(config.answer_key, [])
    if answer_nodes is None:
        answer_nodes = []
    if not isinstance(answer_nodes, list):
        raise TypeError(f"GraphWalks answer_nodes must be a list: {answer_nodes}")

    return GraphWalksExample(
        id=str(obj.get("id", idx)),
        prompt=str(obj[config.question_key]),
        answer_nodes=[str(node) for node in answer_nodes],
        problem_type=str(obj.get("problem_type", "")),
        raw=obj,
    )


def iter_jsonl_rows(path: Path):
    for _, obj in iter_jsonl_objects(path, str(path)):
        yield obj


def iter_json_rows(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        yield from data
        return
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                yield from value
                return
        yield data
        return

    raise TypeError(f"Unsupported JSON dataset shape in {path}: {type(data)}")


def iter_csv_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield dict(row)


def iter_parquet_rows(path: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "GraphWalks parquet files require optional parquet support. "
            "Convert the downloaded files to JSONL/JSON/CSV, or install pyarrow outside bilgeval."
        ) from exc

    table = pq.read_table(path)
    for row in table.to_pylist():
        yield dict(row)


def iter_local_dataset_rows(path: Path):
    if path.is_file():
        files = [path]
    elif path.is_dir():
        all_files = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.suffix.lower() in {".jsonl", ".json", ".csv", ".parquet"}
        )
        lightweight_files = [
            candidate
            for candidate in all_files
            if candidate.suffix.lower() in {".jsonl", ".json", ".csv"}
        ]
        files = lightweight_files or all_files
    else:
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if not files:
        raise FileNotFoundError(
            f"No supported dataset files found under {path}. "
            "Supported extensions: .jsonl, .json, .csv, .parquet."
        )

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            yield from iter_jsonl_rows(file_path)
        elif suffix == ".json":
            yield from iter_json_rows(file_path)
        elif suffix == ".csv":
            yield from iter_csv_rows(file_path)
        elif suffix == ".parquet":
            yield from iter_parquet_rows(file_path)


def iter_s3_dataset_rows(path: str):
    bucket, key = parse_s3_uri(path)
    if key.lower().endswith(".jsonl"):
        for _, obj in iter_jsonl_objects(path, path):
            yield obj
        return

    prefix = key if key.endswith("/") else key + "/"
    client = s3_client()
    paginator = client.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            object_key = item["Key"]
            if object_key.lower().endswith(".jsonl"):
                files.append(object_key)

    if not files:
        raise FileNotFoundError(f"No JSONL dataset objects found under {path}.")

    for object_key in sorted(files):
        for _, obj in iter_jsonl_objects(f"s3://{bucket}/{object_key}", object_key):
            yield obj


def read_graphwalks_dataset(
    config: DatasetConfig,
    max_examples: int | None = None,
) -> list[GraphWalksExample]:
    if config.source not in {"local", "s3"}:
        raise ValueError(f"GraphWalks dataset {config.name} must use source='local' or source='s3'.")
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a path.")

    examples = []
    rows = iter_s3_dataset_rows(config.path) if config.source == "s3" else iter_local_dataset_rows(config.path)
    for idx, row in enumerate(rows):
        if max_examples is not None and len(examples) >= max_examples:
            break
        examples.append(normalize_graphwalks_example(obj=row, idx=idx, config=config))

    return examples


def normalize_oneruler_example(obj: dict[str, Any], idx: int, config: DatasetConfig) -> OneRulerExample:
    task = str(obj.get("task", ""))
    if task not in ONERULER_TASKS:
        raise ValueError(f"OneRuler row {idx} has an unknown task: {task!r}")
    outputs = obj.get(config.answer_key)
    if not isinstance(outputs, list) or not outputs:
        raise TypeError(f"OneRuler row {idx}: {config.answer_key} must be a non-empty list")
    return OneRulerExample(
        id=str(obj.get("id", idx)),
        prompt=str(obj[config.question_key]),
        outputs=[str(output) for output in outputs],
        task=task,
        raw=obj,
    )


def read_oneruler_dataset(config: DatasetConfig, max_examples: int | None = None) -> list[OneRulerExample]:
    if config.source != "local":
        raise ValueError(f"OneRuler dataset {config.name} must use source='local'.")
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a path.")

    examples = []
    for idx, row in enumerate(iter_local_dataset_rows(config.path)):
        if max_examples is not None and len(examples) >= max_examples:
            break
        examples.append(normalize_oneruler_example(obj=row, idx=idx, config=config))
    return examples


OFFICIAL_IFEVAL_SCHEMAS = frozenset({"google.ifeval/v1", "allenai.ifbench/v1"})


def normalize_official_ifeval_example(obj: dict[str, Any]) -> OfficialIFEvalExample:
    instruction_ids = obj.get("instruction_id_list")
    kwargs = obj.get("kwargs")
    if not isinstance(instruction_ids, list) or not instruction_ids:
        raise ValueError("IFEval example is missing instruction_id_list.")
    if not isinstance(kwargs, list) or len(kwargs) != len(instruction_ids):
        raise ValueError("IFEval example needs one kwargs entry per instruction.")
    if not isinstance(obj.get("prompt"), str) or not obj["prompt"].strip():
        raise ValueError("IFEval example is missing its prompt.")
    if obj.get("key") is None:
        raise ValueError("IFEval example is missing its key.")
    return OfficialIFEvalExample(
        id=str(obj["key"]),
        prompt=obj["prompt"],
        instruction_ids=[str(instruction_id) for instruction_id in instruction_ids],
        kwargs=[dict(item or {}) for item in kwargs],
        raw=obj,
    )


def normalize_ifeval_example(obj: dict[str, Any], idx: int, config: DatasetConfig) -> IFEvalExample:
    if config.schema_version in OFFICIAL_IFEVAL_SCHEMAS:
        return normalize_official_ifeval_example(obj)
    instruction_ids = obj.get("instruction_id_list")
    if not isinstance(instruction_ids, list) or not instruction_ids:
        raise ValueError("IFEval example is missing instruction_id_list.")

    return IFEvalExample(
        id=str(obj.get("prompt_id", obj.get("id", idx))),
        prompt=str(obj[config.question_key]),
        instruction_ids=[int(instruction_id) for instruction_id in instruction_ids],
        verifier_kwargs=dict(obj.get("verifier_kwargs") or {}),
        raw=obj,
    )


def read_ifeval_dataset(
    config: DatasetConfig,
    max_examples: int | None = None,
) -> list[IFEvalExample]:
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a path.")

    examples = []
    rows = iter_s3_dataset_rows(config.path) if config.source == "s3" else iter_jsonl_objects(config.path, config.name)
    for idx, row in enumerate(rows):
        if max_examples is not None and len(examples) >= max_examples:
            break
        obj = row[1] if isinstance(row, tuple) else row
        try:
            examples.append(normalize_ifeval_example(obj=obj, idx=idx, config=config))
        except Exception as exc:
            raise ValueError(f"Failed to parse {config.name}:{idx + 1}: {exc}") from exc

    return examples


def normalize_external_response(obj: dict[str, Any], line_no: int) -> ExternalResponse:
    dataset_name = obj.get("dataset_name")
    example_id = obj.get("id")
    candidate_response = obj.get("candidate_response")
    if candidate_response is None:
        candidate_response = obj.get("response")

    if not dataset_name:
        raise ValueError("External response is missing required field: dataset_name")
    if example_id is None:
        raise ValueError("External response is missing required field: id")
    if candidate_response is None:
        raise ValueError("External response is missing required field: candidate_response")

    generation_prompt_text = obj.get("generation_prompt_text")
    if generation_prompt_text is None:
        generation_prompt_text = obj.get("prompt")

    return ExternalResponse(
        dataset_name=str(dataset_name),
        id=str(example_id),
        candidate_response=str(candidate_response),
        generation_prompt_text=None if generation_prompt_text is None else str(generation_prompt_text),
        raw=obj,
    )


def read_external_responses(path: Path | str) -> dict[tuple[str, str], ExternalResponse]:
    response_path = Path(path)
    responses: dict[tuple[str, str], ExternalResponse] = {}

    with open(response_path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                response = normalize_external_response(obj=obj, line_no=line_no)
            except Exception as exc:
                raise ValueError(f"Failed to parse external responses {response_path}:{line_no}: {exc}") from exc

            key = (response.dataset_name, response.id)
            if key in responses:
                raise ValueError(
                    f"Duplicate external response for dataset_name={response.dataset_name!r}, id={response.id!r}"
                )
            responses[key] = response

    return responses


RECORD_NORMALIZERS = {
    "mc": normalize_example,
    "free_response": normalize_free_response_example,
    "humaneval": normalize_humaneval_example,
    "livecodebench": normalize_livecodebench_example,
    "function_call": normalize_function_call_example,
    "safety_behavior": normalize_safety_behavior_example,
    "document_qa": normalize_document_qa_example,
    "multi_turn_judge": normalize_multi_turn_judge_example,
}
