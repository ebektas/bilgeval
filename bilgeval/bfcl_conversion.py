import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .conversion import json_sha256
from .schema import (
    INSTANCE_SCHEMA_VERSION,
    FunctionCallExpectation,
    FunctionCallInput,
    FunctionCallInstance,
    FunctionCallTarget,
    FunctionDefinition,
    FunctionToolRecord,
    MessageRecord,
    Provenance,
)


BFCL_CONVERTER_VERSION = "bilgeval.bfcl-v4-single-turn/v2"
BFCL_SOURCE = "berkeley-function-calling-leaderboard/BFCL_v4"
BFCL_TR_SOURCE = "bilge-function-call/BFCL_v4_tr"
BFCL_EN_CORE_CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)
BFCL_TR_CORE_CATEGORIES = (
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)
BFCL_EN_LIVE_CATEGORIES = (
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
)


@dataclass(frozen=True)
class BFCLConversionResult:
    category: str
    output: str
    row_count: int
    content_hash: str


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid BFCL JSON at {path}:{line_no}: {exc}") from exc
            row_id = str(row.get("id", ""))
            if not row_id:
                raise ValueError(f"BFCL row at {path}:{line_no} has no id.")
            if row_id in rows:
                raise ValueError(f"Duplicate BFCL id {row_id!r} in {path}.")
            rows[row_id] = row
    if not rows:
        raise ValueError(f"BFCL source file is empty: {path}")
    return rows


def normalize_json_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {
        key: normalize_json_schema(item)
        for key, item in value.items()
    }
    type_mapping = {
        "integer": "integer",
        "number": "number",
        "float": "number",
        "string": "string",
        "boolean": "boolean",
        "bool": "boolean",
        "array": "array",
        "list": "array",
        "dict": "object",
        "object": "object",
        "tuple": "array",
        "any": "string",
        "byte": "integer",
        "short": "integer",
        "long": "integer",
        "double": "number",
        "char": "string",
        "ArrayList": "array",
        "Array": "array",
        "HashMap": "object",
        "Hashtable": "object",
        "Queue": "array",
        "Stack": "array",
        "Any": "string",
        "String": "string",
        "Bigint": "integer",
        "Boolean": "boolean",
    }
    schema_type = normalized.get("type")
    if isinstance(schema_type, str) and schema_type in type_mapping:
        normalized["type"] = type_mapping[schema_type]
    return normalized


def convert_tool(raw: dict[str, Any]) -> FunctionToolRecord:
    name = raw.get("name")
    parameters = raw.get("parameters")
    if not isinstance(name, str) or not name:
        raise ValueError("BFCL function documentation is missing its name.")
    if not isinstance(parameters, dict):
        raise ValueError(f"BFCL function {name!r} is missing parameters.")
    normalized_parameters = normalize_json_schema(parameters)
    normalized_parameters.setdefault("type", "object")
    normalized_parameters.setdefault("properties", {})
    normalized_parameters.setdefault("required", [])
    return FunctionToolRecord(
        function=FunctionDefinition(
            name=name,
            description=str(raw.get("description") or ""),
            parameters=normalized_parameters,
        )
    )


def convert_messages(raw_question: Any) -> list[MessageRecord]:
    if (
        not isinstance(raw_question, list)
        or len(raw_question) != 1
        or not isinstance(raw_question[0], list)
        or not raw_question[0]
    ):
        raise ValueError(
            "The initial BFCL converter only accepts one non-empty conversation turn."
        )
    messages = []
    for raw_message in raw_question[0]:
        if not isinstance(raw_message, dict):
            raise ValueError("BFCL messages must be objects.")
        messages.append(
            MessageRecord(
                role=raw_message.get("role", "user"),
                content=str(raw_message.get("content") or ""),
            )
        )
    return messages


def convert_expected_calls(
    raw_ground_truth: Any,
    tools: list[FunctionToolRecord],
) -> list[FunctionCallExpectation]:
    if not isinstance(raw_ground_truth, list):
        raise ValueError("BFCL ground_truth must be a list.")
    calls = []
    for raw_call in raw_ground_truth:
        if not isinstance(raw_call, dict) or len(raw_call) != 1:
            raise ValueError("Each BFCL ground-truth call must contain one function.")
        name, arguments = next(iter(raw_call.items()))
        if not isinstance(arguments, dict):
            raise ValueError(f"BFCL ground truth for {name!r} must contain arguments.")
        accepted_arguments = {}
        for argument_name, accepted_values in arguments.items():
            if not isinstance(accepted_values, list):
                raise ValueError(
                    f"BFCL argument {name}.{argument_name} must define accepted values."
                )
            if not accepted_values:
                tool = next(
                    (tool for tool in tools if tool.function.name == name),
                    None,
                )
                properties = (
                    tool.function.parameters.get("properties", {})
                    if tool is not None
                    else {}
                )
                schema_type = properties.get(argument_name, {}).get("type")
                if schema_type != "array":
                    raise ValueError(
                        f"BFCL argument {name}.{argument_name} has an empty target "
                        f"for non-array schema type {schema_type!r}."
                    )
                accepted_values = [[]]
            accepted_arguments[str(argument_name)] = accepted_values
        calls.append(
            FunctionCallExpectation(
                name=str(name),
                arguments=accepted_arguments,
            )
        )
    return calls


def convert_bfcl_category(
    *,
    question_path: Path,
    answer_path: Path | None,
    output_path: Path,
    category: str,
    source_revision: str,
    source: str = BFCL_SOURCE,
    language: str = "en",
    split: str = "test",
) -> BFCLConversionResult:
    questions = load_jsonl_by_id(question_path)
    answers = load_jsonl_by_id(answer_path) if answer_path is not None else {}
    if answer_path is not None and set(questions) != set(answers):
        missing = sorted(set(questions) - set(answers))
        extra = sorted(set(answers) - set(questions))
        raise ValueError(
            f"BFCL question/answer ids differ for {category}: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_hash = hashlib.sha256()
    row_count = 0
    try:
        with temporary_path.open("wb") as output:
            for row_id, question in questions.items():
                answer = answers.get(row_id)
                tools = [
                    convert_tool(tool)
                    for tool in question.get("function", [])
                ]
                calls = (
                    convert_expected_calls(answer.get("ground_truth"), tools)
                    if answer is not None
                    else []
                )
                scoring_mode = (
                    "call_presence"
                    if category in {"live_irrelevance", "live_relevance"}
                    else "exact"
                )
                should_call = (
                    category == "live_relevance"
                    if scoring_mode == "call_presence"
                    else bool(calls)
                )
                source_value = {
                    "question": question,
                    "possible_answer": answer,
                }
                instance = FunctionCallInstance(
                    schema_version=INSTANCE_SCHEMA_VERSION,
                    id=row_id,
                    task_type="function_call",
                    split=split,
                    language=language,
                    input=FunctionCallInput(
                        messages=convert_messages(question.get("question")),
                        tools=tools,
                    ),
                    target=FunctionCallTarget(
                        calls=calls,
                        should_call=should_call,
                        call_order="unordered",
                        scoring_mode=scoring_mode,
                    ),
                    metadata={
                        "benchmark": "BFCL_v4",
                        "bfcl_category": category,
                    },
                    provenance=Provenance(
                        source=source,
                        source_revision=source_revision,
                        source_record_id=row_id,
                        source_record_sha256=json_sha256(source_value),
                        converter=BFCL_CONVERTER_VERSION,
                    ),
                )
                line = (
                    json.dumps(
                        instance.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                output.write(line)
                output_hash.update(line)
                row_count += 1
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return BFCLConversionResult(
        category=category,
        output=str(output_path.resolve()),
        row_count=row_count,
        content_hash="sha256:" + output_hash.hexdigest(),
    )


def convert_bfcl_core(
    *,
    source_root: Path,
    output_root: Path,
    source_revision: str,
    language: str = "en",
    source: str | None = None,
    categories: tuple[str, ...] | None = None,
) -> list[BFCLConversionResult]:
    if categories is None:
        categories = (
            BFCL_TR_CORE_CATEGORIES
            if language == "tr"
            else BFCL_EN_CORE_CATEGORIES
        )
    if source is None:
        source = BFCL_TR_SOURCE if language == "tr" else BFCL_SOURCE

    results = []
    for category in categories:
        question_path = source_root / language / f"BFCL_v4_{category}.json"
        answer_candidate = (
            source_root
            / language
            / "possible_answer"
            / f"BFCL_v4_{category}.json"
        )
        results.append(
            convert_bfcl_category(
                question_path=question_path,
                answer_path=answer_candidate if answer_candidate.exists() else None,
                output_path=output_root / f"bfcl_v4_{category}.jsonl",
                category=category,
                source_revision=source_revision,
                source=source,
                language=language,
            )
        )
    return results
