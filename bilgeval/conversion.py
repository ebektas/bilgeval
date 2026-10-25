import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reader import (
    iter_jsonl_objects,
    normalize_example,
    normalize_free_response_example,
    normalize_humaneval_example,
)
from .registry import DatasetConfig
from .schema import (
    CodeGenerationInput,
    CodeGenerationInstance,
    CodeGenerationTarget,
    INSTANCE_SCHEMA_VERSION,
    ChoiceRecord,
    FreeResponseInput,
    FreeResponseInstance,
    FreeResponseTarget,
    MultipleChoiceInput,
    MultipleChoiceInstance,
    MultipleChoiceTarget,
    Provenance,
)


CONVERTER_VERSION = "bilgeval.legacy-jsonl/v1"


@dataclass(frozen=True)
class ConversionResult:
    dataset: str
    output: str
    row_count: int
    content_hash: str
    schema_version: str = INSTANCE_SCHEMA_VERSION


def json_sha256(value: dict[str, Any]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(serialized).hexdigest()


def source_metadata(obj: dict[str, Any], consumed_keys: set[str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in obj.items()
        if key not in consumed_keys
    }


def convert_legacy_object(
    obj: dict[str, Any],
    config: DatasetConfig,
    line_no: int,
    split: str,
    language: str,
):
    source_record_hash = json_sha256(obj)
    source_id = str(obj.get("id") or obj.get("task_id") or "")
    instance_id = source_id or f"{config.name}:{line_no}"
    provenance = Provenance(
        source=config.name,
        source_record_id=source_id or None,
        source_record_sha256=source_record_hash,
        converter=CONVERTER_VERSION,
    )

    if config.task_type in {"mc", "mc_logprob"}:
        example = normalize_example(obj, config)
        metadata = source_metadata(
            obj,
            {"id", config.question_key, config.choices_key, config.answer_key, "answer_text"},
        )
        return MultipleChoiceInstance(
            schema_version=INSTANCE_SCHEMA_VERSION,
            id=instance_id,
            task_type="multiple_choice",
            split=split,
            language=language,
            input=MultipleChoiceInput(
                question=example.question,
                choices=[
                    ChoiceRecord(label=choice.label, text=choice.text)
                    for choice in example.choices
                ],
            ),
            target=MultipleChoiceTarget(
                answer_label=example.answer_label,
                answer_text=example.answer_text,
            ),
            metadata=metadata,
            provenance=provenance,
        )

    if config.task_type == "free_response":
        example = normalize_free_response_example(obj, config)
        metadata = source_metadata(
            obj,
            {"id", config.question_key, config.answer_key, "answer_text"},
        )
        if example.answer != example.answer_text:
            metadata["source_answer"] = example.answer
        return FreeResponseInstance(
            schema_version=INSTANCE_SCHEMA_VERSION,
            id=instance_id,
            task_type="free_response",
            split=split,
            language=language,
            input=FreeResponseInput(question=example.question),
            target=FreeResponseTarget(references=[example.answer_text]),
            metadata=metadata,
            provenance=provenance,
        )

    if config.task_type == "humaneval":
        example = normalize_humaneval_example(obj, config)
        metadata = source_metadata(
            obj,
            {
                "id",
                "task_id",
                config.question_key,
                "test",
                "entry_point",
                "canonical_solution",
            },
        )
        return CodeGenerationInstance(
            schema_version=INSTANCE_SCHEMA_VERSION,
            id=instance_id,
            task_type="code_generation",
            split=split,
            language=language,
            input=CodeGenerationInput(prompt=example.prompt),
            target=CodeGenerationTarget(
                test=example.test,
                entry_point=example.entry_point,
                canonical_solution=example.canonical_solution,
            ),
            metadata=metadata,
            provenance=provenance,
        )

    raise ValueError(
        f"Dataset {config.name} has deferred task_type={config.task_type!r}; "
        "only multiple-choice and ordinary free-response conversion is implemented."
    )


def convert_legacy_dataset(
    config: DatasetConfig,
    output_path: Path,
    split: str,
    language: str,
) -> ConversionResult:
    if config.schema_version != "legacy":
        raise ValueError(f"Dataset {config.name} is already canonical.")
    if config.path is None:
        raise ValueError(f"Dataset {config.name} does not define a source path.")
    if config.source not in {"jsonl", "s3"}:
        raise ValueError(
            f"Dataset {config.name} uses source={config.source!r}; only JSONL conversion is supported."
        )
    if isinstance(config.path, Path) and output_path.resolve() == config.path.resolve():
        raise ValueError("Canonical output path must not overwrite the legacy source dataset.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_hash = hashlib.sha256()
    row_count = 0
    seen_ids: set[str] = set()
    try:
        with temporary_path.open("wb") as output_file:
            for line_no, obj in iter_jsonl_objects(config.path, config.name):
                instance = convert_legacy_object(
                    obj=obj,
                    config=config,
                    line_no=line_no,
                    split=split,
                    language=language,
                )
                if instance.id in seen_ids:
                    raise ValueError(f"Duplicate canonical id {instance.id!r} at source line {line_no}.")
                seen_ids.add(instance.id)
                line = (
                    json.dumps(
                        instance.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                output_file.write(line)
                output_hash.update(line)
                row_count += 1
        if row_count == 0:
            raise ValueError(f"Dataset {config.name} contains no records.")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return ConversionResult(
        dataset=config.name,
        output=str(output_path.resolve()),
        row_count=row_count,
        content_hash="sha256:" + output_hash.hexdigest(),
    )
