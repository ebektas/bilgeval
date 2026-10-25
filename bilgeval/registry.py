"""Runtime dataset configs: what the readers need to open a registered dataset.

`DATASETS` is filled by `registry_bundle.install_registry_bundle` from the
active `bilgeval.registry/v1` bundle; importing `bilgeval.main` installs the
built-in registry. Task definitions live only in the registry.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DatasetSource = Literal["jsonl", "local", "s3"]

PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path | str | None = None
    source: DatasetSource = "jsonl"
    question_key: str = "question"
    choices_key: str = "choices"
    answer_key: str = "answer"
    task_type: str = "mc"
    question_only_before_newline: bool = False
    version: str = "legacy"
    schema_version: str = "legacy"
    content_hash: str | None = None
    split: str | None = None
    language: str | None = None
    row_count: int | None = None


def _resolve_registry_path(value: str | None, registry_file: Path) -> Path | str | None:
    if value is None:
        return None
    if value.startswith("s3://"):
        return value
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        registry_file.parent / path,
        PACKAGE_ROOT.parent / path,
        PACKAGE_ROOT / path,
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


DATASETS: dict[str, DatasetConfig] = {}
