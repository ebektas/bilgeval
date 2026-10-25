"""The built-in catalog: task, dataset and suite definitions as data.

`bilgeval/catalog/registry.yaml` names the registry version, the model
profiles and the family files to compile, in order. Each family file holds
`datasets`, `tasks` and `suites` maps plus optional per-kind `defaults`
(applied when an entry omits a key). Entries are sparse: the pydantic
models of `registry_bundle.py` fill the rest and fix the key order, so the
compiled `builtin_registry.json` is a pure function of the YAML.

Derived at build time, and only these: a dataset's `content_hash` (sha256 of
the file, or of a directory's files), its `version` when omitted
(`content-<12 hex>`), and a check of a declared `row_count`.

    python scripts/build_builtin_registry.py        # rewrite builtin_registry.json
    python scripts/build_builtin_registry.py --check  # is the committed file current?

PyYAML is needed to build; the runtime reads only the compiled JSON.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from .manifest import stable_json_sha256
from .registry_bundle import REGISTRY_SCHEMA_VERSION, RegistryBundle


PACKAGE_ROOT = Path(__file__).resolve().parent
CATALOG_DIR = PACKAGE_ROOT / "catalog"
BUILTIN_REGISTRY = PACKAGE_ROOT / "builtin_registry.json"
KINDS = ("datasets", "tasks", "suites")


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    """Hash of a directory: the stable JSON hash of its files' relative paths
    and hashes (the form the dataset versions were first computed with)."""
    entries = [
        {"path": item.relative_to(path).as_posix(), "content_hash": "sha256:" + sha256_file(item)}
        for item in sorted(item for item in path.rglob("*") if item.is_file())
    ]
    if not entries:
        raise ValueError(f"Dataset directory is empty: {path}")
    return stable_json_sha256(entries).removeprefix("sha256:")


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def _with_defaults(entry: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    value = dict(entry)
    for key, default in defaults.items():
        value.setdefault(key, default)
    return value


def _complete_dataset(name: str, entry: dict[str, Any], root: Path) -> dict[str, Any]:
    value = dict(entry)
    uri = value.get("uri")
    if not isinstance(uri, str) or not uri:
        raise ValueError(f"dataset {name!r}: uri is required")
    if not uri.startswith("s3://"):
        path = root / uri
        if not path.exists():
            raise ValueError(f"dataset {name!r}: {uri} does not exist")
        content_hash = "sha256:" + (sha256_directory(path) if path.is_dir() else sha256_file(path))
        declared = value.get("content_hash")
        if declared is not None and declared != content_hash:
            raise ValueError(
                f"dataset {name!r}: declared content_hash {declared} but the file hashes to {content_hash}"
            )
        value["content_hash"] = content_hash
        if value.get("row_count") is not None and path.is_file():
            actual = count_rows(path)
            if actual != value["row_count"]:
                raise ValueError(
                    f"dataset {name!r}: row_count {value['row_count']} declared, {actual} rows in {uri}"
                )
    elif value.get("content_hash") is None:
        raise ValueError(f"dataset {name!r}: an s3 dataset needs an explicit content_hash")
    if value.get("version") is None:
        value["version"] = "content-" + value["content_hash"].removeprefix("sha256:")[:12]
    return value


def load_catalog(catalog_dir: Path = CATALOG_DIR, dataset_root: Path = PACKAGE_ROOT) -> RegistryBundle:
    """Compile a catalog directory; dataset uris are resolved against dataset_root."""
    index = _read_yaml(catalog_dir / "registry.yaml")
    collections: dict[str, dict[str, Any]] = {kind: {} for kind in KINDS}
    for file_name in index.get("files") or []:
        document = _read_yaml(catalog_dir / file_name)
        defaults = document.get("defaults") or {}
        for kind in KINDS:
            for name, entry in (document.get(kind) or {}).items():
                if name in collections[kind]:
                    raise ValueError(f"{file_name}: {kind[:-1]} {name!r} is defined twice")
                value = _with_defaults(entry or {}, defaults.get(kind) or {})
                if kind == "datasets":
                    value = _complete_dataset(name, value, dataset_root)
                collections[kind][name] = value
    bundle = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "registry_version": index["registry_version"],
        **collections,
        "generation_prompts": index.get("generation_prompts") or {},
        "judge_prompts": index.get("judge_prompts") or {},
        "model_profiles": index.get("model_profiles") or {},
        "metadata": index.get("metadata") or {},
    }
    return RegistryBundle.model_validate(bundle)


def render_registry(bundle: RegistryBundle) -> str:
    """The exact text of builtin_registry.json for a bundle."""
    return json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"


def write_registry_bundle(bundle: RegistryBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_registry(bundle), encoding="utf-8")


def build_builtin_registry(output_path: Path = BUILTIN_REGISTRY) -> RegistryBundle:
    bundle = load_catalog()
    write_registry_bundle(bundle, output_path)
    return bundle


def check_builtin_registry(path: Path = BUILTIN_REGISTRY) -> bool:
    return path.read_text(encoding="utf-8") == render_registry(load_catalog())
