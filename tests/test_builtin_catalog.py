"""The built-in registry is compiled from bilgeval/catalog/*.yaml."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

from bilgeval import catalog
from bilgeval.manifest import stable_json_sha256


def write_catalog(root: Path, dataset_entry: dict, dataset_text: str = '{"a": 1}\n{"a": 2}\n') -> Path:
    (root / "datasets").mkdir()
    (root / "datasets" / "rows.jsonl").write_text(dataset_text, encoding="utf-8")
    catalog_dir = root / "catalog"
    catalog_dir.mkdir()
    (catalog_dir / "registry.yaml").write_text(
        yaml.safe_dump({"registry_version": "test-1.0.0", "files": ["family.yaml"]}), encoding="utf-8"
    )
    (catalog_dir / "family.yaml").write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "rows": {
                        "uri": "datasets/rows.jsonl",
                        "schema_version": "legacy",
                        "task_type": "free_response",
                        "split": "test",
                        "language": "und",
                        **dataset_entry,
                    }
                },
                "tasks": {
                    "rows": {
                        "version": "1.0.0",
                        "dataset": "rows",
                        "formatter": {"id": "none", "version": "builtin"},
                        "scorer": {"id": "free_response_exact_match"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return catalog_dir


@unittest.skipUnless(yaml, "pyyaml is a dev dependency")
class BuiltinCatalogTests(unittest.TestCase):
    def test_committed_registry_is_the_compiled_catalog(self):
        self.assertTrue(
            catalog.check_builtin_registry(),
            "builtin_registry.json is stale: run scripts/build_builtin_registry.py",
        )

    def test_render_is_the_committed_writer(self):
        bundle = catalog.load_catalog()
        text = catalog.render_registry(bundle)
        self.assertTrue(text.endswith("}\n"))
        self.assertEqual(json.loads(text)["registry_version"], bundle.registry_version)

    def test_every_dataset_is_read_by_a_task(self):
        bundle = catalog.load_catalog()
        used = {task.dataset for task in bundle.tasks.values()}
        used |= {task.few_shot.dataset for task in bundle.tasks.values() if task.few_shot and task.few_shot.dataset}
        self.assertEqual(set(bundle.datasets) - used, set())

    def test_every_suite_task_exists(self):
        bundle = catalog.load_catalog()
        for suite_name, suite in bundle.suites.items():
            for item in suite.tasks:
                self.assertIn(item.task, bundle.tasks, f"suite {suite_name}: unknown task {item.task}")

    def test_directory_dataset_hash_is_the_stable_json_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b").mkdir()
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "b" / "c.txt").write_text("gamma", encoding="utf-8")
            expected = stable_json_sha256(
                [
                    {"path": "a.txt", "content_hash": "sha256:" + catalog.sha256_file(root / "a.txt")},
                    {"path": "b/c.txt", "content_hash": "sha256:" + catalog.sha256_file(root / "b" / "c.txt")},
                ]
            ).removeprefix("sha256:")
            self.assertEqual(catalog.sha256_directory(root), expected)

    def test_content_hash_and_version_are_derived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = catalog.load_catalog(write_catalog(root, {"row_count": 2}), dataset_root=root)
            dataset = bundle.datasets["rows"]
            digest = catalog.sha256_file(root / "datasets" / "rows.jsonl")
            self.assertEqual(dataset.content_hash, "sha256:" + digest)
            self.assertEqual(dataset.version, "content-" + digest[:12])
            self.assertEqual(bundle.registry_version, "test-1.0.0")

    def test_declared_hash_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_dir = write_catalog(root, {"content_hash": "sha256:" + "0" * 64})
            with self.assertRaisesRegex(ValueError, "declared content_hash"):
                catalog.load_catalog(catalog_dir, dataset_root=root)

    def test_declared_row_count_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_dir = write_catalog(root, {"row_count": 3})
            with self.assertRaisesRegex(ValueError, "row_count 3 declared, 2 rows"):
                catalog.load_catalog(catalog_dir, dataset_root=root)

    def test_names_are_defined_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_dir = write_catalog(root, {})
            (catalog_dir / "registry.yaml").write_text(
                yaml.safe_dump({"registry_version": "test-1.0.0", "files": ["family.yaml", "family.yaml"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "defined twice"):
                catalog.load_catalog(catalog_dir, dataset_root=root)


if __name__ == "__main__":
    unittest.main()
