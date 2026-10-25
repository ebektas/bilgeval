import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from bilgeval.conversion import convert_legacy_dataset
from bilgeval.reader import normalize_example, read_free_response_dataset, read_jsonl_dataset
from bilgeval.registry import DatasetConfig
from bilgeval.schema import INSTANCE_SCHEMA_VERSION, parse_canonical_instance


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class CanonicalSchemaTests(unittest.TestCase):
    def test_multiple_choice_schema_rejects_unknown_and_inconsistent_fields(self):
        value = json.loads((FIXTURES / "canonical_mc.jsonl").read_text(encoding="utf-8").splitlines()[0])
        parsed = parse_canonical_instance(value)
        self.assertEqual(parsed.target.answer_label, "B")

        value["unknown"] = True
        with self.assertRaises(ValidationError):
            parse_canonical_instance(value)

        value.pop("unknown")
        value["target"]["answer_text"] = "3"
        with self.assertRaises(ValidationError):
            parse_canonical_instance(value)

    def test_canonical_reader_adapts_both_supported_types(self):
        mc_config = DatasetConfig(
            name="fixture_mc",
            path=FIXTURES / "canonical_mc.jsonl",
            schema_version=INSTANCE_SCHEMA_VERSION,
            task_type="mc",
        )
        free_response_config = DatasetConfig(
            name="fixture_free_response",
            path=FIXTURES / "canonical_free_response.jsonl",
            schema_version=INSTANCE_SCHEMA_VERSION,
            task_type="free_response",
        )

        mc_examples = read_jsonl_dataset(mc_config)
        free_response_examples = read_free_response_dataset(free_response_config)

        self.assertEqual([example.id for example in mc_examples], ["mc-1", "mc-2"])
        self.assertEqual(mc_examples[0].answer_text, "4")
        self.assertEqual(free_response_examples[0].answer_text, "8")

    def test_legacy_conversion_is_deterministic_and_preserves_metadata(self):
        source_record = {
            "id": "legacy-1",
            "question translated": "Hangisi dogrudur?",
            "choices translated": ["Yanlis", "Dogru"],
            "answer": 1,
            "category": "fixture",
        }
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "legacy.jsonl"
            output_path = directory_path / "canonical.jsonl"
            source_path.write_text(json.dumps(source_record) + "\n", encoding="utf-8")
            config = DatasetConfig(
                name="legacy_fixture",
                path=source_path,
                question_key="question translated",
                choices_key="choices translated",
                answer_key="answer",
            )

            first = convert_legacy_dataset(config, output_path, split="test", language="tr")
            first_bytes = output_path.read_bytes()
            second = convert_legacy_dataset(config, output_path, split="test", language="tr")

            self.assertEqual(first.content_hash, second.content_hash)
            self.assertEqual(first_bytes, output_path.read_bytes())
            converted = parse_canonical_instance(
                json.loads(output_path.read_text(encoding="utf-8"))
            )
            self.assertEqual(converted.target.answer_label, "B")
            self.assertEqual(converted.metadata, {"category": "fixture"})

    def test_legacy_reader_rejects_duplicate_labels_and_ambiguous_gold_text(self):
        config = DatasetConfig(name="legacy_fixture", path=Path("unused.jsonl"))
        duplicate_labels = {
            "question": "Test?",
            "choices": [
                {"label": "A", "text": "One"},
                {"label": "A", "text": "Two"},
            ],
            "answer": "A",
        }
        with self.assertRaisesRegex(ValueError, "labels must be unique"):
            normalize_example(duplicate_labels, config)

        ambiguous_answer = {
            "question": "Test?",
            "choices": [
                {"label": "A", "text": "Same"},
                {"label": "B", "text": "Same"},
            ],
            "answer": "Same",
        }
        with self.assertRaisesRegex(ValueError, "match exactly one choice"):
            normalize_example(ambiguous_answer, config)


if __name__ == "__main__":
    unittest.main()
