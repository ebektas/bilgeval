import json
import unittest
from pathlib import Path

from bilgeval.code_execution import (
    CodeExecutionResult,
    build_humaneval_program,
    extract_python_code,
)
from bilgeval.evaluators import (
    GeneratedTextResult,
    TextGenerationModelClient,
    evaluate_humaneval,
)
from bilgeval.reader import HumanEvalExample, read_humaneval_dataset
import bilgeval.main
from bilgeval.registry import DATASETS
from bilgeval.schema import parse_canonical_instance


class PassGenerationClient(TextGenerationModelClient):
    """One text per request, in order: one example, so one text per sample."""

    def __init__(self, values):
        self.values = iter(values)

    def generate_texts(self, prompts, **kwargs):
        return [GeneratedTextResult(text=next(self.values)) for _ in prompts]


class PythonSyntaxFixtureExecutor:
    name = "fixture"

    def execute(self, program, timeout_seconds):
        passed = "return value + 1" in program
        return CodeExecutionResult(
            status="passed" if passed else "failed",
            elapsed_ms=1,
            exit_code=0 if passed else 1,
        )


class HumanEvalTests(unittest.TestCase):

    def test_canonical_code_generation_schema(self):
        value = {
            "schema_version": "bilgeval.instance/v1",
            "id": "HumanEval/0",
            "task_type": "code_generation",
            "split": "test",
            "language": "tr",
            "input": {"prompt": "def add_one(value):\n"},
            "target": {
                "test": "def check(candidate):\n    assert candidate(1) == 2\n",
                "entry_point": "add_one",
                "canonical_solution": "    return value + 1\n",
            },
            "metadata": {},
            "provenance": {"source": "openai/human-eval"},
        }
        parsed = parse_canonical_instance(value)
        self.assertEqual(parsed.target.entry_point, "add_one")

    def test_code_extraction_accepts_chat_code_fences_and_completion_suffixes(self):
        fenced = "Explanation\n```python\ndef add_one(value):\n    return value + 1\n```"
        self.assertEqual(
            extract_python_code(fenced),
            "def add_one(value):\n    return value + 1",
        )
        program, completion = build_humaneval_program(
            "def add_one(value):\n",
            "    return value + 1\n",
            "def check(candidate):\n    assert candidate(1) == 2\n",
            "add_one",
        )
        self.assertEqual(completion, "    return value + 1")
        self.assertIn("def add_one(value):\n    return value + 1", program)
        self.assertIn("check(add_one)", program)

    def test_humaneval_reports_unit_test_accuracy_and_pass_at_k(self):
        example = HumanEvalExample(
            id="HumanEval/fixture",
            prompt="def add_one(value):\n",
            test="def check(candidate):\n    assert candidate(1) == 2\n",
            entry_point="add_one",
            canonical_solution="    return value + 1\n",
            raw={},
        )
        report = evaluate_humaneval(
            [example],
            PassGenerationClient(["    return value\n", "    return value + 1\n"]),
            PythonSyntaxFixtureExecutor(),
            generation_settings={"temperature": 0.7},
            num_samples=2,
            pass_at_k=(1, 2),
        )

        self.assertEqual(report["num_scored_examples"], 1)
        self.assertEqual(report["correct"], 0)
        self.assertAlmostEqual(
            report["multi_sample_metrics"]["pass_at_k"]["1"],
            0.5,
        )
        self.assertEqual(report["multi_sample_metrics"]["pass_at_k"]["2"], 1.0)
        self.assertEqual(
            report["code_execution"]["candidate_status_counts"],
            {"failed": 1, "passed": 1},
        )


if __name__ == "__main__":
    unittest.main()


def test_full_definition_keeps_prompt_imports_and_helpers():
    from bilgeval.code_execution import build_humaneval_program

    prompt = (
        "from typing import List\nimport math\n\n\ndef helper(x):\n    return x\n\n\n"
        "def target(values: List[int]) -> int:\n    \"\"\"doc\"\"\"\n"
    )
    candidate = "```python\ndef target(values: List[int]) -> int:\n    return helper(len(values))\n```"
    program, _ = build_humaneval_program(
        prompt=prompt,
        candidate_response=candidate,
        test="def check(candidate):\n    assert candidate([1, 2]) == 2\n",
        entry_point="target",
    )
    assert program.startswith("from typing import List\nimport math\n")
    assert program.count("def target(") == 1
    namespace = {}
    exec(program, namespace)
