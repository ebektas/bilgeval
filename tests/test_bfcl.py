import json
import tempfile
import unittest
from pathlib import Path

import bilgeval
from pydantic import ValidationError

from bilgeval.bfcl import (
    parse_functiongemma_tool_calls,
    score_function_call_presence,
    score_function_calls,
)
from bilgeval.bfcl_conversion import (
    BFCL_TR_SOURCE,
    convert_bfcl_category,
    normalize_json_schema,
)
from bilgeval.evaluators import (
    GeneratedToolCallResult,
    OpenAICompatibleToolCallClient,
    ToolCallModelClient,
    evaluate_function_calls,
)
from bilgeval.reader import FunctionCallExample
from bilgeval.registry_bundle import load_registry_bundle
from bilgeval.schema import parse_canonical_instance


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather.lookup",
            "description": "Look up weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_lookup",
            "description": "Look up a date.",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
            },
        },
    },
]


def example(
    *,
    expected_calls=None,
    should_call=True,
    call_order="unordered",
    scoring_mode="exact",
) -> FunctionCallExample:
    expected_calls = expected_calls if expected_calls is not None else [
        {
            "name": "weather.lookup",
            "arguments": {
                "city": ["New York, NY"],
                "unit": ["celsius", ""],
            },
        }
    ]
    return FunctionCallExample(
        id="fixture",
        messages=[{"role": "user", "content": "Weather in New York."}],
        tools=TOOLS,
        expected_calls=expected_calls,
        should_call=should_call,
        call_order=call_order,
        scoring_mode=scoring_mode,
        category="simple_python",
        raw={},
    )


class StaticToolClient(ToolCallModelClient):
    def __init__(self, results):
        self.results = results
        self.settings = []

    def generate_tool_calls(self, examples, **kwargs):
        self.settings.append(kwargs)
        return self.results[: len(examples)]


class BFCLTests(unittest.TestCase):
    def test_schema_rejects_expected_call_for_unknown_tool(self):
        value = {
            "schema_version": "bilgeval.instance/v1",
            "id": "fc-1",
            "task_type": "function_call",
            "split": "test",
            "language": "en",
            "input": {
                "messages": [{"role": "user", "content": "Run it."}],
                "tools": TOOLS[:1],
            },
            "target": {
                "calls": [{"name": "missing", "arguments": {"value": [1]}}],
                "should_call": True,
                "call_order": "unordered",
            },
            "metadata": {"bfcl_category": "simple_python"},
            "provenance": {"source": "fixture"},
        }
        with self.assertRaises(ValidationError):
            parse_canonical_instance(value)

    def test_converter_normalizes_bfcl_schema_and_is_deterministic(self):
        question = {
            "id": "simple_python_0",
            "question": [[{"role": "user", "content": "Find the weather."}]],
            "function": [
                {
                    "name": "weather.lookup",
                    "description": "Look up weather.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "type": {"type": "string"},
                            "temperature": {"type": "float"},
                        },
                        "required": ["type"],
                    },
                }
            ],
        }
        answer = {
            "id": "simple_python_0",
            "ground_truth": [
                {
                    "weather.lookup": {
                        "type": ["current"],
                        "temperature": [20.0, ""],
                    }
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_path = root / "questions.json"
            answer_path = root / "answers.json"
            output_path = root / "output.jsonl"
            question_path.write_text(json.dumps(question) + "\n", encoding="utf-8")
            answer_path.write_text(json.dumps(answer) + "\n", encoding="utf-8")

            first = convert_bfcl_category(
                question_path=question_path,
                answer_path=answer_path,
                output_path=output_path,
                category="simple_python",
                source_revision="fixture",
            )
            first_bytes = output_path.read_bytes()
            second = convert_bfcl_category(
                question_path=question_path,
                answer_path=answer_path,
                output_path=output_path,
                category="simple_python",
                source_revision="fixture",
            )

            self.assertEqual(first.content_hash, second.content_hash)
            self.assertEqual(first_bytes, output_path.read_bytes())
            converted = json.loads(output_path.read_text(encoding="utf-8"))
            parameters = converted["input"]["tools"][0]["function"]["parameters"]
            self.assertEqual(parameters["type"], "object")
            self.assertEqual(parameters["properties"]["temperature"]["type"], "number")
            self.assertEqual(parameters["properties"]["type"]["type"], "string")

    def test_converter_normalizes_java_and_javascript_schema_types(self):
        normalized = normalize_json_schema(
            {
                "type": "dict",
                "properties": {
                    "name": {"type": "String"},
                    "enabled": {"type": "Boolean"},
                    "count": {"type": "long"},
                    "values": {"type": "ArrayList", "items": {"type": "double"}},
                    "metadata": {"type": "HashMap"},
                    "callback": {"type": "any"},
                },
            }
        )

        self.assertEqual(normalized["type"], "object")
        self.assertEqual(normalized["properties"]["name"]["type"], "string")
        self.assertEqual(normalized["properties"]["enabled"]["type"], "boolean")
        self.assertEqual(normalized["properties"]["count"]["type"], "integer")
        self.assertEqual(normalized["properties"]["values"]["type"], "array")
        self.assertEqual(
            normalized["properties"]["values"]["items"]["type"],
            "number",
        )
        self.assertEqual(normalized["properties"]["metadata"]["type"], "object")
        self.assertEqual(normalized["properties"]["callback"]["type"], "string")

    def test_single_call_scoring_accepts_optional_omission_and_string_normalization(self):
        score = score_function_calls(
            expected_calls=example().expected_calls,
            candidate_calls=[
                {
                    "name": "weather.lookup",
                    "arguments": {"city": "new york ny"},
                }
            ],
            tools=TOOLS,
        )
        self.assertTrue(score.valid)
        self.assertTrue(all(score.diagnostics.values()))

    def test_parallel_scoring_is_order_independent(self):
        expected = [
            {"name": "weather.lookup", "arguments": {"city": ["Ankara"], "unit": [""]}},
            {"name": "calendar_lookup", "arguments": {"date": ["2026-07-28"]}},
        ]
        candidates = [
            {"name": "calendar_lookup", "arguments": {"date": "2026/07/28"}},
            {"name": "weather.lookup", "arguments": {"city": "ankara"}},
        ]
        score = score_function_calls(
            expected_calls=expected,
            candidate_calls=candidates,
            tools=TOOLS,
        )
        self.assertTrue(score.valid)

    def test_irrelevance_requires_no_tool_call(self):
        self.assertTrue(
            score_function_calls(
                expected_calls=[],
                candidate_calls=[],
                tools=TOOLS,
            ).valid
        )
        score = score_function_calls(
            expected_calls=[],
            candidate_calls=[{"name": "weather.lookup", "arguments": {"city": "Ankara"}}],
            tools=TOOLS,
        )
        self.assertFalse(score.valid)
        self.assertEqual(score.error_type, "wrong_call_count")

    def test_call_presence_scores_relevance_without_argument_matching(self):
        present = score_function_call_presence(
            should_call=True,
            candidate_calls=[
                {
                    "name": "any_recognized_call",
                    "arguments": {},
                    "parse_error": None,
                }
            ],
        )
        malformed = score_function_call_presence(
            should_call=True,
            candidate_calls=[
                {
                    "name": "any_recognized_call",
                    "arguments": None,
                    "parse_error": "arguments were malformed",
                }
            ],
        )
        missing = score_function_call_presence(
            should_call=True,
            candidate_calls=[],
        )
        irrelevant = score_function_call_presence(
            should_call=False,
            candidate_calls=[],
        )

        self.assertTrue(present.valid)
        self.assertFalse(malformed.valid)
        self.assertFalse(missing.valid)
        self.assertEqual(missing.error_type, "missing_tool_call")
        self.assertTrue(irrelevant.valid)

    def test_evaluator_reports_call_presence_accuracy_separately(self):
        relevance = example(
            expected_calls=[],
            should_call=True,
            scoring_mode="call_presence",
        )
        report = evaluate_function_calls(
            [relevance],
            StaticToolClient(
                [
                    GeneratedToolCallResult(
                        tool_calls=[
                            {
                                "name": "weather.lookup",
                                "arguments": {"city": "Ankara"},
                                "parse_error": None,
                            }
                        ]
                    )
                ]
            ),
        )

        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(
            report["function_call_metrics"]["call_presence_accuracy"],
            1.0,
        )
        self.assertEqual(
            report["function_call_metrics"]["call_presence_support"],
            1,
        )

    def test_invalid_argument_json_is_scored_incorrect_not_failed(self):
        client = StaticToolClient(
            [
                GeneratedToolCallResult(
                    tool_calls=[
                        {
                            "name": "weather.lookup",
                            "arguments": None,
                            "raw_arguments": "{bad",
                            "parse_error": "invalid JSON",
                        }
                    ]
                )
            ]
        )
        report = evaluate_function_calls(
            [example()],
            client,
            generation_settings={"temperature": 0.0, "stop": ["ignored"]},
        )
        self.assertEqual(report["num_scored_examples"], 1)
        self.assertEqual(report["num_failed_examples"], 0)
        self.assertEqual(report["correct"], 0)
        self.assertEqual(report["results"][0]["error_type"], "invalid_arguments_json")
        self.assertEqual(report["function_call_metrics"]["parse_failure_rate"], 1.0)
        self.assertEqual(client.settings[0]["stop"], ["ignored"])

    def test_transport_failures_are_excluded_from_accuracy(self):
        report = evaluate_function_calls(
            [example()],
            StaticToolClient([GeneratedToolCallResult(error="endpoint unavailable")]),
        )
        self.assertEqual(report["num_scored_examples"], 0)
        self.assertEqual(report["num_failed_examples"], 1)
        self.assertEqual(report["coverage"], 0.0)

    def test_wire_tool_names_are_reversible_and_collisions_fail(self):
        wire_tools, mapping = OpenAICompatibleToolCallClient.wire_tools(TOOLS)
        self.assertEqual(wire_tools[0]["function"]["name"], "weather_lookup")
        self.assertEqual(mapping["weather_lookup"], "weather.lookup")
        colliding = [
            TOOLS[0],
            {
                "type": "function",
                "function": {
                    **TOOLS[0]["function"],
                    "name": "weather_lookup",
                },
            },
        ]
        with self.assertRaisesRegex(ValueError, "collide"):
            OpenAICompatibleToolCallClient.wire_tools(colliding)

    def test_functiongemma_control_tokens_parse_nested_arguments(self):
        calls = parse_functiongemma_tool_calls(
            "<|tool_call>call:weather_lookup"
            "{city:<escape>New York, NY<escape>,days:[1,2],"
            "options:{metric:true,limit:3}}<tool_call|>"
        )
        self.assertEqual(
            calls[0]["arguments"],
            {
                "city": "New York, NY",
                "days": [1, 2],
                "options": {"metric": True, "limit": 3},
            },
        )
        self.assertEqual(calls[0]["parser"], "functiongemma_control_tokens")

    def test_functiongemma_parser_ignores_ordinary_assistant_text(self):
        self.assertEqual(
            parse_functiongemma_tool_calls("Try weather_lookup(city='Ankara')."),
            [],
        )

    def test_functiongemma_sglang_quote_tokens_parse_as_strings(self):
        calls = parse_functiongemma_tool_calls(
            '<|tool_call>call:country_info_capital'
            '{country:<|"|>Brazil<|"|>}<tool_call|>'
        )
        self.assertEqual(calls[0]["arguments"], {"country": "Brazil"})



if __name__ == "__main__":
    unittest.main()
