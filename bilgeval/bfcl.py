import re
from dataclasses import dataclass
from typing import Any


_STRING_PUNCTUATION = re.compile(r"[ ,./\-_*^]")
_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass(frozen=True)
class FunctionCallScore:
    valid: bool
    error_type: str | None
    errors: list[str]
    diagnostics: dict[str, bool]


class _FunctionGemmaArgumentParser:
    """Parse FunctionGemma's compact, unquoted argument object syntax."""

    def __init__(self):
        self.root: list[Any] = []
        self.stack: list[Any] = [self.root]
        self.key_stack: list[str | None] = []
        self.buffer: list[str] = []
        self.pending_key: Any = None

    def parse(self, text: str) -> dict[str, Any]:
        cursor = 0
        while cursor < len(text):
            if text.startswith("<escape>", cursor):
                cursor = self._consume_delimited(text, cursor, "<escape>")
                continue
            if text.startswith('<|"|>', cursor):
                cursor = self._consume_delimited(text, cursor, '<|"|>')
                continue
            character = text[cursor]
            if character in "{[:]},":
                self._commit_buffer()
                self._handle_structure(character)
            else:
                self.buffer.append(character)
            cursor += 1
        self._commit_buffer()
        return self.root[0] if self.root and isinstance(self.root[0], dict) else {}

    def _consume_delimited(self, text: str, start: int, delimiter: str) -> int:
        content_start = start + len(delimiter)
        end = text.find(delimiter, content_start)
        if end == -1:
            value = text[start:]
            cursor = len(text)
        else:
            value = text[content_start:end]
            cursor = end + len(delimiter)
        if self.buffer:
            self._commit_buffer()
        self._add_value(value)
        return cursor

    def _handle_structure(self, character: str) -> None:
        if character == "{":
            value: dict[str, Any] = {}
            self._add_value(value)
            self.stack.append(value)
            self.key_stack.append(None)
            self.pending_key = None
        elif character == "[":
            value = []
            self._add_value(value)
            self.stack.append(value)
            self.pending_key = None
        elif character in "}]":
            if len(self.stack) > 1:
                self.stack.pop()
            if character == "}" and self.key_stack:
                self.key_stack.pop()
            self.pending_key = None
        elif character == ":":
            if self.buffer:
                key = "".join(self.buffer).strip()
                self.buffer.clear()
            elif self.pending_key is not None:
                key = str(self.pending_key)
                self.pending_key = None
            else:
                key = None
            if key is not None and isinstance(self.stack[-1], dict) and self.key_stack:
                self.key_stack[-1] = key
        elif character == ",":
            self.pending_key = None

    def _commit_buffer(self) -> None:
        if not self.buffer:
            return
        raw_value = "".join(self.buffer).strip()
        self.buffer.clear()
        if raw_value:
            self._add_value(self._parse_primitive(raw_value))

    def _add_value(self, value: Any) -> None:
        container = self.stack[-1]
        if isinstance(container, list):
            container.append(value)
        elif isinstance(container, dict):
            key = self.key_stack[-1] if self.key_stack else None
            if key is not None:
                container[key] = value
                self.key_stack[-1] = None
            else:
                self.pending_key = value

    @staticmethod
    def _parse_primitive(value: str) -> Any:
        if value == "null":
            return None
        if value == "true":
            return True
        if value == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value


def parse_functiongemma_tool_calls(text: str | None) -> list[dict[str, Any]]:
    """Parse only explicit FunctionGemma control-token call blocks.

    Adapted from BFCL's Apache-2.0 FunctionGemma response parser.
    """
    if not text:
        return []
    token_pairs = (
        ("<|tool_call>", "<tool_call|>"),
        ("<start_function_call>", "<end_function_call>"),
    )
    blocks = []
    for start_token, end_token in token_pairs:
        blocks.extend(
            re.findall(
                re.escape(start_token) + r"(.*?)" + re.escape(end_token),
                text,
                flags=re.DOTALL,
            )
        )

    calls = []
    for block in blocks:
        content = block.strip()
        if not content.startswith("call:"):
            continue
        body = content[len("call:") :]
        opening_brace = body.find("{")
        if opening_brace == -1:
            name = body.strip()
            arguments = {}
            raw_arguments = "{}"
        else:
            name = body[:opening_brace].strip()
            raw_arguments = body[opening_brace:]
            arguments = _FunctionGemmaArgumentParser().parse(raw_arguments)
        if name:
            calls.append(
                {
                    "id": None,
                    "name": name,
                    "arguments": arguments,
                    "raw_arguments": raw_arguments,
                    "parse_error": None,
                    "parser": "functiongemma_control_tokens",
                }
            )
    return calls


def standardize_string(value: str) -> str:
    """Apply the normalization used by BFCL's Python AST checker."""
    return _STRING_PUNCTUATION.sub("", value).lower().replace("'", '"')


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return standardize_string(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_value(item) for key, item in value.items()}
    return value


def _type_matches(value: Any, schema: dict[str, Any], accepted: list[Any]) -> bool:
    schema_type = schema.get("type")
    expected_type = _JSON_TYPES.get(schema_type)
    if expected_type is None:
        return True
    if schema_type in {"integer", "number"} and isinstance(value, bool):
        return False
    if schema_type == "number" and isinstance(value, int):
        return True
    if isinstance(value, expected_type):
        return True

    expected_value = next((item for item in accepted if item != ""), None)
    return expected_value is not None and type(value) is type(expected_value)


def _value_matches(value: Any, accepted: list[Any]) -> bool:
    normalized_value = normalize_value(value)
    return any(normalized_value == normalize_value(candidate) for candidate in accepted)


def _tool_schema(
    tools: list[dict[str, Any]],
    name: str,
) -> dict[str, Any] | None:
    for tool in tools:
        function = tool.get("function", {})
        if function.get("name") == name:
            return function
    return None


def _match_call(
    expected: dict[str, Any],
    candidate: dict[str, Any],
    tools: list[dict[str, Any]],
) -> FunctionCallScore:
    diagnostics = {
        "call_count_correct": True,
        "function_names_correct": False,
        "argument_names_correct": False,
        "argument_values_correct": False,
    }
    expected_name = expected["name"]
    candidate_name = candidate.get("name")
    if candidate_name != expected_name:
        return FunctionCallScore(
            valid=False,
            error_type="wrong_function_name",
            errors=[
                f"Expected function {expected_name!r}, got {candidate_name!r}."
            ],
            diagnostics=diagnostics,
        )
    diagnostics["function_names_correct"] = True

    parse_error = candidate.get("parse_error")
    if parse_error:
        return FunctionCallScore(
            valid=False,
            error_type="invalid_arguments_json",
            errors=[str(parse_error)],
            diagnostics=diagnostics,
        )
    arguments = candidate.get("arguments")
    if not isinstance(arguments, dict):
        return FunctionCallScore(
            valid=False,
            error_type="invalid_arguments_type",
            errors=["Tool-call arguments must decode to a JSON object."],
            diagnostics=diagnostics,
        )

    tool = _tool_schema(tools, expected_name)
    if tool is None:
        return FunctionCallScore(
            valid=False,
            error_type="missing_tool_schema",
            errors=[f"No tool schema is available for {expected_name!r}."],
            diagnostics=diagnostics,
        )
    parameters = tool.get("parameters", {})
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    expected_arguments = expected.get("arguments", {})

    unexpected = sorted(set(arguments) - set(properties))
    missing_required = sorted(required - set(arguments))
    missing_expected = sorted(
        name
        for name, accepted in expected_arguments.items()
        if name not in arguments and "" not in accepted
    )
    if unexpected or missing_required or missing_expected:
        errors = []
        if unexpected:
            errors.append("Unexpected arguments: " + ", ".join(unexpected) + ".")
        if missing_required:
            errors.append("Missing required arguments: " + ", ".join(missing_required) + ".")
        if missing_expected:
            errors.append("Missing expected arguments: " + ", ".join(missing_expected) + ".")
        return FunctionCallScore(
            valid=False,
            error_type="argument_names_mismatch",
            errors=errors,
            diagnostics=diagnostics,
        )

    unscorable = sorted(set(arguments) - set(expected_arguments))
    if unscorable:
        return FunctionCallScore(
            valid=False,
            error_type="unexpected_argument",
            errors=["Arguments absent from the gold answer: " + ", ".join(unscorable) + "."],
            diagnostics=diagnostics,
        )
    diagnostics["argument_names_correct"] = True

    for name, value in arguments.items():
        accepted = expected_arguments[name]
        schema = properties.get(name, {})
        if not _type_matches(value, schema, accepted):
            return FunctionCallScore(
                valid=False,
                error_type="argument_type_mismatch",
                errors=[
                    f"Argument {name!r} has type {type(value).__name__}; "
                    f"expected {schema.get('type', 'any')}."
                ],
                diagnostics=diagnostics,
            )
        if not _value_matches(value, accepted):
            return FunctionCallScore(
                valid=False,
                error_type="argument_value_mismatch",
                errors=[
                    f"Argument {name!r} value {value!r} does not match any accepted value."
                ],
                diagnostics=diagnostics,
            )

    diagnostics["argument_values_correct"] = True
    return FunctionCallScore(
        valid=True,
        error_type=None,
        errors=[],
        diagnostics=diagnostics,
    )


def score_function_calls(
    *,
    expected_calls: list[dict[str, Any]],
    candidate_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    call_order: str = "unordered",
) -> FunctionCallScore:
    diagnostics = {
        "call_count_correct": len(expected_calls) == len(candidate_calls),
        "function_names_correct": False,
        "argument_names_correct": False,
        "argument_values_correct": False,
    }
    if len(expected_calls) != len(candidate_calls):
        return FunctionCallScore(
            valid=False,
            error_type="wrong_call_count",
            errors=[
                f"Expected {len(expected_calls)} call(s), got {len(candidate_calls)}."
            ],
            diagnostics=diagnostics,
        )
    if not expected_calls:
        return FunctionCallScore(
            valid=True,
            error_type=None,
            errors=[],
            diagnostics={name: True for name in diagnostics},
        )
    if call_order not in {"ordered", "unordered"}:
        raise ValueError(f"Unsupported function-call order: {call_order}")

    remaining = list(range(len(candidate_calls)))
    matched_scores: list[FunctionCallScore] = []
    for expected_index, expected in enumerate(expected_calls):
        candidate_indices = (
            [expected_index]
            if call_order == "ordered"
            else list(remaining)
        )
        attempts = [
            (index, _match_call(expected, candidate_calls[index], tools))
            for index in candidate_indices
        ]
        match = next(
            ((index, score) for index, score in attempts if score.valid),
            None,
        )
        if match is None:
            best = max(
                (score for _, score in attempts),
                key=lambda score: sum(score.diagnostics.values()),
            )
            aggregate = {
                name: all(
                    score.diagnostics.get(name, False)
                    for score in matched_scores + [best]
                )
                for name in diagnostics
            }
            return FunctionCallScore(
                valid=False,
                error_type=best.error_type or "no_matching_call",
                errors=best.errors,
                diagnostics=aggregate,
            )
        candidate_index, matched_score = match
        matched_scores.append(matched_score)
        remaining.remove(candidate_index)

    return FunctionCallScore(
        valid=True,
        error_type=None,
        errors=[],
        diagnostics={name: True for name in diagnostics},
    )


def score_function_call_presence(
    *,
    should_call: bool,
    candidate_calls: list[dict[str, Any]],
) -> FunctionCallScore:
    valid_calls = [
        call
        for call in candidate_calls
        if call.get("name") and not call.get("parse_error")
    ]
    call_present = bool(valid_calls)
    valid = call_present == should_call
    if valid:
        return FunctionCallScore(
            valid=True,
            error_type=None,
            errors=[],
            diagnostics={
                "call_count_correct": True,
                "function_names_correct": True,
                "argument_names_correct": True,
                "argument_values_correct": True,
            },
        )

    expected = "at least one tool call" if should_call else "no tool calls"
    observed = (
        f"{len(valid_calls)} valid tool call(s) "
        f"out of {len(candidate_calls)} emitted call(s)"
    )
    return FunctionCallScore(
        valid=False,
        error_type=(
            "missing_tool_call" if should_call else "unexpected_tool_call"
        ),
        errors=[f"Expected {expected}, got {observed}."],
        diagnostics={
            "call_count_correct": False,
            "function_names_correct": False,
            "argument_names_correct": False,
            "argument_values_correct": False,
        },
    )
