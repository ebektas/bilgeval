"""Final-answer extraction and equivalence for generated free-response answers.

Every upstream harness scores the FINAL answer of a response, never a value
that merely appears somewhere in the reasoning:

- lm-evaluation-harness `gsm8k_cot`: "The answer is X" (strict) or the LAST
  number in the response (flexible-extract).
- MATH (Hendrycks et al.): the LAST `\\boxed{...}`, compared after
  `strip_string` normalization.
- MMLU-Pro / SuperGPQA: the final "answer is (X)" / last line.

This module implements that contract once:

1. `extract_final_answer` locates the final-answer span: last `\\boxed{}`,
   else the text after the last answer marker (Cevap/Yanıt/Answer), else the
   last non-empty line.
2. Numeric golds are compared by VALUE. The gold is read with the dataset's
   (English) convention; the candidate is read with both the Turkish and the
   English decimal convention, so `3,5` and `3.5` are the same number and
   `1.000` can be one thousand or one.
3. Other golds are compared as normalized math strings (a port of the MATH
   `strip_string` rules) against the final-answer span only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

ANSWER_PARSER_VERSION = 8

_LETTER = r"\wçğıöşüÇĞİÖŞÜ"
_MARKER_RE = re.compile(
    rf"(?<![{_LETTER}])(?:final\s+answer|answer|nihai\s+cevap|ceva(?:p|b[ıi])|yan[ıi]t[ıi]?)(?![{_LETTER}])",
    flags=re.IGNORECASE,
)
_NUMBER_BODY = r"(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?|[.,]\d+)"
_FRACTION_TEX = r"\\[dt]?frac\s*\{\s*([-+]?\s*\d+)\s*\}\s*\{\s*(\d+)\s*\}"
_FRACTION_TEX_SHORT = r"\\[dt]?frac\s*(\d)\s*(\d)"
_NUMBER_TOKEN_RE = re.compile(
    rf"(?P<sign>[-+−]\s*)?(?:"
    rf"(?P<texfrac>{_FRACTION_TEX})|"
    rf"(?P<texshort>{_FRACTION_TEX_SHORT})|"
    rf"(?P<slash>(?<![\w.,])\d+\s*/\s*\d+(?![\w]))|"
    rf"(?P<plain>(?<![\w.,]){_NUMBER_BODY}(?![\w])(?![.,]\d))"
    rf")"
)
_TRAILING_SUFFIX_RE = re.compile(r"['’](?:d[ıiuü]r|t[ıiuü]r|d[ıiuü]|t[ıiuü])\b", flags=re.IGNORECASE)
_UNIT_WRAPPERS_RE = re.compile(r"\\(?:text|mbox|mathrm|textbf)\s*\{([^{}]*)\}")
_PARENTHESIS_RE = re.compile(r"\s*\([^()]*\)")


@dataclass(frozen=True)
class FinalAnswer:
    text: str
    source: str


def _last_boxed(text: str) -> str | None:
    index = text.rfind("\\boxed")
    while index != -1:
        brace = text.find("{", index)
        if brace != -1 and not text[index + len("\\boxed") : brace].strip():
            depth = 0
            for position in range(brace, len(text)):
                if text[position] == "{":
                    depth += 1
                elif text[position] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[brace + 1 : position]
        index = text.rfind("\\boxed", 0, index)
    return None


def extract_final_answer(text: str) -> FinalAnswer:
    value = (text or "").strip()
    if not value:
        return FinalAnswer("", "empty")

    boxed = _last_boxed(value)
    if boxed is not None and boxed.strip():
        return FinalAnswer(boxed.strip(), "boxed")

    markers = list(_MARKER_RE.finditer(value))
    if markers:
        marker = markers[-1]
        line_end = value.find("\n", marker.end())
        remainder = value[marker.end() : line_end if line_end != -1 else len(value)]
        remainder = re.sub(r"^\s*(?:is|olarak)?\s*[:=]?\s*", "", remainder)
        if not remainder.strip() and line_end != -1:
            following = [line for line in value[line_end:].splitlines() if line.strip()]
            remainder = following[0] if following else ""
        if remainder.strip():
            return FinalAnswer(remainder.strip(), "marker")

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return FinalAnswer(lines[-1] if lines else "", "last_line")


def _read(body: str, decimal: str, thousands: str) -> Fraction | None:
    if body.count(decimal) > 1:
        return None
    integer_part, separator, fraction_part = body.partition(decimal)
    if thousands in fraction_part or (separator and not fraction_part):
        return None
    if thousands in integer_part:
        groups = integer_part.split(thousands)
        if groups[0] in {"", "0"} or len(groups[0]) > 3 or any(len(group) != 3 for group in groups[1:]):
            return None
        integer_part = "".join(groups)
    if not integer_part and not fraction_part:
        return None
    try:
        return Fraction((integer_part or "0") + (f".{fraction_part}" if fraction_part else ""))
    except ValueError:
        return None


def _readings(token: str, *, conventions: tuple[str, ...]) -> set[Fraction]:
    """Values a plain number token can denote under the given conventions.

    English: `.` decimal, `,` thousands. Turkish: `,` decimal, `.` thousands.
    `3,5` is only 3.5; `1.000` is 1 (English) or 1000 (Turkish).
    """
    body = re.sub(r"\s+", "", token)
    if not re.fullmatch(r"[\d.,]+", body) or not re.search(r"\d", body):
        return set()
    values: set[Fraction] = set()
    for convention in conventions:
        decimal, thousands = (".", ",") if convention == "en" else (",", ".")
        value = _read(body, decimal, thousands)
        if value is not None:
            values.add(value)
    return values


def number_values(text: str, *, conventions: tuple[str, ...] = ("tr", "en")) -> list[set[Fraction]]:
    """Every number in `text`, in order, each as the set of values it can denote."""
    found: list[set[Fraction]] = []
    for match in _NUMBER_TOKEN_RE.finditer(text.replace("−", "-")):
        sign = -1 if (match.group("sign") or "").strip() in {"-", "−"} else 1
        if match.group("sign") and match.start() > 0:
            before = text[: match.start()].rstrip()
            if before and (before[-1].isdigit() or before[-1] in ")}]"):
                sign = 1
        values: set[Fraction] = set()
        try:
            if match.group("texfrac"):
                numerator = int(re.sub(r"\s+", "", match.group(3)))
                denominator = int(match.group(4))
                if denominator:
                    values.add(Fraction(numerator, denominator))
            elif match.group("texshort"):
                denominator = int(match.group(7))
                if denominator:
                    values.add(Fraction(int(match.group(6)), denominator))
            elif match.group("slash"):
                numerator, denominator = (
                    int(part) for part in re.split(r"\s*/\s*", match.group("slash"))
                )
                if denominator:
                    values.add(Fraction(numerator, denominator))
            else:
                values = _readings(match.group("plain"), conventions=conventions)
        except ValueError:
            continue
        if values:
            found.append({sign * value for value in values})
    return found


def strip_units(text: str) -> str:
    value = text.strip()
    value = re.sub(r"\\\$|\$", "", value)
    value = re.sub(r"\\%|%", "", value)
    value = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "", value)
    value = _UNIT_WRAPPERS_RE.sub(" ", value)
    value = re.sub(r"\^\s*\{?\s*\d+\s*\}?\s*$", "", value.strip())
    return value.strip()


def gold_number(gold: str) -> Fraction | None:
    """The single numeric value of a gold answer, or None when it is not a plain number.

    Read with the dataset convention (English): `10,000` is ten thousand and
    `-2,1` is a list of two values, not a decimal.
    """
    stripped = strip_units(gold)
    stripped = re.sub(r"\\!|\\,|\\;|\\ ", "", stripped).strip()
    values = number_values(stripped, conventions=("en",))
    if len(values) != 1 or len(values[0]) != 1:
        return None
    remainder = _NUMBER_TOKEN_RE.sub("", stripped.replace("−", "-"), count=1)
    if re.sub(r"[\s.{}]", "", remainder):
        return None
    return next(iter(values[0]))


def drop_restatements(span: str) -> str:
    """The final-answer span without parentheses after its first number.

    "95 minutes (or 1 hour and 35 minutes)" answers 95; the last number of the
    span would be 35. Parentheses before the first number are kept, and
    "4 × (2 + 3) = 20" still ends in its result.
    """
    first = _NUMBER_TOKEN_RE.search(span)
    if first is None:
        return span
    return span[: first.end()] + _PARENTHESIS_RE.sub("", span[first.end() :])


def normalize_math(text: str) -> str:
    """Port of the MATH `strip_string` normalization, plus Turkish suffix removal."""
    value = (text or "").strip()
    value = _TRAILING_SUFFIX_RE.sub("", value)
    value = value.replace("\n", "").replace("\\!", "")
    value = value.replace("\\\\", "\\")
    value = re.sub(r"\\[dt]frac", r"\\frac", value)
    value = value.replace("\\left", "").replace("\\right", "")
    value = re.sub(r"\^\s*\{\s*\\circ\s*\}|\^\s*\\circ", "", value)
    value = value.replace("\\$", "").replace("$", "")
    value = value.replace("\\%", "").replace("%", "")
    value = re.sub(r"\\text\{\s[^{}]*\}\s*$", "", value)
    value = _UNIT_WRAPPERS_RE.sub(lambda match: match.group(1), value)
    value = re.sub(r"\\(?:,|;|:|quad|qquad| )", "", value)
    value = value.replace("−", "-")
    value = value.strip().rstrip(".").strip()
    if "=" in value:
        left, _, right = value.partition("=")
        if len(left.strip()) <= 2 and right.strip():
            value = right
    value = re.sub(r"\\sqrt\s*(\w)", r"\\sqrt{\1}", value)
    value = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", value)
    value = re.sub(r"\\frac\s*(\d)\s*(?=\{)", r"\\frac{\1}", value)
    value = re.sub(r"\\frac\{([^{}]+)\}\s*(\d)", r"\\frac{\1}{\2}", value)
    value = re.sub(r"(?<![\w.])(\d+)\s*/\s*(\d+)(?![\w.])", r"\\frac{\1}{\2}", value)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"(?<!\d)\.(\d)", r"0.\1", value)
    value = re.sub(r"([_^])\{(\w)\}", r"\1\2", value)
    unbraced = re.fullmatch(r"\\\{(.*)\\\}", value)
    if unbraced and "\\{" not in unbraced.group(1):
        value = unbraced.group(1)
    letter = re.fullmatch(r"\(([A-Za-z])\)", value)
    if letter:
        value = letter.group(1)
    return value.casefold()


@dataclass(frozen=True)
class AnswerMatch:
    correct: bool
    extracted: str
    source: str
    kind: str


def match_final_answer(generated_text: str, gold_answer: str) -> AnswerMatch:
    final = extract_final_answer(generated_text)
    gold_value = gold_number(gold_answer)
    if gold_value is not None:
        span = final.text if final.source == "boxed" else drop_restatements(final.text)
        span_numbers = number_values(strip_units(span))
        source = final.source
        if not span_numbers:
            span_numbers = number_values(strip_units(generated_text or ""))
            source = "last_number"
        if not span_numbers:
            return AnswerMatch(False, final.text, final.source, "numeric")
        candidate = span_numbers[-1]
        rendered = "|".join(sorted(_render(value) for value in candidate))
        return AnswerMatch(gold_value in candidate, rendered, source, "numeric")

    normalized_gold = normalize_math(gold_answer)
    normalized_span = normalize_math(final.text)
    if not normalized_gold or not normalized_span:
        return AnswerMatch(False, final.text, final.source, "math_string")
    correct = normalized_span == normalized_gold
    if not correct and final.source in {"marker", "last_line"}:
        position = normalized_span.find(normalized_gold)
        if position != -1:
            before = normalized_span[position - 1] if position > 0 else ""
            after_index = position + len(normalized_gold)
            after = normalized_span[after_index] if after_index < len(normalized_span) else ""
            glued_before = bool(before) and (before.isdigit() or before in "-+.,/^_{\\")
            glued_after = bool(after) and (after.isdigit() or after in "^_/{")
            correct = not glued_before and not glued_after
    return AnswerMatch(correct, normalized_span, final.source, "math_string")


def _render(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
