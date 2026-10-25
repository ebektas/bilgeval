from fractions import Fraction

import pytest

from bilgeval.answer_extraction import (
    extract_final_answer,
    gold_number,
    match_final_answer,
    normalize_math,
    number_values,
)


@pytest.mark.parametrize(
    "text, expected, source",
    [
        ("Önce 12 buluruz.\nCevap: 42", "42", "marker"),
        ("Hesap... \\boxed{\\frac{3}{4}} olur.", "\\frac{3}{4}", "boxed"),
        ("\\boxed{1} yanlış, doğrusu \\boxed{2}", "2", "boxed"),
        ("Cevap 5 değil.\nCevap: 7", "7", "marker"),
        ("Cevap:\n\n16\\sqrt{3}", "16\\sqrt{3}", "marker"),
        ("Sonuç olarak sekiz elma kalır.", "Sonuç olarak sekiz elma kalır.", "last_line"),
        ("The final answer is 18.", "18.", "marker"),
    ],
)
def test_final_answer_span(text, expected, source):
    final = extract_final_answer(text)
    assert (final.text, final.source) == (expected, source)


@pytest.mark.parametrize(
    "token, expected",
    [
        ("3,5", {Fraction(7, 2)}),
        ("3.5", {Fraction(7, 2)}),
        ("1.000", {Fraction(1), Fraction(1000)}),
        ("1,000", {Fraction(1), Fraction(1000)}),
        ("10.000,5", {Fraction(20001, 2)}),
        ("10,000.5", {Fraction(20001, 2)}),
        ("2500", {Fraction(2500)}),
        ("0,15", {Fraction(3, 20)}),
    ],
)
def test_number_readings_cover_turkish_and_english_conventions(token, expected):
    assert number_values(token) == [expected]


def test_gold_numbers_use_the_dataset_convention():
    assert gold_number("10,000") == 10000
    assert gold_number("\\frac{1}{3}") == Fraction(1, 3)
    assert gold_number("\\dfrac{33}{100}") == Fraction(33, 100)
    assert gold_number("-\\frac{3}{8}") == Fraction(-3, 8)
    assert gold_number("\\$36") == 36
    assert gold_number("120^\\circ") == 120
    assert gold_number("15\\mbox{ cm}^2") == 15
    assert gold_number("-2,1") is None
    assert gold_number("(8,-2)") is None
    assert gold_number("6\\sqrt{2}") is None
    assert gold_number("40_9") is None
    assert gold_number("1 - 12i") is None


@pytest.mark.parametrize(
    "response, gold, correct",
    [
        ("Ara adımda \\frac{1}{2} bulunur. Cevap: \\frac{3}{4}", "\\frac{1}{2}", False),
        ("Önce 36 bulunur, sonra 36 - 6 = 30. Cevap 30'dur.", "36", False),
        ("Toplam 5 elma vardı. Cevap 7'dir.", "5", False),
        ("... Cevap: 3,5", "3.5", True),
        ("... Cevap: 3.5", "3.5", True),
        ("... Cevap: 0,5", "\\frac{1}{2}", True),
        ("... Cevap: 1/2", "\\frac{1}{2}", True),
        ("... Cevap: \\dfrac{1}{2}", "\\frac{1}{2}", True),
        ("... Cevap: 0,333", "\\frac{1}{3}", False),
        ("Cevap 1.000'dir.", "1000", True),
        ("Cevap: 1.000", "1", True),
        ("Cevap: 10.000 TL", "10,000", True),
        ("Cevap -5'tir", "-5", True),
        ("Cevap: 5 - 3 = 2", "2", True),
        ("Cevap: 41. Bekle, 42 olmalı.", "42", True),
        ("The answer is 95 minutes (or 1 hour and 35 minutes).", "95", True),
        ("The answer is 48 hours (or 2 days).", "48", True),
        ("Cevap 1.128 dakikadır (18 saat 48 dakika).", "1128", True),
        ("Cevap: 4 × (2 + 3) = 20", "20", True),
        ("Cevap: (2 + 3) × 4 = 20", "20", True),
        ("Sonuç: 18.", "18", True),
        ("Açı 120 derece. Cevap: 120^\\circ", "120^\\circ", True),
        ("Cevap: 36 dolar", "\\$36", True),
        ("\\boxed{25}", "25", True),
        ("Cevap: 16\\sqrt{3}", "16 \\sqrt{3}", True),
        ("Cevap: $16\\sqrt3$ cm", "16 \\sqrt{3}", True),
        ("Cevap: x = 6r^2-4r-24", "6r^2-4r-24", True),
        ("Cevap: \\left(\\frac{3}{5},\\frac{8}{3}\\right]", "\\left(\\frac{3}{5},\\frac{8}{3}\\right]", True),
        ("Cevap: (8, -2)", "(8,-2)", True),
        ("Cevap: -2, 1", "-2,1", True),
        ("Cevap: Navin", "\\text{Navin}", True),
        ("Cevap: \\{3, 5, 7\\}", "3, 5, 7", True),
        ("Cevap: 4210_5", "4210_{5}", True),
        ("Cevap: 4210_6", "4210_{5}", False),
        ("Cevap: 9/19", "\\frac9{19}", True),
        ("Cevap: 9/18", "\\frac9{19}", False),
        ("Cevap: 270/7", "\\frac{270}7\\text{ degrees}", True),
        ("Cevap: C", "\\text{(C)}", True),
        ("Cevap: (C)", "\\text{(C)}", True),
        ("Cevap: D", "\\text{(C)}", False),
        ("Cevap: 1+\\sqrt{19}, 1-\\sqrt{19}", "1 \\pm \\sqrt{19}", False),
        ("Cevap: 116\\sqrt{3}", "16 \\sqrt{3}", False),
        ("Cevap: 16\\sqrt{3}^2", "16 \\sqrt{3}", False),
        ("Önce 16\\sqrt{3} buluruz ama bu yanlış.\nCevap: 8\\sqrt{3}", "16 \\sqrt{3}", False),
        ("", "5", False),
    ],
)
def test_match_final_answer(response, gold, correct):
    assert match_final_answer(response, gold).correct is correct, match_final_answer(response, gold)


def test_pathologically_long_integer_is_unparseable_instead_of_crashing():
    huge_integer = "9" * 5_000

    assert number_values(huge_integer) == []
    assert match_final_answer(f"\\boxed{{{huge_integer}}}", "42").correct is False


@pytest.mark.parametrize(
    "huge_fraction",
    [
        "\\frac{" + "9" * 5_000 + "}{1}",
        "9" * 5_000 + "/1",
    ],
)
def test_pathologically_long_fraction_is_unparseable_instead_of_crashing(huge_fraction):
    assert number_values(huge_fraction) == []
    assert match_final_answer(f"\\boxed{{{huge_fraction}}}", "42").correct is False


def test_normalize_math_matches_the_math_grader_rules():
    assert normalize_math("\\dfrac{1}{2}") == normalize_math("\\frac12") == normalize_math("1/2")
    assert normalize_math("x = 5") == "5"
    assert normalize_math("\\left( 3, 4 \\right)") == "(3,4)"
    assert normalize_math(".5") == "0.5"
