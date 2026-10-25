"""Multiple-choice labels that are numbers (ARC's rows labelled 1-4)."""

from bilgeval.answer_extraction import ANSWER_PARSER_VERSION
from bilgeval.evaluators import build_answer_verifier_prompt, parse_regex_match_prediction
from bilgeval.reader import Choice, Example


def example(labels):
    texts = ["Kinetik enerji azalır.", "Kinetik enerji artar.", "İkisi de azalır.", "İkisi de artar."]
    choices = [Choice(label=label, text=text) for label, text in zip(labels, texts)]
    return Example(id="e", question="Hangisi?", choices=choices, answer_label=labels[1], answer_text=texts[1], raw={})


def test_numbered_labels_parse_after_the_marker_or_at_the_start():
    numbered = example(["1", "2", "3", "4"])
    for response in ["Cevap: 2) Kinetik enerji artar.", "**Cevap: 2**", "Cevap: 2", "2) Kinetik enerji artar.",
                     "Önce düşünelim.\nCevap: (2)", "Doğru seçenek (2)."]:
        assert parse_regex_match_prediction(response, numbered, "label") == ("2", "Kinetik enerji artar."), response


def test_a_number_that_is_not_a_label_is_not_read_as_one():
    numbered = example(["1", "2", "3", "4"])
    for response in ["1. Önce hıza bakalım.\n2. Sonra yüksekliğe.", "Cevap: 2.5 J", "Cevap: 20 joule"]:
        assert parse_regex_match_prediction(response, numbered, "label") is None, response


def test_lettered_labels_parse_as_before():
    lettered = example(["A", "B", "C", "D"])
    assert parse_regex_match_prediction("Cevap: B) Kinetik enerji artar.", lettered, "label") == ("B", "Kinetik enerji artar.")
    assert parse_regex_match_prediction("Sanırım B", lettered, "label") == ("B", "Kinetik enerji artar.")
    assert parse_regex_match_prediction("Cevap: 2", lettered, "label") is None


def test_the_verifier_gets_the_multiple_choice_rules_for_numbered_labels():
    rules = "The gold answer is formatted as OPTION_LABEL: OPTION_TEXT"
    assert rules in build_answer_verifier_prompt("2: Kinetik enerji artar.", "Cevap: 2", multiple_choice=True)
    assert rules not in build_answer_verifier_prompt("2: Kinetik enerji artar.", "Cevap: 2")
    assert rules in build_answer_verifier_prompt("B: Kinetik enerji artar.", "Cevap: B")
    assert rules not in build_answer_verifier_prompt("3:2", "Oran 3:2", multiple_choice=False)


def test_a_revised_final_choice_stated_as_dogru_secenek_wins():
    lettered = example(["A", "B", "C", "D"])
    revised = "Cevap: **B) Kinetik enerji artar.**\n\nAçıklama ...\n\n**Doğru Seçenek: C) İkisi de azalır.**"
    assert parse_regex_match_prediction(revised, lettered, "label") == ("C", "İkisi de azalır.")
    assert parse_regex_match_prediction("Doğru şık (B).", lettered, "label") == ("B", "Kinetik enerji artar.")
    assert parse_regex_match_prediction("Doğru seçenek C değil; cevap: B", lettered, "label") == ("B", "Kinetik enerji artar.")


def test_the_parser_version_moves_the_fingerprint():
    assert ANSWER_PARSER_VERSION == 8


def test_a_letter_names_a_numbered_option_by_position():
    numbered = example(["1", "2", "3", "4"])
    assert parse_regex_match_prediction("En büyük kuvvet 2. seçenektedir.\n\nCevap: B", numbered, "label")[0] == "2"
    assert parse_regex_match_prediction("Cevap: 3", numbered, "label")[0] == "3"
    assert parse_regex_match_prediction("Cevap: E", numbered, "label") is None
    assert parse_regex_match_prediction("Cevap: B", example(["A", "B", "C", "D"]), "label")[0] == "B"
