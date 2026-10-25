from bilgeval.evaluators import (
    GeneratedTextResult,
    TextGenerationModelClient,
    evaluate_oneruler_tr,
    score_oneruler_tr,
    turkish_fold,
)
from bilgeval.reader import OneRulerExample


class TextClient(TextGenerationModelClient):
    def __init__(self, responder):
        self.responder = responder

    def generate_texts(self, prompts, **kwargs):
        results = []
        for prompt in prompts:
            response = self.responder(prompt)
            results.append(response if isinstance(response, GeneratedTextResult) else GeneratedTextResult(text=response))
        return results


def test_needle_answers_are_exactly_the_planted_numbers():
    assert score_oneruler_tr("niah_single", ["2867825"], "<Cevap> 2867825 </Cevap>")
    assert score_oneruler_tr("niah_single", ["2867825"], "Metindeki sayı 2867825.")
    assert not score_oneruler_tr("niah_single", ["2867825"], "<Cevap> 2867824 </Cevap>")
    assert not score_oneruler_tr("niah_single", ["2867825"], "<Cevap> 2867825, 4744854 </Cevap>")
    assert score_oneruler_tr("niah_multivalue", ["5614226", "1419610"], "<Cevap>\n1. 1419610\n2. 5614226\n</Cevap>")
    assert score_oneruler_tr(
        "niah_multiquery",
        ["5614226", "1419610"],
        "Metinde 4744854 de geçiyor ama başka anahtar için.\n<Cevap>\nbilgisayar: 5614226\ntelefon: 1419610\n</Cevap>",
    )


def test_needle_answer_that_says_yok_is_wrong_but_yoksa_is_a_word_of_its_own():
    assert not score_oneruler_tr("niah_single", ["2867825"], "<Cevap> 2867825 ya da yok </Cevap>")
    assert score_oneruler_tr("niah_single", ["2867825"], "Başka sayı yoksa cevap 2867825.")


def test_none_task_wants_the_word_yok():
    assert score_oneruler_tr("niah_none", ["yok"], "<Cevap> yok </Cevap>")
    assert score_oneruler_tr("niah_none", ["yok"], "<Cevap> YOK </Cevap>")
    assert score_oneruler_tr("niah_none", ["yok"], "Böyle bir sayı yok.")
    assert not score_oneruler_tr("niah_none", ["yok"], "<Cevap> 6647119, 5661907 </Cevap>")
    assert not score_oneruler_tr("niah_none", ["yok"], "Sayı yoksa ne olur?")


def test_common_words_match_whole_words_with_turkish_case():
    gold = ["ışık", "fedakâr", "kale"]
    assert score_oneruler_tr("cwe", gold, "<Cevap> IŞIK, Fedakar, kale </Cevap>")
    assert not score_oneruler_tr("cwe", gold, "<Cevap> ışık, fedakâr, kalem </Cevap>")
    assert not score_oneruler_tr("cwe_easy", gold, "<Cevap> ışık, fedakâr </Cevap>")


def test_turkish_fold():
    assert turkish_fold("IŞIK İzmir Hâkim") == "ışık izmir hakim"


def test_evaluate_counts_a_failed_generation_as_wrong_and_strips_reasoning():
    examples = [
        OneRulerExample(id="a", prompt="p1", outputs=["1234567"], task="niah_single", raw={"task": "niah_single"}),
        OneRulerExample(id="b", prompt="p2", outputs=["yok"], task="niah_none", raw={"task": "niah_none"}),
        OneRulerExample(id="c", prompt="p3", outputs=["7654321"], task="niah_single", raw={"task": "niah_single"}),
    ]
    responses = {
        "p1": "<think>9999999 mi?</think><Cevap> 1234567 </Cevap>",
        "p2": "<Cevap> 1111111 </Cevap>",
        "p3": GeneratedTextResult(error="request timed out"),
    }
    report = evaluate_oneruler_tr(examples, TextClient(lambda prompt: responses[prompt]), request_batch_size=8)

    assert [row.get("correct") for row in report["results"]] == [True, False, None]
    assert report["results"][0]["candidate_response"] == "<Cevap> 1234567 </Cevap>"
    assert report["correct"] == 1
    assert report["num_scored_examples"] == 2
    assert report["num_failed_examples"] == 1
    assert report["accuracy"] == 0.5
    assert report["results"][2]["task"] == "niah_single"

