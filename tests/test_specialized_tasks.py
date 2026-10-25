from bilgeval.evaluators import (
    GeneratedTextResult,
    TextGenerationModelClient,
    evaluate_document_qa,
    evaluate_safety_behavior,
)
from bilgeval import main as main_module
from bilgeval.reader import DocumentQAExample, SafetyBehaviorExample
from bilgeval.specialized_tasks import (
    document_exact_match,
    parse_safety_judge_label,
)


class FakeTextClient(TextGenerationModelClient):
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def generate_texts(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        return [
            GeneratedTextResult(
                text=self.responder(prompt),
                request_id=f"fake-{len(self.calls)}",
                finish_reason="stop",
            )
            for prompt in prompts
        ]


def safety_example(example_id, expected, subcategory="Dangerous or sensitive topics"):
    return SafetyBehaviorExample(
        id=example_id,
        prompt=f"prompt-{example_id}",
        subcategory=subcategory,
        expected=expected,
        raw={},
    )


def document_example(
    example_id,
    answer,
    answer_format,
    question=None,
    evidence_spans=None,
):
    return DocumentQAExample(
        id=example_id,
        document="Belge içeriği.",
        question=question or f"question-{example_id}",
        answer=answer,
        answer_format=answer_format,
        task_type="open_qa",
        evidence_spans=evidence_spans or ["Kanıt."],
        raw={},
    )


def test_safety_judge_keeps_harmful_and_contrast_metrics_separate():
    examples = [
        safety_example("harmful", "must_refuse"),
        safety_example("benign", "must_engage"),
    ]
    candidate = FakeTextClient(lambda prompt: f"candidate-{prompt}")
    judge = FakeTextClient(
        lambda prompt: (
            "<label>ACCEPTABLE</label>"
            if "prompt-harmful" in prompt
            else "<label>UNACCEPTABLE</label>"
        )
    )

    report = evaluate_safety_behavior(
        examples,
        candidate_client=candidate,
        judge_client=judge,
        request_batch_size=8,
    )

    assert report["accuracy"] == 0.5
    assert report["safety_metrics"]["must_refuse_success_rate"] == 1.0
    assert report["safety_metrics"]["harmful_compliance_rate"] == 0.0
    assert report["safety_metrics"]["must_engage_success_rate"] == 0.0
    assert report["safety_metrics"]["over_refusal_rate"] == 1.0
    assert report["results"][0]["candidate_finish_reason"] == "stop"
    assert report["results"][0]["judge_finish_reason"] == "stop"


def test_document_eval_scores_all_regimes_and_false_refusal():
    examples = [
        document_example("exact", "7.645.571,43", "span"),
        document_example("free", "Ankara", "free_form"),
        document_example("none", "Belgede bulunmuyor", "unanswerable"),
        document_example(
            "ie",
            [{"alan": "Şehir", "değer": "Ankara"}, {"alan": "Yıl", "değer": "2026"}],
            "list",
        ),
    ]

    def candidate_response(prompt):
        question = prompt.split("SORU:\n", 1)[1].split("\n\nCEVAP:", 1)[0]
        return {
            "question-exact": "7.645.571,43",
            "question-free": "Ankara",
            "question-none": "Belgede bu bilgi bulunmuyor.",
            "question-ie": '[{"Şehir": "Ankara", "Yıl": "2026"}]',
        }[question]

    def judge_response(prompt):
        if "Model cevaplamayı reddediyor" in prompt:
            return "FAIL"
        return "PASS"

    report = evaluate_document_qa(
        examples,
        candidate_client=FakeTextClient(candidate_response),
        judge_client=FakeTextClient(judge_response),
        request_batch_size=16,
    )

    assert report["accuracy"] == 1.0
    assert report["document_metrics"]["ie_cell_micro_accuracy"] == 1.0
    assert report["document_metrics"]["false_refusal_rate"] == 0.0
    assert report["document_metrics"]["false_refusal_coverage"] == 1.0
    assert all(not result["failed"] for result in report["results"])
    assert report["results"][0]["candidate_finish_reason"] == "stop"
    assert report["results"][0]["judge_checks"][0]["finish_reason"] == "stop"


def test_turkish_number_normalization_and_safety_label_parser():
    assert document_exact_match("7.645.571,43", "7645571,43")
    assert not document_exact_match("7.645.571,43 TL", "7645571,43")
    assert parse_safety_judge_label("Açıklama\n<label>acceptable</label>") == "ACCEPTABLE"
    assert parse_safety_judge_label("UNACCEPTABLE") == "UNACCEPTABLE"


def test_safety_judge_retries_only_malformed_labels():
    candidate = FakeTextClient(lambda prompt: "candidate answer")
    judge = FakeTextClient(
        lambda prompt: (
            "<label>ACCEPTABLE</label>"
            if "CLASSIFICATION RETRY" in prompt
            else "candidate answer copied without a label"
        )
    )

    report = evaluate_safety_behavior(
        [safety_example("retry", "must_refuse")],
        candidate_client=candidate,
        judge_client=judge,
        request_batch_size=8,
    )

    assert report["coverage"] == 1.0
    assert report["results"][0]["judge_label"] == "ACCEPTABLE"
    assert report["results"][0]["judge_response"] == "candidate answer copied without a label"
    assert report["results"][0]["judge_retry_response"] == "<label>ACCEPTABLE</label>"
    assert len(judge.calls) == 2
    assert judge.calls[1][1]["max_tokens"] == 64


def test_document_judge_retries_only_malformed_verdicts():
    example = document_example("retry", "gold answer", "free_form")
    candidate = FakeTextClient(lambda prompt: "candidate answer")
    judge = FakeTextClient(
        lambda prompt: "PASS" if "CLASSIFICATION RETRY" in prompt else "candidate answer"
    )

    report = evaluate_document_qa(
        [example],
        candidate_client=candidate,
        judge_client=judge,
        request_batch_size=8,
    )

    assert report["coverage"] == 1.0
    assert report["results"][0]["correct"] is True
    assert all(
        check.get("retry_verdict") is True
        for check in report["results"][0]["judge_checks"]
    )
    assert len(judge.calls) == 2
    assert judge.calls[1][1]["max_tokens"] == 64




def turkish_ifeval_no_comma(response):
    from bilgeval.evaluators import evaluate_ifeval
    from bilgeval.reader import IFEvalExample

    example = IFEvalExample(
        id="p1",
        prompt="Bir şiir yaz.\nTüm yanıtın boyunca hiçbir virgül kullanma.",
        instruction_ids=[26],
        verifier_kwargs={"26": {}},
        raw={},
    )
    return evaluate_ifeval([example], model_client=FakeTextClient(lambda prompt: response))


def test_turkish_ifeval_loose_never_passes_an_empty_variant():
    report = turkish_ifeval_no_comma("Bir, iki, üç.")
    assert report["prompt_level_strict_accuracy"] == 0.0
    assert report["prompt_level_loose_accuracy"] == 0.0
    report = turkish_ifeval_no_comma("Virgülsüz bir yanıt\nNot: bir, iki")
    assert report["prompt_level_strict_accuracy"] == 0.0
    assert report["prompt_level_loose_accuracy"] == 1.0


def test_turkish_ifeval_unclosed_reasoning_meets_no_instruction():
    report = turkish_ifeval_no_comma("<think>plan, never closed")
    assert report["ifeval_metrics"]["num_reasoning_blocks_stripped"] == 1
    assert report["prompt_level_strict_accuracy"] == 0.0
    assert report["prompt_level_loose_accuracy"] == 0.0


def test_turkish_ifeval_verifier_revision_is_in_the_fingerprint():
    from types import SimpleNamespace

    from bilgeval import ifeval_verifier
    from bilgeval.manifest import instruction_verifier

    turkish = SimpleNamespace(schema_version="legacy", eval_mode="ifeval")
    english = SimpleNamespace(schema_version="google.ifeval/v1", eval_mode="ifeval")
    other = SimpleNamespace(schema_version="legacy", eval_mode="generate")
    assert instruction_verifier(turkish) == {
        "instruction_verifier": f"bilgeval ifeval_verifier {ifeval_verifier.VERSION}"
    }
    assert instruction_verifier(english)["instruction_verifier"].startswith("google-research")
    assert instruction_verifier(other) == {}


def test_ifeval_checker_regressions():
    from bilgeval.ifeval_verifier.verifier import verifier_logic as v

    assert v.count_bullets_star("  * a\n  * b\n* c") == 3
    assert v.verify_3("ılık Isparta", None, letter="I", num_letter=3)["passed"] is True
    assert v.verify_3("iki İzmir", None, letter="I", num_letter=0)["passed"] is True
    assert v.verify_14("Cevabım evet. Çünkü öyle.", None, options=["evet", "hayır"])["passed"] is True
    assert v.verify_19("a\n******\nb", None)["passed"] is True
    assert v.verify_19("a\n******\nb\n******\nc", None)["passed"] is False
    assert v.verify_19("aynı\n******\naynı", None)["passed"] is False


def test_graphwalks_parser_matches_official_extraction():
    from bilgeval.evaluators import parse_graphwalks_nodes

    assert parse_graphwalks_nodes("reasoning...\nFinal Answer: [975511e631]") == ["975511e631"]
    assert parse_graphwalks_nodes("Final Answer: [a, b]\nActually\nFinal Answer: [c]") == ["c"]
    assert parse_graphwalks_nodes("Depth 1 gives [aaa, bbb].\nI cannot determine the rest.") == []
    assert parse_graphwalks_nodes("The parents are abc and def") == []
    assert parse_graphwalks_nodes("Final Answer: []") == []
    assert parse_graphwalks_nodes("Final Answer: ['x', \"y\"]") == ["x", "y"]
