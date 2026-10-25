import unittest

from bilgeval.evaluators import (
    GeneratedTextResult,
    TextGenerationModelClient,
    build_answer_verifier_prompt,
    evaluate_exact_match_mc,
    evaluate_free_response,
    extract_final_integer,
    parse_answer_verifier_response,
    parse_free_response_prediction,
)
from bilgeval.reader import Choice, Example, FreeResponseExample


class PassFixtureClient(TextGenerationModelClient):
    """One value per request, in order: with one example, one value per sample."""

    def __init__(self, values):
        self.values = iter(values)
        self.requests = []

    def generate_texts(self, prompts, **kwargs):
        results = []
        for _ in prompts:
            self.requests.append(kwargs)
            value = next(self.values)
            results.append(value if isinstance(value, GeneratedTextResult)
                           else GeneratedTextResult(text=value, request_id=f"req-{len(self.requests)}"))
        return results


class StaticFixtureClient(TextGenerationModelClient):
    def __init__(self, value):
        self.value = value
        self.prompts = []

    def generate_texts(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        return [GeneratedTextResult(text=self.value) for _ in prompts]


class SequentialFixtureClient(TextGenerationModelClient):
    def __init__(self, values):
        self.values = iter(values)
        self.prompts = []

    def generate_texts(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        return [GeneratedTextResult(text=next(self.values)) for _ in prompts]


class MultiSampleEvaluatorTests(unittest.TestCase):
    def test_answer_verifier_prompt_has_task_specific_rules(self):
        mc_prompt = build_answer_verifier_prompt("B: Mavi", "Cevap B")
        numeric_prompt = build_answer_verifier_prompt("42", "Sonu\u00e7: 42,0")

        self.assertIn("OPTION_LABEL: OPTION_TEXT", mc_prompt)
        self.assertIn("flawed reasoning", mc_prompt)
        self.assertIn("decimal separators", numeric_prompt)
        self.assertIn("42,0", numeric_prompt)

    def test_regex_then_llm_is_rejected(self):
        example = FreeResponseExample(
            id="removed-mode",
            question="Kirk iki yaz.",
            answer="42",
            answer_text="42",
            raw={},
        )
        with self.assertRaisesRegex(ValueError, "Unsupported answer verification mode"):
            evaluate_free_response(
                [example],
                StaticFixtureClient("Yanit: 42"),
                generation_prompt="turkish_integer_answer_v1",
                answer_verification="regex_then_llm",
                verifier_client=StaticFixtureClient('{"match":true}'),
            )

    def test_llm_verifier_overrides_successful_mc_regex_match(self):
        example = Example(
            id="mc-llm-only",
            question="Hangisi dogru?",
            choices=[
                Choice(label="A", text="Kirmizi"),
                Choice(label="B", text="Mavi"),
            ],
            answer_label="A",
            answer_text="Kirmizi",
            raw={},
        )
        verifier = StaticFixtureClient('{"match":false}')
        report = evaluate_exact_match_mc(
            [example],
            StaticFixtureClient("Cevap: A"),
            append=True,
            score_target="label",
            match_mode="regex_match",
            answer_verification="llm_as_regex",
            verifier_client=verifier,
        )

        sample = report["results"][0]["samples"][0]
        self.assertEqual(len(verifier.prompts), 1)
        self.assertTrue(sample["answer_verification"]["regex_match"])
        self.assertTrue(sample["answer_verification"]["llm_checked"])
        self.assertFalse(sample["answer_verification"]["llm_match"])
        self.assertFalse(report["results"][0]["correct"])

    def test_malformed_verifier_response_is_retried_once(self):
        example = FreeResponseExample(
            id="math-verifier-retry",
            question="Kirk iki yaz.",
            answer="42",
            answer_text="42",
            raw={},
        )
        verifier = SequentialFixtureClient(["yes", '{"match":true}'])

        report = evaluate_free_response(
            [example],
            StaticFixtureClient("Yanit: 42"),
            generation_prompt="turkish_integer_answer_v1",
            answer_verification="llm_as_regex",
            verifier_client=verifier,
        )

        self.assertEqual(report["num_verifier_requests"], 2)
        self.assertEqual(report["num_failed_examples"], 0)
        self.assertEqual(report["correct"], 1)
        self.assertIn("CLASSIFICATION RETRY", verifier.prompts[1])

    def test_malformed_verifier_response_can_use_second_retry(self):
        example = FreeResponseExample(
            id="math-verifier-second-retry",
            question="Kirk iki yaz.",
            answer="42",
            answer_text="42",
            raw={},
        )
        verifier = SequentialFixtureClient(["yes", "still malformed", '{"match":true}'])

        report = evaluate_free_response(
            [example],
            StaticFixtureClient("Yanit: 42"),
            generation_prompt="turkish_integer_answer_v1",
            answer_verification="llm_as_regex",
            verifier_client=verifier,
        )

        self.assertEqual(report["num_verifier_requests"], 3)
        self.assertEqual(report["num_failed_examples"], 0)
        self.assertEqual(report["correct"], 1)

    def test_llm_verifier_overrides_successful_free_response_regex_match(self):
        example = FreeResponseExample(
            id="math-llm-only",
            question="Kirk iki yaz.",
            answer="42",
            answer_text="42",
            raw={},
        )
        verifier = StaticFixtureClient('{"match":false}')
        report = evaluate_free_response(
            [example],
            StaticFixtureClient("Yanit: 42"),
            generation_prompt="turkish_integer_answer_v1",
            answer_verification="llm_as_regex",
            verifier_client=verifier,
        )

        sample = report["results"][0]["samples"][0]
        self.assertEqual(len(verifier.prompts), 1)
        self.assertTrue(sample["answer_verification"]["regex_match"])
        self.assertTrue(sample["answer_verification"]["llm_checked"])
        self.assertFalse(sample["answer_verification"]["llm_match"])
        self.assertFalse(report["results"][0]["correct"])

    def test_free_response_integer_parser_prefers_marked_final_answer(self):
        response = (
            "Aylık maaş 4000 dolardır. Toplam ödeme 260 * 4000 = 1.040.000 dolardır. "
            "Cevap 1.040.000'dir."
        )
        self.assertEqual(extract_final_integer(response), 1_040_000)
        self.assertEqual(
            parse_free_response_prediction(response, "1040000", "regex_match"),
            "1040000",
        )
        self.assertEqual(
            parse_free_response_prediction(response, "1,040,000", "regex_match"),
            "1,040,000",
        )

    def test_free_response_integer_parser_does_not_accept_intermediate_gold(self):
        response = "Önce 42 buldum ama düzelttim. Cevap 41'dir."
        self.assertEqual(extract_final_integer(response), 41)
        self.assertIsNone(parse_free_response_prediction(response, "42", "regex_match"))

    def test_answer_verifier_response_requires_one_boolean_field(self):
        self.assertTrue(parse_answer_verifier_response('{"match":true}'))
        self.assertFalse(parse_answer_verifier_response('```json\n{"match": false}\n```'))
        self.assertIsNone(parse_answer_verifier_response('{"match":"true"}'))
        self.assertIsNone(parse_answer_verifier_response('{"match":true,"reason":"same"}'))
        self.assertIsNone(parse_answer_verifier_response("MATCH"))

    def test_mc_malformed_verifier_response_is_a_coverage_failure(self):
        example = Example(
            id="mc-verifier-error",
            question="Hangisi dogru?",
            choices=[
                Choice(label="A", text="Kirmizi"),
                Choice(label="B", text="Mavi"),
            ],
            answer_label="A",
            answer_text="Kirmizi",
            raw={},
        )
        report = evaluate_exact_match_mc(
            [example],
            StaticFixtureClient("Ilk secenegi tercih ediyorum."),
            append=True,
            score_target="label",
            match_mode="regex_match",
            answer_verification="llm_as_regex",
            verifier_client=StaticFixtureClient("yes"),
        )

        self.assertEqual(report["num_failed_examples"], 1)
        self.assertEqual(report["num_scored_examples"], 0)
        self.assertEqual(report["answer_verification"]["num_failed_checks"], 1)

    def test_mc_llm_as_regex_verifies_every_response(self):
        examples = [
            Example(
                id="parsed-correct",
                question="Dogru secenek?",
                choices=[Choice(label="A", text="Dogru"), Choice(label="B", text="Yanlis")],
                answer_label="A",
                answer_text="Dogru",
                raw={},
            ),
            Example(
                id="parsed-wrong",
                question="Dogru secenek?",
                choices=[Choice(label="A", text="Dogru"), Choice(label="B", text="Yanlis")],
                answer_label="A",
                answer_text="Dogru",
                raw={},
            ),
        ]
        verifier = StaticFixtureClient('{"match":true}')

        class PerPromptClient(TextGenerationModelClient):
            def generate_texts(self, prompts, **kwargs):
                return [
                    GeneratedTextResult(text="Cevap: A"),
                    GeneratedTextResult(text="Cevap: B"),
                ]

        report = evaluate_exact_match_mc(
            examples,
            PerPromptClient(),
            append=True,
            score_target="label",
            match_mode="regex_match",
            answer_verification="llm_as_regex",
            verifier_client=verifier,
        )

        self.assertEqual(len(verifier.prompts), 2)
        self.assertEqual(report["answer_verification"]["mode"], "llm_as_regex")
        self.assertEqual(report["answer_verification"]["num_candidates_checked"], 2)
        self.assertEqual(report["correct"], 2)
        self.assertTrue(
            all(
                result["samples"][0]["answer_verification"]["llm_checked"]
                for result in report["results"]
            )
        )

    def test_malformed_verifier_response_is_a_coverage_failure(self):
        example = FreeResponseExample(
            id="math-format",
            question="Yarimi yaz.",
            answer="0.5",
            answer_text="0.5",
            raw={},
        )
        report = evaluate_free_response(
            [example],
            StaticFixtureClient(r"\frac{1}{2}"),
            generation_prompt="turkish_integer_answer_v1",
            answer_verification="llm_as_regex",
            verifier_client=StaticFixtureClient("yes"),
        )

        self.assertEqual(report["num_failed_examples"], 1)
        self.assertEqual(report["num_scored_examples"], 0)
        self.assertEqual(report["answer_verification"]["num_failed_checks"], 1)

    def test_mc_consensus_and_pass_at_k_preserve_one_primary_result(self):
        example = Example(
            id="mc-1",
            question="Dogru secenek?",
            choices=[Choice(label="A", text="Yanlis"), Choice(label="B", text="Dogru")],
            answer_label="B",
            answer_text="Dogru",
            raw={},
        )
        client = PassFixtureClient(["A", "B", "B"])
        progress = []

        report = evaluate_exact_match_mc(
            [example],
            client,
            append=True,
            score_target="label",
            generation_settings={"temperature": 0.7},
            num_samples=3,
            pass_at_k=(1, 2, 3),
            aggregation="consensus",
            progress_callback=lambda completed, total, requests: progress.append(
                (completed, total, requests)
            ),
        )

        self.assertEqual(len(client.requests), 3)
        self.assertTrue(all("seed" not in kwargs for kwargs in client.requests))
        self.assertEqual(progress[-1], (3, 3, 3))
        self.assertEqual(len(report["results"]), 1)
        result = report["results"][0]
        self.assertEqual(len(result["samples"]), 3)
        self.assertEqual(result["prediction_label"], "B")
        self.assertTrue(result["correct"])
        self.assertEqual(result["consensus"]["vote_counts"], {"A": 1, "B": 2})
        self.assertAlmostEqual(result["sample_metrics"]["pass_at_k"]["1"], 2 / 3)
        self.assertEqual(result["sample_metrics"]["pass_at_k"]["2"], 1.0)
        self.assertEqual(report["multi_sample_metrics"]["consensus_accuracy"], 1.0)

    def test_first_aggregation_counts_failed_attempt_as_incorrect(self):
        example = Example(
            id="mc-2",
            question="Dogru secenek?",
            choices=[Choice(label="A", text="Yanlis"), Choice(label="B", text="Dogru")],
            answer_label="B",
            answer_text="Dogru",
            raw={},
        )
        client = PassFixtureClient(
            [GeneratedTextResult(error="endpoint unavailable"), "B", "A"]
        )

        report = evaluate_exact_match_mc(
            [example],
            client,
            append=True,
            score_target="label",
            generation_settings={"temperature": 0.7},
            num_samples=3,
            pass_at_k=(1, 2),
            aggregation="first",
        )

        self.assertTrue(report["results"][0]["failed"])
        self.assertEqual(report["num_failed_examples"], 1)
        self.assertAlmostEqual(report["multi_sample_metrics"]["average_sample_accuracy"], 1 / 3)
        self.assertAlmostEqual(report["multi_sample_metrics"]["pass_at_k"]["2"], 2 / 3)

    def test_free_response_consensus_uses_canonical_integer_keys(self):
        example = FreeResponseExample(
            id="math-1",
            question="Kirk iki yaz.",
            answer="42",
            answer_text="42",
            raw={},
        )
        client = PassFixtureClient(["41", "Yanıt: 42", "042"])

        report = evaluate_free_response(
            [example],
            client,
            generation_prompt="turkish_integer_answer_v1",
            generation_settings={"temperature": 0.8},
            num_samples=3,
            pass_at_k=(1, 2),
            aggregation="consensus",
        )

        result = report["results"][0]
        self.assertTrue(result["correct"])
        self.assertEqual(result["consensus"]["prediction_key"], "42")
        self.assertEqual(result["consensus"]["vote_counts"], {"41": 1, "42": 2})
        self.assertAlmostEqual(report["multi_sample_metrics"]["pass_at_k"]["1"], 2 / 3)


class SampleMatrixTests(unittest.TestCase):
    def test_all_samples_go_out_in_one_request_list(self):
        from bilgeval.batching import generate_sample_matrix

        calls = []

        class Recorder:
            def generate_texts(self, prompts, **kwargs):
                calls.append(list(prompts))
                return [GeneratedTextResult(text=f"{prompt}#{len(calls)}") for prompt in prompts]

        matrix, _ = generate_sample_matrix(
            prompts=["p1", "p2"],
            model_client=Recorder(),
            max_tokens=8,
            generation_settings={"temperature": 0.7},
            num_samples=3,
            request_batch_size=64,
            error_prefix="test",
            progress_callback=None,
        )

        self.assertEqual(calls, [["p1", "p2", "p1", "p2", "p1", "p2"]])
        self.assertEqual([[r.text for r in row] for row in matrix], [["p1#1"] * 3, ["p2#1"] * 3])


if __name__ == "__main__":
    unittest.main()


class ParserRegressionTests(unittest.TestCase):
    def test_label_regex_takes_last_explicit_answer_and_ignores_inflections(self):
        from bilgeval.evaluators import parse_regex_match_prediction

        example = Example(
            id="x",
            question="q",
            choices=[Choice(label=l, text=t) for l, t in (("A", "a1"), ("B", "b1"), ("C", "c1"))],
            answer_label="C",
            answer_text="c1",
            raw={},
        )
        parse = lambda text, **kw: parse_regex_match_prediction(text, example, score_target="label", **kw)
        self.assertEqual(parse("Bence cevap: B değil, C")[0], "C")
        self.assertEqual(parse("Bu yanıta göre doğru şık **C**'dir. Cevap: C")[0], "C")
        self.assertIsNone(parse("Bu yanıta göre hiçbir şey belli değil."))
        self.assertEqual(parse("Cevap C'dir.")[0], "C")
        self.assertEqual(parse("Yanıt: (B)")[0], "B")
        self.assertEqual(parse("Answer: $A$", allow_fallback=False)[0], "A")
        self.assertEqual(parse("Yanıt: $B$", allow_fallback=False)[0], "B")
        self.assertIsNone(parse("A noktasından B noktasına gidiyoruz ama sonuç yok x", allow_fallback=False))
        self.assertEqual(parse("Sonuç olarak B")[0], "B")
        self.assertEqual(parse("Cevap: B. Çünkü A ve C seçenekleri yanlıştır.")[0], "B")
        self.assertEqual(parse("Cevap: A\nGerekçe... Cevap: C")[0], "C")

    def test_judge_verdict_and_pass_fail_take_the_last_verdict(self):
        from bilgeval.evaluators import parse_judge_verdict
        from bilgeval.specialized_tasks import parse_pass_fail

        self.assertEqual(parse_judge_verdict("CORRECT/INCORRECT: INCORRECT"), "INCORRECT")
        self.assertEqual(parse_judge_verdict("Karar: CORRECT"), "CORRECT")
        self.assertIs(parse_pass_fail("PASS or FAIL? The answer is FAIL"), False)
        self.assertIs(parse_pass_fail("Karar: PASS"), True)

    def test_final_integer_extraction_regressions(self):
        cases = {
            "Sonuç: 18.": 18,
            "Toplam: 18, yani bitti": 18,
            "Cevap -5'tir": -5,
            "Cevap: 5 - 3 = 2": 2,
            "Cevap: 41. Bekle, 42 olmalı.": 42,
            "Hesap 12 * 3 = 36\\n\\boxed{36}": 36,
            "Cevabı 12'dir": 12,
            "Cevap 1.000'dir": 1000,
            "The answer is 18.": 18,
        }
        for text, expected in cases.items():
            self.assertEqual(extract_final_integer(text), expected, text)

    def test_mtbench_reference_solution_for_mr_gr(self):
        from bilgeval.reader import MultiTurnJudgeExample
        from bilgeval.specialized_tasks import mtbench_judge_messages

        turns = [
            {"role": "human", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "human", "content": "times 3?"},
            {"role": "assistant", "content": "12"},
        ]
        def make(task_id):
            return MultiTurnJudgeExample(
                id="d", task_id=task_id, task_name="t", topic="math", turns=turns,
                eval_turn_positions=[0, 1], evaluation_initial="init", evaluation_prompt="rubric",
                scoring_section="", scoring_format="DIALOGUE", raw={},
            ) if False else MultiTurnJudgeExample(
                id="d", task_id=task_id, task_name="t", topic="math", turns=turns,
                eval_turn_positions=[0, 1], evaluation_initial="init", evaluation_prompt="rubric",
                scoring_format="DIALOGUE", raw={},
            )
        with_ref = mtbench_judge_messages(make("MR"), 1, "candidate")[1]["content"]
        without_ref = mtbench_judge_messages(make("CM"), 1, "candidate")[1]["content"]
        self.assertIn("### reference solution ###\n12\n###", with_ref)
        self.assertNotIn("reference solution", without_ref)
