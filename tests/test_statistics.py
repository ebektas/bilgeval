import unittest
from types import SimpleNamespace

from bilgeval.statistics import (
    bootstrap_accuracy_interval,
    build_multi_model_comparisons,
    extract_sample_outcomes,
    paired_accuracy_delta,
    pass_at_k_estimate,
    subgroup_metrics,
)


def outcomes(values):
    return [
        {
            "sample_id": sample_id,
            "index": index,
            "scored": value is not None,
            "correct": value,
        }
        for index, (sample_id, value) in enumerate(values)
    ]


class StatisticsTests(unittest.TestCase):
    def test_pass_at_k_uses_unbiased_combinatorial_estimator(self):
        self.assertAlmostEqual(pass_at_k_estimate(3, 1, 1), 1 / 3)
        self.assertAlmostEqual(pass_at_k_estimate(3, 1, 2), 2 / 3)
        self.assertEqual(pass_at_k_estimate(3, 1, 3), 1.0)
        self.assertEqual(pass_at_k_estimate(3, 0, 3), 0.0)

    def test_accuracy_bootstrap_is_deterministic_and_uses_scored_count(self):
        first = bootstrap_accuracy_interval(
            [True, True, False, False],
            resamples=2_000,
            seed="fixture-seed",
            seed_context="fixture-task",
        )
        second = bootstrap_accuracy_interval(
            [True, True, False, False],
            resamples=2_000,
            seed="fixture-seed",
            seed_context="fixture-task",
        )

        self.assertEqual(first, second)
        self.assertEqual(first["point_estimate"], 0.5)
        self.assertEqual(first["num_scored_examples"], 4)
        self.assertLessEqual(first["lower"], 0.5)
        self.assertGreaterEqual(first["upper"], 0.5)

    def test_sample_outcomes_keep_failures_separate_from_accuracy(self):
        extracted = extract_sample_outcomes(
            [
                {"idx": 0, "id": "one", "failed": False, "correct": True},
                {"idx": 1, "id": "two", "failed": True, "error": "timeout"},
            ]
        )

        self.assertEqual(extracted[0]["correct"], True)
        self.assertFalse(extracted[1]["scored"])
        self.assertIsNone(extracted[1]["correct"])

    def test_duplicate_sample_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate sample id"):
            extract_sample_outcomes(
                [
                    {"idx": 0, "id": "same", "failed": False, "correct": True},
                    {"idx": 1, "id": "same", "failed": False, "correct": False},
                ]
            )

    def test_paired_delta_counts_unscored_samples_as_incorrect(self):
        model_a = outcomes(
            [("one", True), ("two", True), ("three", False), ("four", False)]
        )
        model_b = outcomes(
            [("one", True), ("two", False), ("three", True), ("four", None)]
        )

        comparison = paired_accuracy_delta(
            model_a,
            model_b,
            model_a="checkpoint-a",
            model_b="checkpoint-b",
            dataset_name="fixture",
            resamples=2_000,
            seed="fixture-seed",
        )

        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["num_paired_examples"], 4)
        self.assertEqual(comparison["num_paired_scored"], 3)
        self.assertEqual(comparison["num_model_b_unscored"], 1)
        self.assertEqual(comparison["model_a_wins"], 1)
        self.assertEqual(comparison["model_b_wins"], 1)
        self.assertEqual(comparison["ties"], 2)
        self.assertEqual(comparison["delta_model_b_minus_model_a"], 0.0)

    def test_paired_delta_rejects_different_selected_sample_sets(self):
        comparison = paired_accuracy_delta(
            outcomes([("one", True)]),
            outcomes([("two", True)]),
            model_a="checkpoint-a",
            model_b="checkpoint-b",
            dataset_name="fixture",
            resamples=100,
        )

        self.assertEqual(comparison["status"], "incompatible_sample_sets")
        self.assertFalse(comparison["sample_sets_match"])

    def test_multi_model_comparison_builds_every_model_pair(self):
        reports = [
            {
                "model": "a",
                "_sample_outcomes": {"task": outcomes([("one", True), ("two", False)])},
            },
            {
                "model": "b",
                "_sample_outcomes": {"task": outcomes([("one", True), ("two", True)])},
            },
            {
                "model": "c",
                "_sample_outcomes": {"task": outcomes([("one", False), ("two", True)])},
            },
        ]

        summary = build_multi_model_comparisons(reports, resamples=100, seed="fixture")

        self.assertEqual(len(summary["comparisons"]), 3)
        self.assertEqual(
            summary["comparisons"]["a__vs__b"]["datasets"]["task"][
                "delta_model_b_minus_model_a"
            ],
            0.5,
        )

    def test_subgroup_metrics_support_metadata_fallback_and_missing_values(self):
        examples = [
            SimpleNamespace(raw={"metadata": {"subject": "math"}}),
            SimpleNamespace(raw={"metadata": {"subject": "math"}}),
            SimpleNamespace(raw={"metadata": {"subject": "geography"}}),
            SimpleNamespace(raw={"metadata": {}}),
        ]
        results = [
            {"idx": 0, "failed": False, "correct": True},
            {"idx": 1, "failed": False, "correct": False},
            {"idx": 2, "failed": False, "correct": True},
            {"idx": 3, "failed": False, "correct": False},
        ]

        report = subgroup_metrics(
            examples,
            results,
            ["subject"],
            resamples=200,
            seed="fixture",
        )["subject"]

        self.assertEqual(report["num_groups"], 2)
        self.assertEqual(report["num_missing_examples"], 1)
        self.assertEqual(report["micro_accuracy"], 0.5)
        self.assertEqual(report["macro_accuracy"], 0.75)
        math = next(group for group in report["groups"] if group["value"] == "math")
        self.assertEqual(math["num_scored_examples"], 2)
        self.assertEqual(math["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()


def test_counted_correctness_treats_unscored_as_incorrect():
    from bilgeval.statistics import counted_correctness, extract_sample_outcomes, scored_correctness

    results = [
        {"id": "a", "correct": True, "failed": False},
        {"id": "b", "correct": False, "failed": False},
        {"id": "c", "failed": True, "error": "Empty content after 900 characters of reasoning"},
    ]
    outcomes = extract_sample_outcomes(results)
    assert scored_correctness(outcomes) == [True, False]
    assert counted_correctness(outcomes) == [True, False, False]
