import unittest
from copy import deepcopy

from bilgeval.main import resolve_benchmark_runs
from bilgeval.manifest import leaderboard_fingerprints


PRETRAIN_SAMPLE_SEED = "bilge-pt-v1"
PRETRAIN_MC_DATASETS = (
    "tr_arc_challenge_translated",
    "tr_mmlu_pro_translated",
    "tr_mmlu_redux_translated",
    "tr_turkish_mmlu_5shot_eval",
    "tr_gpqa_diamond_translated",
    "tr_supergpqa_translated",
    "tr_cultural_commonsense_qa",
    "tr_hellaswag",
    "tr_geography",
    "tr_proverbs",
    "tr_superstitions",
    "tr_movies",
    "commonsense_qa_validation",
)


class TaskVariantTests(unittest.TestCase):
    def test_leaderboard_identity_ignores_suite_and_groups_scorer_variants(self):
        run = {
            "run_name": "suite_a__fixture",
            "requested_target": "suite_a",
            "suite": {"name": "suite_a", "version": "1"},
            "dataset": {"name": "fixture", "content_hash": "sha256:data"},
            "formatter": {"id": "mc", "append_choices": True},
            "request_type": "logprob",
            "scorer": {"eval_mode": "logprob", "score_target": "label"},
            "filters": [],
            "generation": {"temperature": 0.0},
            "sampling": {"selected_example_ids_fingerprint": "sha256:selection"},
            "few_shot": {"requested_count": 0},
            "subgroup_fields": [],
        }
        suite_variant = deepcopy(run)
        suite_variant["run_name"] = "suite_b__fixture"
        suite_variant["requested_target"] = "suite_b"
        suite_variant["suite"] = {"name": "suite_b", "version": "4"}

        self.assertEqual(
            leaderboard_fingerprints(run),
            leaderboard_fingerprints(suite_variant),
        )

        scorer_variant = deepcopy(run)
        scorer_variant["scorer"]["score_target"] = "text"
        evaluation, comparison = leaderboard_fingerprints(run)
        variant_evaluation, variant_comparison = leaderboard_fingerprints(scorer_variant)
        self.assertNotEqual(evaluation, variant_evaluation)
        self.assertEqual(comparison, variant_comparison)



if __name__ == "__main__":
    unittest.main()
