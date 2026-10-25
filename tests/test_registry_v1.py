import contextlib
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import bilgeval.main as main_module

from bilgeval.generation_prompts import GENERATION_PROMPTS
from bilgeval.judge_prompts import JUDGE_PROMPTS
from bilgeval.manifest import stable_json_sha256
from bilgeval.main import (
    ACTIVE_MODEL_PROFILES,
    ACTIVE_REGISTRY_CONTEXT,
    load_registry_file,
    resolve_benchmark_runs,
    run_benchmark,
    resolve_model_profile,
)
from bilgeval.cli import run_inspect_command, run_validate_command
from bilgeval.registry import DATASETS
from bilgeval.registry_bundle import RegistryBundle
from bilgeval.evaluators import CompletionScoreResult, GeneratedTextResult, LogprobModelClient


FIXTURES = Path(__file__).resolve().parent / "fixtures"
REGISTRY_PATH = FIXTURES / "registry_v1.json"
BUILTIN_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "bilgeval" / "builtin_registry.json"
)


class RegistryV1Tests(unittest.TestCase):
    def setUp(self):
        self.datasets = dict(DATASETS)
        self.prompts = dict(GENERATION_PROMPTS)
        self.judge_prompts = dict(JUDGE_PROMPTS)
        self.registry_context = dict(ACTIVE_REGISTRY_CONTEXT)
        self.model_profiles = dict(ACTIVE_MODEL_PROFILES)
        self.registry_bundle = main_module.ACTIVE_REGISTRY_BUNDLE

    def tearDown(self):
        DATASETS.clear()
        DATASETS.update(self.datasets)
        GENERATION_PROMPTS.clear()
        GENERATION_PROMPTS.update(self.prompts)
        JUDGE_PROMPTS.clear()
        JUDGE_PROMPTS.update(self.judge_prompts)
        ACTIVE_REGISTRY_CONTEXT.clear()
        ACTIVE_REGISTRY_CONTEXT.update(self.registry_context)
        ACTIVE_MODEL_PROFILES.clear()
        ACTIVE_MODEL_PROFILES.update(self.model_profiles)
        main_module.ACTIVE_REGISTRY_BUNDLE = self.registry_bundle

    def test_registry_loads_tasks_and_suite_into_native_runtime(self):
        load_registry_file(str(REGISTRY_PATH))

        self.assertEqual(DATASETS["fixture_mc"].schema_version, "bilgeval.instance/v1")
        self.assertIn("fixture_mc_logprob", main_module.ACTIVE_REGISTRY_BUNDLE.tasks)
        self.assertIn("fixture_mc:cloze-pmi", main_module.ACTIVE_REGISTRY_BUNDLE.tasks)
        runs = resolve_benchmark_runs(["fixture_suite"])
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0].task_name, "fixture_mc_logprob")
        self.assertEqual(runs[0].run_name, "fixture_mc_logprob__max1")
        self.assertEqual(runs[1].run_name, "fixture_math_regex")
        self.assertEqual(runs[0].task_spec.few_shot.dataset, "fixture_mc_few_shot")
        self.assertEqual(runs[0].task_spec.few_shot.seed, "fixture-few-shot-v1")
        self.assertEqual(runs[0].task_spec.formatter.version, "1.0.0")
        self.assertEqual(runs[0].task_spec.resolved_request_type, "logprob")
        self.assertEqual(runs[0].task_spec.subgroup_fields, ["subject"])
        self.assertEqual(runs[0].max_examples, 1)
        pmi_run = resolve_benchmark_runs(["fixture_mc:cloze-pmi"])[0]
        self.assertEqual(pmi_run.task_spec.scorer.score_target, "text")
        self.assertEqual(pmi_run.task_spec.scorer.normalization, "pmi")
        self.assertEqual(pmi_run.task_spec.scorer.pmi_reference_prompt, "Yanıt:")
        self.assertFalse(pmi_run.task_spec.formatter.append_choices)
        profile_name, profile = resolve_model_profile("google/gemma-3-27b-it")
        self.assertEqual(profile_name, "gemma_completion")
        self.assertEqual(profile.bos_mode, "auto")
        self.assertEqual(profile.serialization.tokenizer_revision, "fixture-revision")
        self.assertEqual(profile.serialization.logprob.mode, "template")

    def test_registry_rejects_scorer_dataset_type_mismatch(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        raw["tasks"]["fixture_math_regex"]["scorer"]["id"] = "mc_regex_match"
        with self.assertRaises(ValueError):
            RegistryBundle.model_validate(raw)

    def test_a_registry_with_only_model_profiles_loads(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        for key in ("datasets", "tasks", "suites"):
            raw[key] = {}

        bundle = RegistryBundle.model_validate(raw)

        self.assertEqual(bundle.tasks, {})
        self.assertIn("gemma_completion", bundle.model_profiles)

    def test_registry_accepts_llm_verification_for_mc_regex_task(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        task = raw["tasks"]["fixture_mc_logprob"]
        task["request_type"] = "generate"
        task["generation_api"] = "chat"
        task["scorer"] = {
            "id": "mc_regex_match",
            "score_target": "label",
            "answer_verification": "llm_as_regex",
        }

        bundle = RegistryBundle.model_validate(raw)

        self.assertEqual(
            bundle.tasks["fixture_mc_logprob"].scorer.answer_verification,
            "llm_as_regex",
        )

    def test_registry_maps_legacy_llm_alias_and_rejects_regex_then_llm(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        task = raw["tasks"]["fixture_mc_logprob"]
        task["request_type"] = "generate"
        task["generation_api"] = "chat"
        task["scorer"] = {
            "id": "mc_regex_match",
            "score_target": "label",
            "answer_verification": "llm",
        }
        bundle = RegistryBundle.model_validate(raw)
        self.assertEqual(
            bundle.tasks["fixture_mc_logprob"].scorer.answer_verification,
            "llm_as_regex",
        )

        task["scorer"]["answer_verification"] = "regex_then_llm"
        with self.assertRaises(ValueError):
            RegistryBundle.model_validate(raw)


    def test_registry_preserves_reader_mapping_and_installs_typed_judge_prompt(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        raw["datasets"]["fixture_mc"]["reader"] = {
            "question_key": "legacy_question",
            "choices_key": "legacy_choices",
            "answer_key": "legacy_answer",
            "question_only_before_newline": True,
        }
        raw["judge_prompts"] = {
            "fixture_judge": {
                "template": (
                    "{question}\n{choices}\n{gold_label}\n{gold_text}\n"
                    "{candidate_answer}"
                ),
                "output_type": "binary_correctness",
            }
        }
        judge_task = raw["tasks"]["fixture_mc_logprob"]
        judge_task["request_type"] = "generate"
        judge_task["scorer"] = {
            "id": "mc_judge",
            "score_target": "label",
            "judge_prompt": "fixture_judge",
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            load_registry_file(str(path))

        self.assertEqual(DATASETS["fixture_mc"].question_key, "legacy_question")
        self.assertTrue(DATASETS["fixture_mc"].question_only_before_newline)
        self.assertEqual(JUDGE_PROMPTS["fixture_judge"].output_type, "binary_correctness")

    def test_registry_rejects_missing_custom_judge_prompt(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        task = raw["tasks"]["fixture_mc_logprob"]
        task["request_type"] = "generate"
        task["scorer"] = {
            "id": "mc_judge",
            "score_target": "label",
            "judge_prompt": "missing_judge",
        }
        with self.assertRaisesRegex(ValueError, "unknown judge prompt"):
            RegistryBundle.model_validate(raw)



    def test_legacy_registry_is_rejected(self):
        legacy_path = Path(__file__).resolve().parent / "fixtures" / "legacy_smoke_registry.json"
        with self.assertRaisesRegex(ValueError, "not supported"):
            load_registry_file(str(legacy_path))

    def test_registry_rejects_multiple_choice_task_without_rendered_choices(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        raw["tasks"]["fixture_mc_logprob"]["formatter"]["append_choices"] = False
        with self.assertRaises(ValueError):
            RegistryBundle.model_validate(raw)

    def test_registry_rejects_request_type_scorer_mismatch(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        raw["tasks"]["fixture_mc_logprob"]["request_type"] = "generate"
        with self.assertRaisesRegex(ValueError, "request_type"):
            RegistryBundle.model_validate(raw)

    def test_registry_rejects_generation_api_on_logprob_task(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        raw["tasks"]["fixture_mc_logprob"]["generation_api"] = "chat"
        with self.assertRaisesRegex(ValueError, "logprob task"):
            RegistryBundle.model_validate(raw)

    def test_registered_generation_api_is_required_for_unknown_model(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        raw["tasks"]["fixture_math_regex"]["generation_api"] = "auto"
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "registry.json"
            registry_path.write_text(json.dumps(raw), encoding="utf-8")
            load_registry_file(str(registry_path))
            with self.assertRaisesRegex(ValueError, "Generation API could not be resolved"):
                run_benchmark(
                    dataset_names=["fixture_math_regex"],
                    output_dir=directory,
                    model="unknown-model",
                    dry_run=True,
                )

    def test_registered_generation_api_conflicts_with_cli_override(self):
        load_registry_file(str(REGISTRY_PATH))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "conflicts with the registered benchmark"):
                run_benchmark(
                    dataset_names=["fixture_math_regex"],
                    output_dir=directory,
                    model="unknown-model",
                    generation_api="chat",
                    dry_run=True,
                )

    def test_suite_rejects_conflicting_registered_generation_apis(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        chat_task = dict(raw["tasks"]["fixture_math_regex"])
        chat_task["generation_api"] = "chat"
        raw["tasks"]["fixture_math_chat"] = chat_task
        raw["suites"]["mixed_generation_apis"] = {
            "version": "1.0.0",
            "tasks": [
                {"task": "fixture_math_regex"},
                {"task": "fixture_math_chat"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "registry.json"
            registry_path.write_text(json.dumps(raw), encoding="utf-8")
            load_registry_file(str(registry_path))
            with self.assertRaisesRegex(ValueError, "conflicting generation APIs"):
                run_benchmark(
                    dataset_names=["mixed_generation_apis"],
                    output_dir=directory,
                    model="unknown-model",
                    dry_run=True,
                )

    def test_registry_validates_generation_sampling_parameters(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        generation = raw["tasks"]["fixture_math_regex"]["generation"]
        generation.update(
            {
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": 50,
                "stop": ["</final>"],
            }
        )

        bundle = RegistryBundle.model_validate(raw)
        parsed = bundle.tasks["fixture_math_regex"].generation
        self.assertEqual(parsed.temperature, 0.6)
        self.assertEqual(parsed.top_p, 0.9)
        self.assertEqual(parsed.top_k, 50)
        self.assertEqual(parsed.stop, ["</final>"])

        generation["stop"] = ["", ""]
        with self.assertRaisesRegex(ValueError, "stop sequences"):
            RegistryBundle.model_validate(raw)

        generation["stop"] = ["</final>"]
        generation["seed"] = 17
        with self.assertRaisesRegex(ValueError, "seed"):
            RegistryBundle.model_validate(raw)

    def test_pmi_task_variant_requires_and_preserves_reference_prompt(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        task = raw["tasks"]["fixture_mc_logprob"]
        task["scorer"]["normalization"] = "pmi"
        with self.assertRaisesRegex(ValueError, "pmi_reference_prompt"):
            RegistryBundle.model_validate(raw)

        task["scorer"]["pmi_reference_prompt"] = "Yanıt:"
        bundle = RegistryBundle.model_validate(raw)
        self.assertEqual(
            bundle.tasks["fixture_mc_logprob"].scorer.pmi_reference_prompt,
            "Yanıt:",
        )

    def test_colon_named_task_variant_is_a_runnable_target(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        task = raw["tasks"].pop("fixture_mc_logprob")
        variant_name = "fixture_mc:label-logprob"
        raw["tasks"][variant_name] = task
        raw["suites"]["fixture_suite"]["tasks"][0]["task"] = variant_name
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            load_registry_file(str(path))
            runs = resolve_benchmark_runs([variant_name])

        self.assertEqual(runs[0].task_name, variant_name)
        self.assertEqual(runs[0].task_spec.resolved_request_type, "logprob")

    def test_legacy_model_profile_bos_flag_maps_to_policy(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        profile = raw["model_profiles"]["gemma_completion"]
        profile.pop("bos_mode")
        profile["add_bos_token"] = True

        bundle = RegistryBundle.model_validate(raw)

        self.assertEqual(
            bundle.model_profiles["gemma_completion"].resolved_bos_mode,
            "always",
        )

    def test_model_profile_rejects_ambiguous_serialization_templates(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        renderer = raw["model_profiles"]["gemma_completion"]["serialization"]["logprob"]
        renderer["template"] = "{prompt}\n{prompt}"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            RegistryBundle.model_validate(raw)

        renderer["template"] = "{prompt}"
        renderer["template_hash"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "template_hash"):
            RegistryBundle.model_validate(raw)

    def test_validate_inspect_and_dry_run_do_not_call_an_endpoint(self):
        validate_stdout = io.StringIO()
        with contextlib.redirect_stdout(validate_stdout):
            run_validate_command(["--registry-file", str(REGISTRY_PATH), "fixture_suite"])
        validation = json.loads(validate_stdout.getvalue())
        self.assertTrue(validation["valid"])
        self.assertTrue(all(item["content_hash_verified"] for item in validation["datasets"]))

        inspect_stdout = io.StringIO()
        with contextlib.redirect_stdout(inspect_stdout):
            run_inspect_command(
                ["--registry-file", str(REGISTRY_PATH), "--limit", "1", "fixture_suite"]
            )
        inspection = json.loads(inspect_stdout.getvalue())
        self.assertIn("Seçenekler", inspection["runs"][0]["examples"][0]["rendered_prompt"])

        with tempfile.TemporaryDirectory() as directory:
            report = run_benchmark(
                dataset_names=["fixture_suite"],
                output_dir=directory,
                model="offline-gemma-fixture-model",
                dry_run=True,
            )
            manifest = json.loads(Path(report["manifest_file"]).read_text(encoding="utf-8"))

        self.assertTrue(report["dry_run"])
        self.assertEqual(manifest["schema_version"], "bilgeval.run-manifest/v1")
        self.assertEqual(manifest["registry"]["schema_version"], "bilgeval.registry/v1")
        self.assertEqual(manifest["runtime"]["model_profile"]["name"], "gemma_completion")
        self.assertEqual(manifest["runtime"]["bos_mode"], "auto")
        self.assertTrue(manifest["runtime"]["model_profile"]["fingerprint"].startswith("sha256:"))
        self.assertEqual(
            manifest["runtime"]["serialization"]["tokenizer"]["revision"],
            "fixture-revision",
        )
        self.assertTrue(
            manifest["runtime"]["serialization"]["logprob"]["template_hash"].startswith(
                "sha256:"
            )
        )
        self.assertTrue(
            manifest["runtime"]["serialization"]["fingerprint"].startswith("sha256:")
        )
        self.assertEqual(manifest["runtime"]["statistics"]["bootstrap_resamples"], 10_000)
        self.assertEqual(manifest["runs"][0]["suite"], {"name": "fixture_suite", "version": "1.0.0"})
        self.assertTrue(manifest["runs"][0]["task"]["spec_fingerprint"].startswith("sha256:"))
        self.assertEqual(manifest["runs"][0]["request_type"], "logprob")
        self.assertEqual(manifest["runs"][0]["formatter"]["version"], "1.0.0")
        self.assertEqual(manifest["runs"][0]["few_shot"]["seed"], "fixture-few-shot-v1")
        self.assertEqual(manifest["runs"][0]["subgroup_fields"], ["subject"])
        self.assertTrue(
            manifest["runs"][0]["sampling"]["selected_example_ids_fingerprint"].startswith(
                "sha256:"
            )
        )
        self.assertEqual(len(manifest["runs"][0]["sampling"]["selected_example_ids"]), 1)
        self.assertEqual(
            manifest["runs"][0]["few_shot"]["selected_example_ids"],
            ["mc-demo-1"],
        )
        self.assertTrue(manifest["code"]["untracked_content_hash"].startswith("sha256:"))

    def test_cli_generation_overrides_are_fingerprinted_in_dry_run_manifest(self):
        load_registry_file(str(REGISTRY_PATH))
        with tempfile.TemporaryDirectory() as directory:
            report = run_benchmark(
                dataset_names=["fixture_math_regex"],
                output_dir=directory,
                model="fixture-model",
                generation_temperature=0.8,
                generation_top_p=0.95,
                generation_top_k=32,
                generation_stop=["</answer>"],
                generation_num_samples=3,
                generation_pass_at_k=[1, 2],
                generation_aggregation="consensus",
                dry_run=True,
            )
            manifest = json.loads(Path(report["manifest_file"]).read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["runs"][0]["generation"],
            {
                "candidate": {"disable_thinking": False, "extra_body": None},
                "max_tokens": 64,
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 32,
                "seed": None,
                "stop": ["</answer>"],
                "num_samples": 3,
                "pass_at_k": [1, 2],
                "aggregation": "consensus",
                "settings_fingerprint": manifest["runs"][0]["generation"]["settings_fingerprint"],
            },
        )

    def test_a_run_carries_no_seed_and_keeps_the_unseeded_fingerprints(self):
        load_registry_file(str(REGISTRY_PATH))
        with tempfile.TemporaryDirectory() as directory:
            report = run_benchmark(
                dataset_names=["fixture_math_regex"],
                output_dir=directory,
                model="fixture-model",
                generation_temperature=0.8,
                dry_run=True,
            )
            manifest = json.loads(Path(report["manifest_file"]).read_text(encoding="utf-8"))
        run = manifest["runs"][0]

        self.assertIsNone(run["generation"]["seed"])
        comparison = {
            "fingerprint_version": run["fingerprint_version"],
            **{
                key: run[key]
                for key in (
                    "dataset", "formatter", "request_type", "filters", "generation",
                    "sampling", "few_shot", "subgroup_fields",
                )
            },
        }
        self.assertEqual(run["comparison_group_fingerprint"], stable_json_sha256(comparison))
        self.assertEqual(
            run["evaluation_fingerprint"],
            stable_json_sha256({**comparison, "scorer": run["scorer"]}),
        )

    def test_registry_validates_multi_sample_generation(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        generation = raw["tasks"]["fixture_math_regex"]["generation"]
        generation.update(
            {
                "temperature": 0.7,
                "num_samples": 3,
                "pass_at_k": [1, 2, 3],
                "aggregation": "consensus",
            }
        )
        bundle = RegistryBundle.model_validate(raw)
        parsed = bundle.tasks["fixture_math_regex"].generation
        self.assertEqual(parsed.num_samples, 3)
        self.assertEqual(parsed.pass_at_k, [1, 2, 3])
        self.assertEqual(parsed.aggregation, "consensus")

        generation["temperature"] = 0.0
        with self.assertRaisesRegex(ValueError, "temperature"):
            RegistryBundle.model_validate(raw)

        generation["temperature"] = 0.7
        generation["pass_at_k"] = [4]
        with self.assertRaisesRegex(ValueError, "pass_at_k"):
            RegistryBundle.model_validate(raw)

        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        raw["tasks"]["fixture_mc_logprob"]["generation"] = {
            "temperature": 0.7,
            "num_samples": 2,
        }
        with self.assertRaisesRegex(ValueError, "non-generation"):
            RegistryBundle.model_validate(raw)

    def test_overlapping_model_profiles_require_explicit_selection(self):
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        raw["model_profiles"]["gemma_second"] = {
            **raw["model_profiles"]["gemma_completion"],
            "model_pattern": "gemma-3",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            load_registry_file(str(path))
            with self.assertRaisesRegex(ValueError, "multiple profiles"):
                resolve_model_profile("google/gemma-3-27b-it")

    def test_run_report_includes_reproducible_accuracy_statistics(self):
        class FixtureLogprobClient:
            model = "fixture-model"
            transport_name = "fixture"
            effective_add_bos_token = False
            bos_diagnostics = {"source": "fixture"}

            def score_prompt_completion_pairs(self, pairs, **kwargs):
                return [
                    CompletionScoreResult(
                        score=-0.1 if completion.strip() == "B" else -1.0,
                        num_completion_tokens=1,
                    )
                    for _, completion in pairs
                ]

        load_registry_file(str(REGISTRY_PATH))
        with tempfile.TemporaryDirectory() as directory, patch(
            "bilgeval.main.create_logprob_client",
            return_value=FixtureLogprobClient(),
        ):
            report = run_benchmark(
                dataset_names=["fixture_mc_logprob"],
                output_dir=directory,
                model="fixture-model",
                skip_preflight=True,
                collect_sample_outcomes=True,
                bootstrap_resamples=500,
                statistics_seed="fixture-statistics",
            )
            saved = json.loads(
                (Path(report["report_dir"]) / "final_report.json").read_text(
                    encoding="utf-8"
                )
            )
            response_artifact = next(iter(report["artifacts"]["responses"].values()))
            response_path = Path(report["report_dir"]) / response_artifact["path"]
            with gzip.open(response_path, "rt", encoding="utf-8") as file:
                response_rows = [json.loads(line) for line in file]

        run_name, dataset = next(iter(saved["datasets"].items()))
        self.assertEqual(saved["schema_version"], "bilgeval.report/v4")
        self.assertEqual(dataset["benchmark_name"], "fixture_mc")
        self.assertEqual(dataset["task_name"], "fixture_mc_logprob")
        self.assertIsNone(dataset["suite_name"])
        self.assertTrue(dataset["evaluation_fingerprint"].startswith("sha256:"))
        self.assertTrue(dataset["comparison_group_fingerprint"].startswith("sha256:"))
        self.assertEqual(dataset["accuracy"], 0.5)
        self.assertEqual(dataset["accuracy_denominator"], "all_examples")
        self.assertEqual(dataset["accuracy_confidence_interval"]["resamples"], 500)
        self.assertTrue(dataset["sample_outcome_fingerprint"].startswith("sha256:"))
        self.assertEqual(dataset["subgroup_metrics"]["subject"]["num_groups"], 2)
        self.assertEqual(dataset["subgroup_metrics"]["subject"]["macro_accuracy"], 0.5)
        self.assertEqual(len(report["_sample_outcomes"][run_name]), 2)
        self.assertEqual(len(response_rows), 2)
        self.assertEqual(response_rows[0]["schema_version"], "bilgeval.response/v1")

    def test_a_bos_decided_by_the_model_name_is_announced(self):
        class FixtureLogprobClient(LogprobModelClient):
            model = "fixture-model"
            transport_name = "fixture"

            def __init__(self, probe):
                self.bos_diagnostics = {"source": "fixture", "sequence_start_probe": probe}

            def score_prompt_completion_pairs(self, pairs, **kwargs):
                return [CompletionScoreResult(score=-1.0, num_completion_tokens=1) for _ in pairs]

        load_registry_file(str(REGISTRY_PATH))
        for probe, warned in ((None, True), ({"source": "vllm_tokenize_chat_template"}, False)):
            stderr = io.StringIO()
            with self.subTest(warned=warned), tempfile.TemporaryDirectory() as directory, patch(
                "bilgeval.main.create_logprob_client",
                return_value=FixtureLogprobClient(probe),
            ), contextlib.redirect_stderr(stderr):
                run_benchmark(
                    dataset_names=["fixture_mc_logprob"],
                    output_dir=directory,
                    model="fixture-model",
                    skip_preflight=True,
                    silent=True,
                )
            self.assertEqual("by the model-name rule" in stderr.getvalue(), warned)

    def test_a_run_that_lost_its_requests_fails_the_request_gate_and_keeps_its_report(self):
        class UnreachableLogprobClient(LogprobModelClient):
            model = "fixture-model"
            transport_name = "fixture"
            bos_diagnostics = {"source": "fixture"}

            def score_prompt_completion_pairs(self, pairs, **kwargs):
                raise RuntimeError("Error code: 400 - prompt list length exceeds the maximum")

        load_registry_file(str(REGISTRY_PATH))
        with tempfile.TemporaryDirectory() as directory, patch(
            "bilgeval.main.create_logprob_client",
            return_value=UnreachableLogprobClient(),
        ):
            with self.assertRaisesRegex(ValueError, "Request gate failed .* fixture_mc_logprob logprob lost"):
                run_benchmark(
                    dataset_names=["fixture_mc_logprob"],
                    output_dir=directory,
                    model="fixture-model",
                    skip_preflight=True,
                )
            (report_file,) = Path(directory).glob("*/final_report.json")
            saved = json.loads(report_file.read_text(encoding="utf-8"))

        dataset = next(iter(saved["datasets"].values()))
        self.assertGreater(dataset["requests"]["logprob"]["sent"], 0)
        self.assertEqual(dataset["requests"]["logprob"]["failed"], dataset["requests"]["logprob"]["sent"])
        self.assertFalse(saved["request_gate"]["passed"])
        self.assertEqual(saved["max_request_failure_rate"], 0.01)

    def test_the_request_gate_passes_a_healthy_run_and_can_be_turned_off(self):
        class FixtureLogprobClient(LogprobModelClient):
            model = "fixture-model"
            transport_name = "fixture"
            bos_diagnostics = {"source": "fixture"}

            def __init__(self, fail):
                self.fail = fail

            def score_prompt_completion_pairs(self, pairs, **kwargs):
                if self.fail:
                    raise RuntimeError("Connection error.")
                return [CompletionScoreResult(score=-1.0, num_completion_tokens=1) for _ in pairs]

        load_registry_file(str(REGISTRY_PATH))
        for fail, rate in ((False, 0.01), (True, 1.0)):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory, patch(
                "bilgeval.main.create_logprob_client",
                return_value=FixtureLogprobClient(fail),
            ):
                report = run_benchmark(
                    dataset_names=["fixture_mc_logprob"],
                    output_dir=directory,
                    model="fixture-model",
                    skip_preflight=True,
                    max_request_failure_rate=rate,
                )
                self.assertTrue(report["request_gate"]["passed"])
                counts = next(iter(report["datasets"].values()))["requests"]["logprob"]
                self.assertEqual(counts["failed"], counts["sent"] if fail else 0)

    def test_run_benchmark_wires_multi_sample_settings_into_generation(self):
        class FixtureLogprobClient:
            transport_name = "fixture"
            effective_add_bos_token = False
            bos_diagnostics = {"source": "fixture"}

        class FixtureGenerationClient:
            """One value per request, in order: one example, so one value per sample."""

            def __init__(self):
                self.values = iter(["7", "8", "8"])
                self.requests = []

            def generate_texts(self, prompts, **kwargs):
                results = []
                for _ in prompts:
                    self.requests.append(kwargs)
                    results.append(GeneratedTextResult(text=next(self.values), request_id=f"req-{len(self.requests)}"))
                return results

        generation_client = FixtureGenerationClient()
        load_registry_file(str(REGISTRY_PATH))
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("bilgeval.main.create_logprob_client", return_value=FixtureLogprobClient()),
            patch(
                "bilgeval.main.OpenAICompatibleTextGenerationClient",
                return_value=generation_client,
            ),
        ):
            report = run_benchmark(
                dataset_names=["fixture_math_regex"],
                output_dir=directory,
                model="fixture-model",
                generation_temperature=0.7,
                generation_num_samples=3,
                generation_pass_at_k=[1, 2],
                generation_aggregation="consensus",
                skip_preflight=True,
                output_level="summary",
            )
            manifest = json.loads(Path(report["manifest_file"]).read_text(encoding="utf-8"))
            self.assertEqual(report["artifacts"]["responses"], {})
            self.assertEqual(list(Path(report["report_dir"]).glob("*_responses.jsonl.gz")), [])

        _, dataset = next(iter(report["datasets"].items()))
        self.assertEqual(len(generation_client.requests), 3)
        self.assertTrue(all("seed" not in kwargs for kwargs in generation_client.requests))
        self.assertEqual(dataset["accuracy"], 1.0)
        self.assertAlmostEqual(dataset["multi_sample_metrics"]["pass_at_k"]["1"], 2 / 3)
        self.assertTrue(manifest["runtime"]["model_runtime_fingerprint"].startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
