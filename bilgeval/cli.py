"""The command line: `bilgeval <targets>` and the validate / inspect / convert
subcommands. Everything here parses arguments and prints; the work is
`main.run_benchmark` and the registry helpers it builds on."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from . import main as runtime
from .artifacts import OUTPUT_LEVELS
from .batching import BatchRequestError
from .conversion import convert_legacy_dataset
from .evaluators import DEFAULT_REQUEST_TIMEOUT_SECONDS
from .main import (
    ACTIVE_REGISTRY_CONTEXT,
    apply_max_examples_override,
    load_registry_file,
    prepare_run,
    resolve_benchmark_runs,
    resolve_run_config,
    run_benchmark,
)
from .manifest import sha256_file
from .preflight import PreflightError
from .reader import FunctionCallExample, MultiTurnJudgeExample, validate_dataset_records
from .registry import DATASETS
from .report import DEFAULT_MAX_REQUEST_FAILURE_RATE, MULTI_MODEL_REPORT_SCHEMA_VERSION
from .scorer_table import scorer
from .scorers import generation_request_settings, inspect_prompt, multi_sample_settings
from .statistics import build_multi_model_comparisons


DEFAULT_MODEL = "google/gemma-4-31B-it"

UNIVERSAL_DATA_SCHEME_TEMPLATE = {
    "schema_version": "bilgeval.instance/v1",
    "id": "example_001",
    "task_type": "multiple_choice",
    "split": "validation",
    "language": "tr",
    "input": {
        "question": "Soru metni burada.",
        "choices": [
            {"label": "A", "text": "Birinci seçenek"},
            {"label": "B", "text": "İkinci seçenek"},
        ],
    },
    "target": {"answer_label": "B", "answer_text": "İkinci seçenek"},
    "metadata": {"optional": "extra fields"},
    "provenance": {
        "source": "my_dataset",
        "source_revision": "v1",
        "source_record_id": "source_001",
    },
}


BRING_YOUR_OWN_RESPONSES_TEMPLATE = {
    "dataset_name": "tr_movies",
    "id": "example_001",
    "candidate_response": "B",
    "generation_prompt_text": "Optional: the prompt used by your external inference system.",
    "metadata": {
        "model": "external-model-name",
        "optional": "extra fields are preserved in raw parsing but not required",
    },
}


def print_template(template_name: str) -> None:
    templates = {
        "universal_data_scheme": UNIVERSAL_DATA_SCHEME_TEMPLATE,
        "bring_your_own_responses": BRING_YOUR_OWN_RESPONSES_TEMPLATE,
    }
    print(json.dumps(templates[template_name], ensure_ascii=False, indent=2))


def parse_model_args(values: list[str] | None) -> list[str]:
    raw_values = values if values else [os.getenv("BILGEVAL_MODEL", DEFAULT_MODEL)]
    models: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        for item in str(value).split(","):
            model = item.strip()
            if model and model not in seen:
                models.append(model)
                seen.add(model)
    if not models:
        raise ValueError("At least one model is required.")
    return models


def datasets_for_targets(targets: list[str]) -> list:
    if not targets:
        return list(DATASETS.values())
    names: list[str] = []
    for run in resolve_benchmark_runs(targets):
        names.append(run.dataset_name)
        if run.task_spec.few_shot.dataset:
            names.append(run.task_spec.few_shot.dataset)
    return [DATASETS[name] for name in dict.fromkeys(names)]


def run_validate_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="bilgeval validate",
        description="Validate registry references and dataset records without calling a model.",
    )
    parser.add_argument("targets", nargs="*", help="Optional task or suite names to validate.")
    parser.add_argument("--registry-file", help="Registry JSON to validate and load.")
    parser.add_argument(
        "--max-records",
        type=int,
        help="Validate only the first N records of each selected dataset.",
    )
    args = parser.parse_args(argv)
    try:
        load_registry_file(args.registry_file)
        results = []
        for config in datasets_for_targets(args.targets):
            if config.task_type == "typed_decision":
                from .td_reader import read_td_dataset
                examples = read_td_dataset(config)
                result = {"dataset": config.name, "status": "valid", "task_type": config.task_type, "records_validated": len(examples) if args.max_records is None else min(len(examples), args.max_records)}
            else:
                result = validate_dataset_records(config, max_records=args.max_records)
            if (
                result["status"] == "valid"
                and args.max_records is None
                and config.content_hash
                and isinstance(config.path, Path)
                and config.path.is_file()
            ):
                actual_hash = sha256_file(config.path)
                if actual_hash != config.content_hash:
                    raise ValueError(
                        f"Dataset {config.name} content hash mismatch: "
                        f"expected {config.content_hash}, got {actual_hash}."
                    )
                result["content_hash_verified"] = True
            else:
                result["content_hash_verified"] = False
            if (
                result["status"] == "valid"
                and args.max_records is None
                and config.row_count is not None
                and result["records_validated"] != config.row_count
            ):
                raise ValueError(
                    f"Dataset {config.name} row count mismatch: expected {config.row_count}, "
                    f"got {result['records_validated']}."
                )
            result["row_count_verified"] = (
                result["status"] == "valid"
                and args.max_records is None
                and config.row_count is not None
            )
            results.append(result)
        run_results = []
        if args.targets and args.max_records is None:
            for run in resolve_benchmark_runs(args.targets):
                prepared = prepare_run(run, resolve_run_config(run))
                run_results.append(
                    {
                        "run_name": run.run_name,
                        "dataset": run.dataset_name,
                        "selected_examples": len(prepared.examples),
                        "requested_few_shot_examples": prepared.config.few_shot_count,
                        "few_shot_match_field": prepared.config.few_shot_match_field,
                        "loaded_few_shot_candidates": len(prepared.few_shot_examples),
                        "few_shot_leakage_checked": True,
                    }
                )
    except ValueError as exc:
        parser.exit(status=1, message=f"bilgeval: error: {exc}\n")
    print(
        json.dumps(
            {
                "valid": True,
                "registry": dict(ACTIVE_REGISTRY_CONTEXT),
                "datasets": results,
                "runs": run_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def inspect_example(example) -> dict:
    if isinstance(example, FunctionCallExample):
        return {
            "id": example.id,
            "messages": example.messages,
            "tools": example.tools,
            "target": {
                "calls": example.expected_calls,
                "should_call": example.should_call,
                "call_order": example.call_order,
            },
            "category": example.category,
        }
    if isinstance(example, MultiTurnJudgeExample):
        return {
            "id": example.id,
            "task_id": example.task_id,
            "task_name": example.task_name,
            "topic": example.topic,
            "turns": example.turns,
            "eval_turn_positions": example.eval_turn_positions,
            "judge_criteria": example.evaluation_prompt,
        }
    choices = getattr(example, "choices", None)
    value = {
        "id": str(getattr(example, "id", "")),
        "question": str(getattr(example, "question", getattr(example, "prompt", ""))),
    }
    if hasattr(example, "expected"):
        value["target"] = {
            "expected": example.expected,
            "subcategory": example.subcategory,
        }
    elif hasattr(example, "answer_format"):
        value["document"] = example.document
        value["target"] = {
            "answer": example.answer,
            "answer_format": example.answer_format,
            "task_type": example.task_type,
            "evidence_spans": example.evidence_spans,
        }
    elif hasattr(example, "entry_point"):
        value["target"] = {
            "entry_point": example.entry_point,
            "has_unit_tests": bool(example.test),
        }
    elif choices is not None:
        value["choices"] = [
            {"label": choice.label, "text": choice.text}
            for choice in choices
        ]
        value["target"] = {
            "answer_label": example.answer_label,
            "answer_text": example.answer_text,
        }
    else:
        value["target"] = {
            "references": [str(getattr(example, "answer_text", getattr(example, "answer", "")))]
        }
    return value


def run_inspect_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="bilgeval inspect",
        description="Render selected examples and prompts without calling a model.",
    )
    parser.add_argument("targets", nargs="+", help="Task or suite names to inspect.")
    parser.add_argument("--registry-file", help="Registry JSON to load.")
    parser.add_argument("--limit", type=int, default=3, help="Examples to show per run.")
    parser.add_argument("--offset", type=int, default=0, help="First selected example to show.")
    parser.add_argument("--max-examples", type=int, help="Override each run's sample limit.")
    args = parser.parse_args(argv)
    if args.limit < 1 or args.offset < 0:
        parser.error("--limit must be at least 1 and --offset must not be negative.")
    try:
        load_registry_file(args.registry_file)
        runs = apply_max_examples_override(resolve_benchmark_runs(args.targets), args.max_examples)
        prepared_runs = [prepare_run(run, resolve_run_config(run)) for run in runs]
        output = []
        for prepared in prepared_runs:
            examples = []
            for index, example in enumerate(
                prepared.examples[args.offset : args.offset + args.limit],
                start=args.offset,
            ):
                examples.append(
                    {
                        "instance": inspect_example(example),
                        "rendered_prompt": inspect_prompt(prepared, example, index),
                    }
                )
            output.append(
                {
                    "run_name": prepared.run.run_name,
                    "task_name": prepared.run.task_name,
                    "task_version": prepared.run.task_spec.version,
                    "dataset_name": prepared.config.dataset_name,
                    "dataset_version": prepared.config.dataset_version,
                    "schema_version": prepared.config.schema_version,
                    "num_selected_examples": len(prepared.examples),
                    "generation": {
                        "max_tokens": prepared.config.generation_max_tokens,
                        **generation_request_settings(prepared.config),
                        **multi_sample_settings(prepared.config),
                    },
                    "few_shot_candidate_ids": [
                        example.id for example in prepared.few_shot_examples
                    ],
                    "few_shot_ids_per_example": [
                        [example.id for example in selected]
                        for selected in prepared.few_shot_examples_per_example[
                            args.offset : args.offset + args.limit
                        ]
                    ],
                    "examples": examples,
                }
            )
    except ValueError as exc:
        parser.exit(status=1, message=f"bilgeval: error: {exc}\n")
    print(json.dumps({"targets": args.targets, "runs": output}, ensure_ascii=False, indent=2))


def run_convert_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="bilgeval convert",
        description="Convert a registered legacy JSONL dataset to bilgeval.instance/v1 JSONL.",
    )
    parser.add_argument("dataset", help="Registered legacy dataset name.")
    parser.add_argument("--output", required=True, help="Output canonical JSONL path.")
    parser.add_argument("--split", required=True, help="Source split recorded on every instance.")
    parser.add_argument("--language", required=True, help="BCP-47-style language code, such as tr or en.")
    parser.add_argument(
        "--registry-file",
        help="Optional bilgeval.registry/v1 file containing the dataset to convert.",
    )
    args = parser.parse_args(argv)
    try:
        load_registry_file(args.registry_file)
        config = DATASETS.get(args.dataset)
        if config is None:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        result = convert_legacy_dataset(
            config=config,
            output_path=Path(args.output).expanduser().resolve(),
            split=args.split,
            language=args.language,
        )
    except ValueError as exc:
        parser.exit(status=1, message=f"bilgeval: error: {exc}\n")
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bilgeval",
        description="Lightweight LLM benchmarking with logprob-based multiple-choice evaluation.",
        epilog=(
            "Data tooling commands: 'bilgeval validate', 'bilgeval inspect', and "
            "'bilgeval convert'."
        ),
    )
    parser.add_argument("--td-config", help="TD tokenizer paths and runtime identity JSON (offline)")
    parser.add_argument(
        "targets",
        nargs="*",
        help="Tasks or suites to evaluate. Use registry target names or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Parent directory where a dated run folder will be created.",
    )
    parser.add_argument(
        "--registry-file",
        help="A bilgeval.registry/v1 bundle.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("BILGEVAL_BASE_URL", "http://127.0.0.1:8181/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help=(
            "Model name used for completions scoring. Repeat this option or pass a comma-separated "
            "list to run the same targets against multiple models. Defaults to BILGEVAL_MODEL."
        ),
    )
    parser.add_argument(
        "--models",
        dest="models",
        action="append",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("BILGEVAL_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY",
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("BILGEVAL_JUDGE_MODEL"),
        help="Model used by MC, safety, and document-understanding judge scorers.",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.getenv("BILGEVAL_JUDGE_BASE_URL"),
        help="OpenAI-compatible judge endpoint. Defaults to --base-url.",
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.getenv("BILGEVAL_JUDGE_API_KEY"),
        help="Judge endpoint API key. Defaults to --api-key.",
    )
    parser.add_argument(
        "--judge-insecure-skip-verify",
        action="store_true",
        default=(
            True
            if os.getenv("BILGEVAL_JUDGE_INSECURE_SKIP_VERIFY", "").lower()
            in {"1", "true", "yes"}
            else None
        ),
        help="Disable TLS verification for the judge endpoint; otherwise inherit the main policy.",
    )
    parser.add_argument(
        "--judge-generation-api",
        choices=("completions", "chat"),
        default=os.getenv("BILGEVAL_JUDGE_GENERATION_API", "chat"),
        help="OpenAI generation API used by the judge endpoint.",
    )
    parser.add_argument(
        "--judge-disable-thinking",
        action="store_true",
        default=os.getenv("BILGEVAL_JUDGE_DISABLE_THINKING", "").lower()
        in {"1", "true", "yes"},
        help=(
            "Pass chat_template_kwargs.thinking=false and enable_thinking=false "
            "to the judge endpoint."
        ),
    )
    parser.add_argument(
        "--judge-mtbench-guided-rating",
        action="store_true",
        default=os.getenv("BILGEVAL_JUDGE_MTBENCH_GUIDED_RATING", "").lower()
        in {"1", "true", "yes"},
        help=(
            "Constrain MT-Bench judge output to a parser-compatible [[score]] "
            "rating. Only valid when every selected task uses mtbench_judge."
        ),
    )
    parser.add_argument(
        "--verifier-model",
        default=os.getenv("BILGEVAL_VERIFIER_MODEL"),
        help=(
            "Model used only for strict free-response answer equivalence checks. "
            "Required by tasks with answer_verification=llm_as_regex."
        ),
    )
    parser.add_argument(
        "--verifier-base-url",
        default=os.getenv("BILGEVAL_VERIFIER_BASE_URL"),
        help="OpenAI-compatible verifier endpoint. Defaults to --base-url.",
    )
    parser.add_argument(
        "--verifier-api-key",
        default=os.getenv("BILGEVAL_VERIFIER_API_KEY"),
        help="Verifier endpoint API key. Defaults to --api-key.",
    )
    parser.add_argument(
        "--verifier-insecure-skip-verify",
        action="store_true",
        default=(
            True
            if os.getenv("BILGEVAL_VERIFIER_INSECURE_SKIP_VERIFY", "").lower()
            in {"1", "true", "yes"}
            else None
        ),
        help="Disable TLS verification for the verifier endpoint; otherwise inherit the main endpoint policy.",
    )
    parser.add_argument(
        "--verifier-disable-thinking",
        action="store_true",
        default=os.getenv("BILGEVAL_VERIFIER_DISABLE_THINKING", "").lower()
        in {"1", "true", "yes"},
        help="Pass chat_template_kwargs.thinking=false to the answer-verifier endpoint.",
    )
    parser.add_argument(
        "--verifier-json-schema",
        action="store_true",
        default=os.getenv("BILGEVAL_VERIFIER_JSON_SCHEMA", "").lower()
        in {"1", "true", "yes"},
        help=(
            "Constrain answer-verifier output to the strict {match: boolean} JSON schema; "
            "requires endpoint support for response_format=json_schema."
        ),
    )
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=int(os.getenv("BILGEVAL_REQUEST_BATCH_SIZE", "64")),
        help="Number of examples placed in each client-side request chunk.",
    )
    parser.add_argument(
        "--completion-concurrency",
        type=int,
        default=int(os.getenv("BILGEVAL_COMPLETION_CONCURRENCY", "1")),
        help=(
            "Maximum simultaneous /completions HTTP requests for candidate and judge. "
            "Default: 1. Does not affect chat, tools, verifier, or logprob requests."
        ),
    )
    parser.add_argument(
        "--completion-batch-size",
        type=int,
        default=(
            int(os.environ["BILGEVAL_COMPLETION_BATCH_SIZE"])
            if os.getenv("BILGEVAL_COMPLETION_BATCH_SIZE") else None
        ),
        help=(
            "Maximum prompts per /completions HTTP request within each "
            "--request-batch-size chunk. If omitted, use ceil(chunk size / "
            "completion concurrency); defaults keep one request per chunk."
        ),
    )
    parser.add_argument(
        "--generation-client",
        choices=("sync", "async"),
        default=os.getenv("BILGEVAL_GENERATION_CLIENT", "sync"),
        help=(
            "Text transport for candidate, judge and verifier: sync (default), or "
            "async for high concurrency (requires bilgeval[async]). "
            "Does not change tool-call or logprob transport."
        ),
    )
    parser.add_argument(
        "--generation-concurrency",
        type=int,
        default=int(os.getenv("BILGEVAL_GENERATION_CONCURRENCY", "16")),
        help="Maximum simultaneous candidate chat-completion requests.",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=int(os.getenv("BILGEVAL_JUDGE_CONCURRENCY", "16")),
        help="Maximum simultaneous judge chat-completion requests.",
    )
    parser.add_argument(
        "--verifier-concurrency",
        type=int,
        default=int(os.getenv("BILGEVAL_VERIFIER_CONCURRENCY", "16")),
        help="Maximum simultaneous answer-verifier chat-completion requests.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Temporarily limit each selected benchmark run to the first N examples.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print dataset evaluation progress to stderr.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=int(os.getenv("BILGEVAL_PROGRESS_EVERY", "10")),
        help="When --progress is enabled, print an update every N requests.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Do not print benchmark output or progress to the terminal.",
    )
    parser.add_argument(
        "--output-level",
        choices=OUTPUT_LEVELS,
        default=os.getenv("BILGEVAL_OUTPUT_LEVEL", "responses"),
        help=(
            "summary writes aggregate scores only; responses also writes compressed per-example "
            "model results; debug additionally captures bounded, deduplicated API traces."
        ),
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=int(os.getenv("BILGEVAL_BOOTSTRAP_RESAMPLES", "10000")),
        help="Number of deterministic bootstrap replicates used for accuracy intervals and paired deltas.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=float(os.getenv("BILGEVAL_CONFIDENCE_LEVEL", "0.95")),
        help="Confidence level for bootstrap intervals, strictly between 0 and 1.",
    )
    parser.add_argument(
        "--statistics-seed",
        default=os.getenv("BILGEVAL_STATISTICS_SEED", "bilgeval-statistics-v1"),
        help="Stable seed string used to reproduce bootstrap intervals.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the small startup health checks for the evaluated model and judge model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve tasks, validate data, and write a run manifest without calling the model endpoint.",
    )
    parser.add_argument(
        "--insecure-skip-verify",
        action="store_true",
        default=os.getenv("BILGEVAL_INSECURE_SKIP_VERIFY", "").lower() in {"1", "true", "yes"},
        help="Disable TLS certificate verification for OpenAI-compatible endpoints with internal certificates.",
    )
    parser.add_argument(
        "--add-bos-token",
        action="store_true",
        default=os.getenv("BILGEVAL_ADD_BOS_TOKEN", "").lower() in {"1", "true", "yes"},
        help="Deprecated alias for --bos-mode always.",
    )
    parser.add_argument(
        "--bos-mode",
        choices=("auto", "always", "never"),
        default=os.getenv("BILGEVAL_BOS_MODE"),
        help=(
            "BOS policy for completion scoring. Auto verifies Gemma endpoints and avoids duplicate BOS tokens; "
            "always enforces one BOS for every model; never disables insertion."
        ),
    )
    parser.add_argument(
        "--bos-token",
        default=os.getenv("BILGEVAL_BOS_TOKEN"),
        help="BOS token string used by BOS verification and insertion.",
    )
    parser.add_argument(
        "--logprob-api",
        choices=("auto", "openai", "sglang"),
        default=os.getenv("BILGEVAL_LOGPROB_API"),
        help=(
            "Logprob transport. Auto uses native SGLang /generate when detected and otherwise "
            "uses OpenAI /v1/completions."
        ),
    )
    parser.add_argument(
        "--generation-api",
        choices=("completions", "chat"),
        default=os.getenv("BILGEVAL_GENERATION_API"),
        help=(
            "OpenAI generation API to use. Chat mode applies the server's chat template; "
            "log-probability scoring still uses /v1/completions."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        dest="generation_max_tokens",
        type=int,
        default=int(os.environ["BILGEVAL_MAX_TOKENS"]) if os.getenv("BILGEVAL_MAX_TOKENS") else None,
        help="Override the candidate max_tokens of every selected generation task.",
    )
    parser.add_argument(
        "--generation-extra-body",
        default=os.getenv("BILGEVAL_GENERATION_EXTRA_BODY"),
        help=(
            "JSON object merged into every candidate request body (vLLM/SGLang "
            'extras such as {"min_p": 0, "presence_penalty": 1.5, "reasoning_effort": "medium"}).'
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.getenv("BILGEVAL_REQUEST_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT_SECONDS))),
        help=(
            "HTTP read timeout in seconds for candidate, judge and verifier. "
            "With --generation-client async, also bounds each complete attempt "
            "until response headers arrive."
        ),
    )
    parser.add_argument(
        "--stream",
        dest="stream_generation",
        action="store_true",
        default=os.getenv("BILGEVAL_STREAM", "").lower() in {"1", "true", "yes"},
        help=(
            "Stream candidate chat completions. Needed behind a gateway that closes a "
            "connection without traffic (some answer 504 after 300 seconds); the answer is "
            "the same."
        ),
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=os.getenv("BILGEVAL_DISABLE_THINKING", "").lower() in {"1", "true", "yes"},
        help=(
            "Pass chat_template_kwargs.thinking=false / enable_thinking=false to the "
            "candidate endpoint (Qwen3-style thinking models)."
        ),
    )
    parser.add_argument(
        "--temperature",
        "--generation-temperature",
        dest="generation_temperature",
        type=float,
        default=(
            float(os.environ["BILGEVAL_TEMPERATURE"])
            if os.getenv("BILGEVAL_TEMPERATURE")
            else None
        ),
        help="Override generation temperature for every selected task.",
    )
    parser.add_argument(
        "--top-p",
        "--generation-top-p",
        dest="generation_top_p",
        type=float,
        default=float(os.environ["BILGEVAL_TOP_P"]) if os.getenv("BILGEVAL_TOP_P") else None,
        help="Override nucleus-sampling top_p for every selected task.",
    )
    parser.add_argument(
        "--top-k",
        "--generation-top-k",
        dest="generation_top_k",
        type=int,
        default=int(os.environ["BILGEVAL_TOP_K"]) if os.getenv("BILGEVAL_TOP_K") else None,
        help="Override provider-specific top_k for every selected task.",
    )
    parser.add_argument(
        "--stop",
        dest="generation_stop",
        action="append",
        help="Override stop sequences. Repeat the option to provide multiple sequences.",
    )
    parser.add_argument(
        "--num-samples",
        dest="generation_num_samples",
        type=int,
        default=(
            int(os.environ["BILGEVAL_NUM_SAMPLES"])
            if os.getenv("BILGEVAL_NUM_SAMPLES")
            else None
        ),
        help="Override generated samples per example (1-64).",
    )
    parser.add_argument(
        "--pass-at-k",
        dest="generation_pass_at_k",
        action="append",
        type=int,
        help="Report an unbiased pass@k estimate. Repeat for multiple k values.",
    )
    parser.add_argument(
        "--generation-aggregation",
        choices=("first", "consensus", "mean"),
        help=(
            "Select the top-level result from the first sample or majority consensus, "
            "or report the mean over all samples as the headline."
        ),
    )
    parser.add_argument(
        "--code-executor",
        choices=("auto", "bubblewrap", "unix"),
        default=os.getenv("BILGEVAL_CODE_EXECUTOR", "auto"),
        help=(
            "HumanEval execution backend. Auto uses a configured Unix socket, otherwise "
            "it requires Bubblewrap."
        ),
    )
    parser.add_argument(
        "--code-executor-socket",
        default=os.getenv("BILGEVAL_CODE_EXECUTOR_SOCKET"),
        help="Unix socket for the dedicated networkless HumanEval executor service.",
    )
    parser.add_argument(
        "--code-execution-timeout",
        type=float,
        default=float(os.getenv("BILGEVAL_CODE_EXECUTION_TIMEOUT", "3")),
        help="Per-candidate HumanEval unit-test timeout in seconds (0.1-30).",
    )
    parser.add_argument(
        "--model-profile",
        help=(
            "Select a model profile from bilgeval.registry/v1. Without this option, exactly one "
            "matching profile is applied automatically."
        ),
    )
    parser.add_argument(
        "--external-responses-file",
        help=(
            "JSONL file with externally generated candidate responses to judge. "
            "Only valid with mc_judge tasks."
        ),
    )
    parser.add_argument(
        "--replay-responses-file",
        help=(
            "Replay candidate generations from a bilgeval response artifact and rerun scoring. "
            "Empty/failed saved candidates fall back to the live candidate endpoint. Exactly one "
            "task run must be selected."
        ),
    )
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        help="Write the report but exit non-zero when any selected task coverage is below this value.",
    )
    parser.add_argument(
        "--max-request-failure-rate",
        type=float,
        default=DEFAULT_MAX_REQUEST_FAILURE_RATE,
        help=(
            "Write the report but exit non-zero when a task lost more than this share of its "
            "endpoint requests (candidate, verifier, judge or log-probability) to transport or "
            "HTTP errors after retries; a lost item scores as wrong. 1 turns the check off. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--show-template",
        choices=("universal_data_scheme", "bring_your_own_responses"),
        help="Print a JSON template and exit.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List core registered datasets and exit.",
    )
    parser.add_argument(
        "--list-benchmark-sets",
        "--list-benchmark_sets",
        action="store_true",
        help="List runnable task and suite targets and exit.",
    )
    parser.add_argument(
        "--list-meta-configs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"validate", "inspect", "convert"}:
        command = sys.argv[1]
        command_argv = sys.argv[2:]
        if command == "validate":
            run_validate_command(command_argv)
        elif command == "inspect":
            run_inspect_command(command_argv)
        else:
            run_convert_command(command_argv)
        return

    parser = build_parser()
    args = parser.parse_args()

    try:
        load_registry_file(args.registry_file)
    except ValueError as exc:
        parser.exit(status=1, message=f"bilgeval: error: {exc}\n")

    if args.show_template:
        if not args.silent:
            print_template(args.show_template)
        return

    if args.list_datasets:
        if not args.silent:
            for dataset_name in DATASETS:
                print(dataset_name)
        return

    if args.list_benchmark_sets or args.list_meta_configs:
        if not args.silent:
            bundle = runtime.ACTIVE_REGISTRY_BUNDLE
            if bundle is None:
                parser.error("No TaskSpec registry is active.")
            for task_name, task in bundle.tasks.items():
                print(
                    f"{task_name}: task dataset={task.dataset} "
                    f"scorer={task.scorer.id}"
                )
            for suite_name, suite in bundle.suites.items():
                task_list = ", ".join(item.task for item in suite.tasks)
                print(f"{suite_name}: suite [{task_list}]")
        return

    if not args.targets:
        parser.error(
            "Provide at least one task or suite, or use --list-datasets/--list-benchmark-sets."
        )

    selected_runs = resolve_benchmark_runs(args.targets)
    requires_judge_model = any(
        "judge" in scorer(resolve_run_config(run).scorer_id).requires
        for run in selected_runs
    )
    if requires_judge_model and not args.judge_model and not args.dry_run:
        parser.error(
            "At least one selected benchmark uses eval_mode='judge', so --judge-model is required."
        )
    requires_verifier_model = any(
        resolve_run_config(run).answer_verification != "regex"
        for run in selected_runs
    )
    if requires_verifier_model and not args.verifier_model and not args.dry_run:
        parser.error(
            "At least one selected benchmark uses LLM answer verification, so "
            "--verifier-model is required."
        )

    try:
        models = parse_model_args(args.models)
        parsed_extra_body = None
        if args.generation_extra_body:
            try:
                parsed_extra_body = json.loads(args.generation_extra_body)
            except json.JSONDecodeError as exc:
                parser.error(f"--generation-extra-body is not valid JSON: {exc}")
            if not isinstance(parsed_extra_body, dict):
                parser.error("--generation-extra-body must be a JSON object.")
        if args.external_responses_file and len(models) > 1:
            parser.error("--external-responses-file can only be used with one model at a time.")
        if args.replay_responses_file and len(models) > 1:
            parser.error("--replay-responses-file can only be used with one model at a time.")
        reports = []
        for model in models:
            if args.progress and not args.silent and len(models) > 1:
                print(f"[bilgeval] model start {model}", file=sys.stderr, flush=True)
            reports.append(
                run_benchmark(
                    dataset_names=args.targets,
                    output_dir=args.output_dir,
                    base_url=args.base_url,
                    model=model,
                    api_key=args.api_key,
                    judge_model=args.judge_model,
                    judge_base_url=args.judge_base_url,
                    judge_api_key=args.judge_api_key,
                    judge_insecure_skip_verify=args.judge_insecure_skip_verify,
                    judge_generation_api=args.judge_generation_api,
                    judge_disable_thinking=args.judge_disable_thinking,
                    judge_mtbench_guided_rating=args.judge_mtbench_guided_rating,
                    verifier_model=args.verifier_model,
                    verifier_base_url=args.verifier_base_url,
                    verifier_api_key=args.verifier_api_key,
                    verifier_insecure_skip_verify=args.verifier_insecure_skip_verify,
                    verifier_default_headers={
                        key: value
                        for key, value in {
                            "Authorization": os.getenv(
                                "BILGEVAL_VERIFIER_AUTHORIZATION_HEADER"
                            ),
                            "x-api-key": os.getenv("BILGEVAL_VERIFIER_X_API_KEY"),
                        }.items()
                        if value
                    },
                    verifier_disable_thinking=args.verifier_disable_thinking,
                    verifier_json_schema=args.verifier_json_schema,
                    request_batch_size=args.request_batch_size,
                    completion_concurrency=args.completion_concurrency,
                    completion_batch_size=args.completion_batch_size,
                    generation_client=args.generation_client,
                    generation_concurrency=args.generation_concurrency,
                    judge_concurrency=args.judge_concurrency,
                    verifier_concurrency=args.verifier_concurrency,
                    show_progress=args.progress,
                    progress_every=args.progress_every,
                    silent=args.silent,
                    skip_preflight=args.skip_preflight,
                    external_responses_file=args.external_responses_file,
                    replay_responses_file=args.replay_responses_file,
                    minimum_coverage=args.minimum_coverage,
                    max_request_failure_rate=args.max_request_failure_rate,
                    max_examples=args.max_examples,
                    insecure_skip_verify=args.insecure_skip_verify,
                    output_level=args.output_level,
                    add_bos_token=args.add_bos_token,
                    bos_mode=args.bos_mode,
                    bos_token=args.bos_token,
                    logprob_api=args.logprob_api,
                    generation_api=args.generation_api,
                    model_profile=args.model_profile,
                    generation_max_tokens=args.generation_max_tokens,
                    candidate_disable_thinking=args.disable_thinking,
                    generation_extra_body=parsed_extra_body,
                    request_timeout=args.request_timeout,
                    stream_generation=args.stream_generation,
                    generation_temperature=args.generation_temperature,
                    generation_top_p=args.generation_top_p,
                    generation_top_k=args.generation_top_k,
                    generation_stop=args.generation_stop,
                    generation_num_samples=args.generation_num_samples,
                    generation_pass_at_k=args.generation_pass_at_k,
                    generation_aggregation=args.generation_aggregation,
                    code_executor_mode=args.code_executor,
                    code_executor_socket=args.code_executor_socket,
                    code_execution_timeout_seconds=args.code_execution_timeout,
                    bootstrap_resamples=args.bootstrap_resamples,
                    confidence_level=args.confidence_level,
                    statistics_seed=args.statistics_seed,
                    td_config=json.loads(Path(args.td_config).read_text()) if args.td_config else None,
                    collect_sample_outcomes=len(models) > 1 and not args.dry_run,
                    dry_run=args.dry_run,
                )
            )
            if args.progress and not args.silent and len(models) > 1:
                print(f"[bilgeval] model done {model}", file=sys.stderr, flush=True)
        paired_comparisons = None
        if len(reports) > 1 and not args.dry_run:
            paired_comparisons = build_multi_model_comparisons(
                reports,
                resamples=args.bootstrap_resamples,
                confidence_level=args.confidence_level,
                seed=args.statistics_seed,
            )
        for item in reports:
            item.pop("_sample_outcomes", None)
        report = reports[0] if len(reports) == 1 else {
            "schema_version": MULTI_MODEL_REPORT_SCHEMA_VERSION,
            "targets": args.targets,
            "models": models,
            "reports": reports,
            "macro_accuracy_by_family_by_model": {
                item["model"]: item.get("macro_accuracy_by_family", {})
                for item in reports
            },
            "paired_comparisons": paired_comparisons,
        }
        if len(reports) > 1:
            from .typed_decision import common_reports
            report["td_common_tables"] = common_reports(reports)
            summary_path = Path(args.output_dir) / f"{datetime.now().strftime('%Y-%m-%d')}_multi_model_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as file:
                json.dump(report, file, ensure_ascii=False, indent=2)
            report["summary_file"] = str(summary_path.resolve())
    except (PreflightError, BatchRequestError) as exc:
        parser.exit(status=1, message=f"bilgeval: error: {exc}\n")
    except (ValueError, ImportError) as exc:
        parser.exit(status=1, message=f"bilgeval: error: {exc}\n")

    if not args.silent:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
