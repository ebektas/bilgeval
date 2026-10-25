"""Runtime bindings of the scorer table.

`scorer_table.py` says which scorers and dataset types exist; this module says
what runs for each: the reader that opens a dataset type, the evaluate
function that scores a scorer (with the arguments it takes from the run), and
the prompt `bilgeval inspect` renders. The evaluate functions themselves are
benchmark-specific and live in `evaluators.py`; everything here is wiring.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import instruction_following, livecodebench
from .td_reader import read_td_dataset
from .typed_decision import evaluate_context
from .evaluators import (
    build_prompt,
    evaluate_document_qa,
    evaluate_exact_match_mc,
    evaluate_free_response,
    evaluate_function_calls,
    evaluate_graphwalks,
    evaluate_humaneval,
    evaluate_ifeval,
    evaluate_judge_external_responses_mc,
    evaluate_judge_mc,
    evaluate_livecodebench,
    evaluate_logprob_mc,
    evaluate_multi_turn_judge,
    evaluate_official_ifeval,
    evaluate_oneruler_tr,
    evaluate_safety_behavior,
)
from .reader import (
    read_document_qa_dataset,
    read_free_response_dataset,
    read_function_call_dataset,
    read_graphwalks_dataset,
    read_humaneval_dataset,
    read_ifeval_dataset,
    read_jsonl_dataset,
    read_livecodebench_dataset,
    read_multi_turn_judge_dataset,
    read_oneruler_dataset,
    read_safety_behavior_dataset,
)
from .scorer_table import SCORERS, ScorerDef, dataset_type
from .specialized_tasks import build_document_prompt, mtbench_candidate_messages


READERS: dict[str, Callable[[Any, int | None], list]] = {
    "typed_decision": read_td_dataset,
    "mc": lambda config, max_examples: read_jsonl_dataset(config),
    "free_response": lambda config, max_examples: read_free_response_dataset(config),
    "humaneval": lambda config, max_examples: read_humaneval_dataset(config),
    "livecodebench": lambda config, max_examples: read_livecodebench_dataset(config),
    "ifeval": lambda config, max_examples: read_ifeval_dataset(config, max_examples=max_examples),
    "graphwalks": lambda config, max_examples: read_graphwalks_dataset(config, max_examples=max_examples),
    "safety_behavior": lambda config, max_examples: read_safety_behavior_dataset(config),
    "document_qa": lambda config, max_examples: read_document_qa_dataset(config),
    "function_call": lambda config, max_examples: read_function_call_dataset(config),
    "multi_turn_judge": lambda config, max_examples: read_multi_turn_judge_dataset(config),
    "oneruler": lambda config, max_examples: read_oneruler_dataset(config, max_examples=max_examples),
}


def read_examples(config, max_examples: int | None) -> list:
    return READERS[dataset_type(config.task_type).internal](config, max_examples)


@dataclass(frozen=True)
class RunContext:
    """What one run hands to its evaluate function."""

    run: Any
    config: Any
    examples: list
    few_shot_examples: list
    few_shot_examples_per_example: list
    generation_settings: dict
    request_batch_size: int
    progress_callback: Callable[[int, int, int], None]
    output_level: str
    code_execution_timeout_seconds: float
    td_client: Any = None
    td_length_check: Any = None
    logprob_client: Any = None
    generation_client: Any = None
    judge_client: Any = None
    verifier_client: Any = None
    tool_call_client: Any = None
    code_executor: Any = None
    external_responses: Any = None

    @property
    def scorer(self) -> ScorerDef:
        return SCORERS[self.config.scorer_id]

    def max_tokens(self, default: int | None = None) -> int:
        return self.config.generation_max_tokens or (
            default if default is not None else self.scorer.default_max_tokens
        )

    def client(self, role: str):
        value = {
            "td": self.td_client,
            "logprob": self.logprob_client,
            "generation": self.generation_client,
            "judge": self.judge_client,
            "tool": self.tool_call_client,
            "executor": self.code_executor,
        }[role]
        if value is None:
            raise ValueError(
                f"Benchmark {self.run.run_name} ({self.config.scorer_id}) needs a {role} "
                "client, but none is available."
            )
        return value

    @property
    def batching(self) -> dict:
        return {
            "request_batch_size": self.request_batch_size,
            "progress_callback": self.progress_callback,
        }

    @property
    def multi_sample(self) -> dict:
        return multi_sample_settings(self.config)

    @property
    def few_shot(self) -> dict:
        return {
            "few_shot_examples": self.few_shot_examples,
            "few_shot_examples_per_example": self.few_shot_examples_per_example,
        }


def generation_request_settings(config) -> dict:
    """The sampling settings a task sends with every generation request."""
    return {
        "temperature": config.generation_temperature,
        "top_p": config.generation_top_p,
        "top_k": config.generation_top_k,
        "stop": list(config.generation_stop) or None,
    }


def multi_sample_settings(config) -> dict:
    return {
        "num_samples": config.generation_num_samples,
        "pass_at_k": list(config.generation_pass_at_k),
        "aggregation": config.generation_aggregation,
    }


def load_livecodebench_demonstrations(config) -> dict | None:
    """The upstream one-shot examples for completion endpoints, next to the dataset."""
    if config.path is None or str(config.path).startswith("s3://"):
        return None
    path = Path(config.path).parent / "base_model_demonstrations.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return {key: value[key] for key in ("functional", "stdin")}


def _logprob_mc(ctx: RunContext) -> dict:
    config = ctx.config
    return evaluate_logprob_mc(
        examples=ctx.examples,
        model_client=ctx.client("logprob"),
        generation_prompt=config.generation_prompt,
        append=config.append,
        score_target=config.score_target,
        normalization=config.normalization,
        pmi_reference_prompt=config.pmi_reference_prompt,
        length_normalization=config.length_normalization,
        **ctx.batching,
        include_analysis=ctx.output_level != "summary",
        include_api_calls=ctx.output_level == "debug",
        **ctx.few_shot,
    )


def _generated_mc(ctx: RunContext) -> dict:
    config = ctx.config
    return evaluate_exact_match_mc(
        examples=ctx.examples,
        model_client=ctx.client("generation"),
        generation_prompt=config.generation_prompt,
        append=config.append,
        score_target=config.score_target,
        match_mode=config.eval_mode,
        **ctx.few_shot,
        max_tokens=ctx.max_tokens(),
        generation_settings=ctx.generation_settings,
        **ctx.multi_sample,
        **ctx.batching,
        answer_verification=config.answer_verification,
        verifier_client=ctx.verifier_client,
    )


def _judge_mc(ctx: RunContext) -> dict:
    config = ctx.config
    if not config.judge_prompt:
        raise ValueError(
            f"Benchmark {ctx.run.run_name} uses eval_mode='judge' but does not define a judge_prompt."
        )
    judge_client = ctx.client("judge")
    if ctx.external_responses is not None:
        return evaluate_judge_external_responses_mc(
            examples=ctx.examples,
            external_responses=ctx.external_responses,
            dataset_name=ctx.run.dataset_name,
            judge_client=judge_client,
            judge_prompt=config.judge_prompt,
            **ctx.batching,
        )
    return evaluate_judge_mc(
        examples=ctx.examples,
        candidate_client=ctx.generation_client,
        judge_client=judge_client,
        generation_prompt=config.generation_prompt,
        append=config.append,
        judge_prompt=config.judge_prompt,
        **ctx.few_shot,
        max_tokens=ctx.max_tokens(),
        generation_settings=ctx.generation_settings,
        **ctx.batching,
    )


def _free_response(ctx: RunContext) -> dict:
    config = ctx.config
    return evaluate_free_response(
        examples=ctx.examples,
        model_client=ctx.client("generation"),
        generation_prompt=config.generation_prompt,
        max_tokens=ctx.max_tokens(),
        match_mode=config.eval_mode,
        **ctx.few_shot,
        generation_settings=ctx.generation_settings,
        **ctx.multi_sample,
        **ctx.batching,
        answer_verification=config.answer_verification,
        verifier_client=ctx.verifier_client,
    )


def _humaneval(ctx: RunContext) -> dict:
    return evaluate_humaneval(
        examples=ctx.examples,
        model_client=ctx.client("generation"),
        code_executor=ctx.client("executor"),
        max_tokens=ctx.max_tokens(),
        execution_timeout_seconds=ctx.code_execution_timeout_seconds,
        generation_settings=ctx.generation_settings,
        **ctx.multi_sample,
        **ctx.batching,
        chat_prompt=ctx.config.generation_prompt,
    )


def _livecodebench(ctx: RunContext) -> dict:
    return evaluate_livecodebench(
        examples=ctx.examples,
        model_client=ctx.client("generation"),
        code_executor=ctx.client("executor"),
        max_tokens=ctx.max_tokens(livecodebench.OFFICIAL_MAX_TOKENS),
        generation_settings=ctx.generation_settings,
        **ctx.multi_sample,
        base_model_demonstrations=load_livecodebench_demonstrations(ctx.config),
        **ctx.batching,
    )


def _ifeval(ctx: RunContext) -> dict:
    model_client = ctx.client("generation")
    package = instruction_following.PROTOCOLS.get(ctx.config.schema_version)
    if package is not None:
        return evaluate_official_ifeval(
            examples=ctx.examples,
            model_client=model_client,
            package=package,
            headline="loose" if package == "ifbench" else "strict",
            max_tokens=ctx.max_tokens(4096),
            generation_settings=ctx.generation_settings,
            **ctx.batching,
        )
    return evaluate_ifeval(
        examples=ctx.examples,
        model_client=model_client,
        max_tokens=ctx.max_tokens(),
        generation_settings=ctx.generation_settings,
        **ctx.batching,
    )


def _graphwalks(ctx: RunContext) -> dict:
    return evaluate_graphwalks(
        examples=ctx.examples,
        model_client=ctx.client("generation"),
        max_tokens=ctx.max_tokens(),
        generation_settings=ctx.generation_settings,
        **ctx.batching,
    )


def _oneruler_tr(ctx: RunContext) -> dict:
    return evaluate_oneruler_tr(
        examples=ctx.examples,
        model_client=ctx.client("generation"),
        max_tokens=ctx.max_tokens(),
        generation_settings=ctx.generation_settings,
        **ctx.batching,
    )


def _safety(ctx: RunContext) -> dict:
    return evaluate_safety_behavior(
        examples=ctx.examples,
        candidate_client=ctx.client("generation"),
        judge_client=ctx.client("judge"),
        max_tokens=ctx.max_tokens(),
        generation_settings=ctx.generation_settings,
        **ctx.batching,
    )


def _document_qa(ctx: RunContext) -> dict:
    return evaluate_document_qa(
        examples=ctx.examples,
        candidate_client=ctx.client("generation"),
        judge_client=ctx.client("judge"),
        max_tokens=ctx.max_tokens(),
        generation_settings=ctx.generation_settings,
        **ctx.batching,
    )


def _function_calls(ctx: RunContext) -> dict:
    return evaluate_function_calls(
        examples=ctx.examples,
        model_client=ctx.client("tool"),
        max_tokens=ctx.max_tokens(),
        generation_settings=ctx.generation_settings,
        **ctx.batching,
    )


def _multi_turn_judge(ctx: RunContext) -> dict:
    return evaluate_multi_turn_judge(
        examples=ctx.examples,
        candidate_client=ctx.client("generation"),
        judge_client=ctx.client("judge"),
        max_tokens=ctx.max_tokens(),
        generation_settings=ctx.generation_settings,
        **ctx.batching,
    )


@dataclass(frozen=True)
class ScorerRuntime:
    spec: ScorerDef
    evaluate: Callable[[RunContext], dict]


SCORER_RUNTIMES: dict[str, ScorerRuntime] = {
    scorer_id: ScorerRuntime(SCORERS[scorer_id], evaluate)
    for scorer_id, evaluate in {
        "typed_decision": evaluate_context,
        "mc_logprob": _logprob_mc,
        "mc_exact_match": _generated_mc,
        "mc_regex_match": _generated_mc,
        "mc_judge": _judge_mc,
        "free_response_exact_match": _free_response,
        "free_response_regex_match": _free_response,
        "humaneval": _humaneval,
        "ifeval": _ifeval,
        "graphwalks_exact_match": _graphwalks,
        "safety_judge": _safety,
        "document_qa": _document_qa,
        "bfcl_ast": _function_calls,
        "mtbench_judge": _multi_turn_judge,
        "livecodebench": _livecodebench,
        "oneruler_tr": _oneruler_tr,
    }.items()
}
assert set(SCORER_RUNTIMES) == set(SCORERS), "every scorer needs an evaluate binding"


def evaluate_run(context: RunContext) -> dict:
    return SCORER_RUNTIMES[context.config.scorer_id].evaluate(context)


def _inspect_mc(prepared, example, index):
    config = prepared.config
    return build_prompt(
        example,
        generation_prompt=config.generation_prompt,
        append=config.append,
        few_shot_examples=prepared.few_shot_examples_per_example[index],
        score_target=config.score_target,
    )


INSPECT_PROMPTS: dict[str, Callable[[Any, Any, int], Any]] = {
    "typed_decision": lambda prepared, example, index: {"state": example.state, "questions": example.questions},
    "mc": _inspect_mc,
    "free_response": _inspect_mc,
    "humaneval": lambda prepared, example, index: example.prompt,
    "safety_behavior": lambda prepared, example, index: example.prompt,
    "document_qa": lambda prepared, example, index: build_document_prompt(example),
    "function_call": lambda prepared, example, index: {
        "messages": example.messages,
        "tools": example.tools,
        "tool_choice": "auto",
    },
    "multi_turn_judge": lambda prepared, example, index: {
        str(turn_position): mtbench_candidate_messages(example, turn_position)
        for turn_position in example.eval_turn_positions
    },
    "livecodebench": lambda prepared, example, index: {
        "messages": livecodebench.chat_messages(example.question_content, example.starter_code)
    },
}


def inspect_prompt(prepared, example, index: int):
    render = INSPECT_PROMPTS.get(dataset_type(prepared.config.task_type).internal)
    if render is None:
        raise ValueError(
            f"Inspect currently supports multiple-choice and free-response tasks, not "
            f"{prepared.config.task_type!r}."
        )
    return render(prepared, example, index)
