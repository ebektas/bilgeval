"""Preflight: one cheap probe per client before the real requests start.

A probe asks each client the kind of request the run will make and checks
the shape of the answer: a judge must return a verdict it can parse, a
verifier `{"match": true}`, a logprob endpoint finite scores on every
choice, a tool endpoint a native tool call, an executor a passing program.
Probes are per scorer (`PROBES`) and dedupe on what they exercise: one
generation probe per prompt family and generation settings, one judge
probe per judge prompt, one verifier, one executor, one tool call.

The probe texts are fixed; the equivalence stub answers them by text.
"""

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from . import instruction_following, livecodebench
from .evaluators import (
    build_answer_verifier_prompt,
    build_judge_prompt,
    build_prompt,
    build_scored_completion,
    extract_non_empty_generated_text,
    parse_answer_verifier_response,
    parse_judge_verdict,
)
from .manifest import stable_json_sha256
from .reader import Choice, Example, FunctionCallExample
from .scorer_table import SCORERS
from .scorers import generation_request_settings
from .specialized_tasks import parse_mtbench_rating, parse_pass_fail, parse_safety_judge_label


class PreflightError(RuntimeError):
    pass


PREFLIGHT_EXAMPLE = Example(
    id="preflight",
    question="2 + 2 kac eder?",
    choices=[
        Choice(label="A", text="3"),
        Choice(label="B", text="4"),
        Choice(label="C", text="5"),
        Choice(label="D", text="6"),
    ],
    answer_label="B",
    answer_text="4",
    raw={},
)


FUNCTION_CALL_PREFLIGHT_EXAMPLE = FunctionCallExample(
    id="function_call_preflight",
    messages=[
        {
            "role": "user",
            "content": "Call add_numbers with left=2 and right=3. Use the provided tool.",
        }
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "left": {"type": "integer"},
                        "right": {"type": "integer"},
                    },
                    "required": ["left", "right"],
                },
            },
        }
    ],
    expected_calls=[
        {
            "name": "add_numbers",
            "arguments": {"left": [2], "right": [3]},
        }
    ],
    should_call=True,
    call_order="unordered",
    scoring_mode="exact",
    category="preflight",
    raw={},
)


def preflight_budget(config, minimum: int) -> int:
    """Preflight token budget: never below the task's own candidate budget.

    Thinking models spend tokens in reasoning before any visible answer, so a
    fixed 32/64-token probe fails even when the real run would succeed.
    """
    configured = getattr(config, "generation_max_tokens", None) or 0
    return max(minimum, int(configured))


@dataclass
class Preflight:
    """The clients under test and what has been probed so far."""

    td_client: Any = None
    td_checked: bool = False
    logprob_client: Any = None
    generation_client: Any = None
    tool_call_client: Any = None
    judge_client: Any = None
    verifier_client: Any = None
    code_executor: Any = None
    code_execution_timeout_seconds: float = 3.0
    external_responses_enabled: bool = False
    show_progress: bool = False
    checked_logprob: set = field(default_factory=set)
    checked_generation: set = field(default_factory=set)
    checked_judge: set = field(default_factory=set)
    verifier_checked: bool = False
    code_executor_checked: bool = False
    livecodebench_executor_checked: bool = False
    tool_call_checked: bool = False

    def say(self, message: str) -> None:
        if self.show_progress:
            print(f"[preflight] {message}", file=sys.stderr, flush=True)


def _generated_text(result, role: str, failure: str, model: str) -> str:
    text, error = extract_non_empty_generated_text(result, role=role)
    if error is not None:
        raise PreflightError(f"{failure}: model={model} error={error}")
    return text


def _generate(client, prompts: list, *, role: str, failure: str, **settings):
    """One generate_texts call whose exception or empty answer is a PreflightError."""
    try:
        result = client.generate_texts(prompts, **settings)[0]
    except Exception as exc:
        raise PreflightError(f"{failure}: model={client.model} error={exc}") from exc
    _generated_text(result, role, failure, client.model)
    return result


def _probe_function_call(state: Preflight, run, config) -> None:
    if state.tool_call_client is None:
        raise PreflightError("Function-call preflight requires a native tool-call client.")
    if state.tool_call_checked:
        return
    state.say("native function calling")
    generated = state.tool_call_client.generate_tool_calls(
        [FUNCTION_CALL_PREFLIGHT_EXAMPLE],
        max_tokens=preflight_budget(config, 64),
        temperature=0.0,
    )[0]
    if generated.error is not None:
        raise PreflightError(
            "Function-call preflight failed: "
            f"model={state.tool_call_client.model} error={generated.error}"
        )
    if not generated.tool_calls:
        raise PreflightError(
            "Function-call preflight failed: the model returned no native "
            "tool_calls. Use --skip-preflight only when this is expected."
        )
    state.tool_call_checked = True


def _probe_livecodebench(state: Preflight, run, config) -> None:
    client = state.generation_client
    if client is None:
        raise PreflightError("LiveCodeBench preflight requires a text generation client.")
    if state.code_executor is None:
        raise PreflightError("LiveCodeBench preflight requires a code executor.")
    if not state.livecodebench_executor_checked:
        tests = livecodebench.encode_tests(["7\n" + "0" * 1_000_000], ["14"], None)
        execution = state.code_executor.execute(
            livecodebench.build_program("import sys\nprint(int(sys.stdin.readline()) * 2)\n"),
            livecodebench.execution_timeout_seconds(1, livecodebench.OFFICIAL_TEST_TIMEOUT_SECONDS),
            attachments={livecodebench.TESTS_ATTACHMENT: livecodebench.tests_attachment(tests)},
            memory_limit_mb=livecodebench.DEFAULT_MEMORY_LIMIT_MB,
        )
        if not execution.passed:
            raise PreflightError(
                "LiveCodeBench executor preflight failed: "
                f"executor={state.code_executor.name} status={execution.status} "
                f"error={execution.error}. An executor service must run the "
                "current bilgeval code (restart it after an update)."
            )
        state.livecodebench_executor_checked = True
    generation_key = ("livecodebench", stable_json_sha256(generation_request_settings(config)))
    if generation_key in state.checked_generation:
        return
    settings = generation_request_settings(config)
    settings["stop"] = None
    failure = "LiveCodeBench generation preflight failed"
    try:
        if client.generation_api == "chat":
            result = client.generate_conversations(
                [livecodebench.chat_messages("Read an integer and print it.", "")],
                max_tokens=preflight_budget(config, 64),
                **settings,
            )[0]
        else:
            result = client.generate_texts(
                ["### Question\nRead an integer and print it.\n\n### Answer\n\n"],
                max_tokens=preflight_budget(config, 64),
                **settings,
            )[0]
    except Exception as exc:
        raise PreflightError(f"{failure}: model={client.model} error={exc}") from exc
    _generated_text(result, "LiveCodeBench generation preflight", failure, client.model)
    state.checked_generation.add(generation_key)


def _probe_humaneval(state: Preflight, run, config) -> None:
    client = state.generation_client
    if client is None:
        raise PreflightError("HumanEval preflight requires a text generation client.")
    if state.code_executor is None:
        raise PreflightError("HumanEval preflight requires a code executor.")
    if not state.code_executor_checked:
        execution = state.code_executor.execute(
            "def identity(value):\n    return value\n\n"
            "def check(candidate):\n    assert candidate(7) == 7\n\n"
            "check(identity)\n",
            state.code_execution_timeout_seconds,
        )
        if not execution.passed:
            raise PreflightError(
                "HumanEval executor preflight failed: "
                f"executor={state.code_executor.name} status={execution.status} "
                f"error={execution.error}"
            )
        state.code_executor_checked = True
    generation_key = ("humaneval", stable_json_sha256(generation_request_settings(config)))
    if generation_key in state.checked_generation:
        return
    settings = generation_request_settings(config)
    if client.generation_api == "chat":
        settings["stop"] = None
    _generate(
        client,
        [
            "Yalnızca tamamlanmış Python kodunu döndür.\n\n"
            "def add_one(value: int) -> int:\n"
            "    \"\"\"Return value + 1.\"\"\"\n"
        ],
        role="HumanEval generation preflight",
        failure="HumanEval generation preflight failed",
        max_tokens=preflight_budget(config, 32),
        **settings,
    )
    state.checked_generation.add(generation_key)


def _probe_mtbench(state: Preflight, run, config) -> None:
    client, judge = state.generation_client, state.judge_client
    if client is None or judge is None:
        raise PreflightError("MTBench preflight requires candidate and judge generation clients.")
    generation_key = ("multi_turn_judge", stable_json_sha256(generation_request_settings(config)))
    if generation_key not in state.checked_generation:
        candidate_result = client.generate_conversations(
            [[{"role": "user", "content": "Yalnızca 'tamam' yaz."}]],
            max_tokens=preflight_budget(config, 16),
            **generation_request_settings(config),
        )[0]
        _, candidate_error = extract_non_empty_generated_text(
            candidate_result, role="MTBench candidate preflight"
        )
        if candidate_error is not None:
            raise PreflightError(
                f"MTBench candidate preflight failed: model={client.model} error={candidate_error}"
            )
        state.checked_generation.add(generation_key)

    judge_key = "specialized:multi_turn_judge"
    if judge_key in state.checked_judge:
        return
    judge_result = judge.generate_conversations(
        [[
            {"role": "system", "content": "Return the requested rating format exactly."},
            {"role": "user", "content": "Return only: Rating: [[8]]"},
        ]],
        max_tokens=16,
        temperature=0.0,
        stop=None,
    )[0]
    judge_text, judge_error = extract_non_empty_generated_text(judge_result, role="MTBench judge preflight")
    if judge_error is not None or parse_mtbench_rating(judge_text or "") != 8.0:
        raise PreflightError(
            "MTBench judge preflight failed: expected rating 8, "
            f"got={judge_result.text!r}, error={judge_error}"
        )
    state.checked_judge.add(judge_key)


def _probe_judged_generation(state: Preflight, run, config) -> None:
    """Safety and document QA: a Turkish answer, then a judge label."""
    kind = config.task_type
    client, judge = state.generation_client, state.judge_client
    if client is None:
        raise PreflightError(f"{kind} preflight requires a text generation client.")
    if judge is None:
        raise PreflightError(f"{kind} preflight requires a judge client.")
    generation_key = (kind, stable_json_sha256(generation_request_settings(config)))
    if generation_key not in state.checked_generation:
        _generate(
            client,
            ["Bu soruya kısa bir Türkçe yanıt ver: Türkiye'nin başkenti neresidir?"],
            role=f"{kind} generation preflight",
            failure=f"{kind} generation preflight failed",
            max_tokens=preflight_budget(config, 32),
            **generation_request_settings(config),
        )
        state.checked_generation.add(generation_key)

    judge_key = f"specialized:{kind}"
    if judge_key in state.checked_judge:
        return
    expected = "<label>ACCEPTABLE</label>" if kind == "safety_behavior" else "PASS"
    prompt = f"Bu bir API biçim kontrolüdür. Yalnızca şu çıktıyı döndür: {expected}"
    try:
        judge_result = judge.generate_texts([prompt], max_tokens=16, temperature=0.0, stop=None)[0]
    except Exception as exc:
        raise PreflightError(f"{kind} judge preflight failed: model={judge.model} error={exc}") from exc
    judge_text, judge_error = extract_non_empty_generated_text(judge_result, role=f"{kind} judge preflight")
    parsed = (
        parse_safety_judge_label(judge_text or "")
        if kind == "safety_behavior"
        else parse_pass_fail(judge_text or "")
    )
    if judge_error is not None or parsed not in {"ACCEPTABLE", True}:
        raise PreflightError(
            f"{kind} judge preflight failed: expected={expected!r}, "
            f"got={judge_result.text!r}, error={judge_error}"
        )
    state.checked_judge.add(judge_key)


def _probe_instruction(state: Preflight, run, config) -> None:
    """GraphWalks, IFEval and OneRuler: one short free-form answer."""
    generation_key = (
        config.task_type,
        config.eval_mode,
        str(config.path),
        stable_json_sha256(generation_request_settings(config)),
    )
    official_ifeval = config.schema_version in instruction_following.PROTOCOLS
    if official_ifeval:
        try:
            instruction_following.verifier_info(instruction_following.PROTOCOLS[config.schema_version])
        except instruction_following.MissingVerifierDependency as exc:
            raise PreflightError(str(exc)) from exc
    if generation_key in state.checked_generation:
        return
    client = state.generation_client
    if client is None:
        raise PreflightError(f"{config.task_type} preflight requires a text generation client.")
    state.say(f"{config.task_type} generation")
    _generate(
        client,
        [
            "Answer this short instruction with one English sentence."
            if official_ifeval
            else "Bu kısa talimata tek cümlelik Türkçe bir yanıt ver."
        ],
        role=f"{config.task_type} generation preflight",
        failure=f"{config.task_type} generation preflight failed",
        max_tokens=preflight_budget(config, 32),
        **generation_request_settings(config),
    )
    state.checked_generation.add(generation_key)


def _probe_generated_answer(state: Preflight, run, config) -> None:
    """Generated multiple choice and free response: the candidate prompt,
    then the answer verifier and (for judged tasks) the judge prompt."""
    generation_key = (
        config.generation_prompt,
        config.append,
        config.eval_mode,
        stable_json_sha256(generation_request_settings(config)),
    )
    skip_candidate = state.external_responses_enabled and config.eval_mode == "judge"
    if not skip_candidate and generation_key not in state.checked_generation:
        client = state.generation_client
        if client is None:
            raise PreflightError("Generation preflight requires a text generation client.")
        prompt = build_prompt(
            PREFLIGHT_EXAMPLE,
            generation_prompt=config.generation_prompt,
            append=config.append,
        )
        state.say(
            "generation "
            f"eval_mode={config.eval_mode} "
            f"generation_prompt={config.generation_prompt} "
            f"append={config.append} "
            f"score_target={config.score_target}"
        )
        _generate(
            client,
            [prompt],
            role="Generation preflight",
            failure="Generation preflight failed",
            max_tokens=preflight_budget(config, 32),
            **generation_request_settings(config),
        )
        state.checked_generation.add(generation_key)

    if config.answer_verification != "regex" and not state.verifier_checked:
        verifier = state.verifier_client
        if verifier is None:
            raise PreflightError("Answer verifier preflight requires a verifier client.")
        state.say(f"answer verifier model={verifier.model}")
        try:
            result = verifier.generate_texts(
                [build_answer_verifier_prompt("42", "Sonuç: 42,0")],
                max_tokens=16,
                temperature=0.0,
                stop=None,
            )[0]
        except Exception as exc:
            raise PreflightError(
                f"Answer verifier preflight failed: verifier_model={verifier.model} error={exc}"
            ) from exc
        response, error = extract_non_empty_generated_text(result, role="Answer verifier preflight")
        if error is not None:
            raise PreflightError(
                f"Answer verifier preflight failed: verifier_model={verifier.model} error={error}"
            )
        if parse_answer_verifier_response(response) is not True:
            raise PreflightError(
                "Answer verifier preflight failed: "
                f"verifier_model={verifier.model} expected match=true got={response!r}"
            )
        state.verifier_checked = True

    if config.eval_mode != "judge":
        return
    judge_prompt_name = config.judge_prompt
    if judge_prompt_name is None:
        raise PreflightError(
            f"Benchmark {run.run_name} uses eval_mode='judge' but does not define a judge_prompt."
        )
    if judge_prompt_name in state.checked_judge:
        return
    judge = state.judge_client
    if judge is None:
        raise PreflightError("Judge preflight requires a judge client.")
    judge_prompt_text = build_judge_prompt(
        PREFLIGHT_EXAMPLE,
        candidate_answer=f"{PREFLIGHT_EXAMPLE.answer_label}) {PREFLIGHT_EXAMPLE.answer_text}",
        prompt_name=judge_prompt_name,
    )
    state.say(f"judge judge_prompt={judge_prompt_name}")
    try:
        result = judge.generate_texts([judge_prompt_text], max_tokens=8, stop=None)[0]
    except Exception as exc:
        raise PreflightError(
            f"Judge generation preflight failed: judge_model={judge.model} error={exc}"
        ) from exc
    response, error = extract_non_empty_generated_text(result, role="Judge generation preflight")
    if error is not None:
        raise PreflightError(
            f"Judge generation preflight failed: judge_model={judge.model} error={error}"
        )
    if parse_judge_verdict(response) != "CORRECT":
        raise PreflightError(
            "Judge generation preflight failed: "
            f"judge_model={judge.model} expected=CORRECT got={response!r}"
        )
    state.checked_judge.add(judge_prompt_name)


def _probe_logprob(state: Preflight, run, config) -> None:
    logprob_key = (
        config.generation_prompt,
        config.append,
        config.score_target,
        config.normalization,
        config.pmi_reference_prompt,
        config.length_normalization,
    )
    if logprob_key in state.checked_logprob:
        return
    client = state.logprob_client
    if client is None:
        raise PreflightError("Logprob preflight requires a logprob model client.")
    prompt = build_prompt(
        PREFLIGHT_EXAMPLE,
        generation_prompt=config.generation_prompt,
        append=config.append,
    )
    pairs = []
    labels = []
    for choice in PREFLIGHT_EXAMPLE.choices:
        completion = build_scored_completion(
            choice_label=choice.label,
            choice_text=choice.text,
            score_target=config.score_target,
        )
        pairs.append((prompt, completion))
        labels.append(f"{choice.label}:conditional")
        if config.normalization == "pmi":
            pairs.append((config.pmi_reference_prompt or "", completion))
            labels.append(f"{choice.label}:pmi_reference")
    state.say(
        "logprob "
        f"generation_prompt={config.generation_prompt} "
        f"append={config.append} "
        f"score_target={config.score_target} "
        f"normalization={config.normalization}"
    )
    try:
        scores = client.score_prompt_completion_pairs(
            pairs,
            normalization="none" if config.normalization == "pmi" else config.normalization,
            length_normalization=config.length_normalization,
        )
    except Exception as exc:
        raise PreflightError(f"Logprob preflight failed: model={client.model} error={exc}") from exc
    if len(scores) != len(pairs):
        raise PreflightError(
            "Logprob preflight failed: "
            f"model={client.model} returned {len(scores)} scores for "
            f"{len(pairs)} conditional/reference scores."
        )
    for label, score in zip(labels, scores):
        if score.error is not None:
            raise PreflightError(
                f"Logprob preflight failed: model={client.model} score={label} error={score.error}"
            )
        if score.score is None or not math.isfinite(score.score) or score.num_completion_tokens < 1:
            raise PreflightError(
                f"Logprob preflight failed: model={client.model} score={label} did not return a valid score."
            )
    state.checked_logprob.add(logprob_key)


def _probe_td(state, run, config):
    if not state.td_checked:
        try:
            state.td_client.probe()
        except Exception as exc:
            raise PreflightError(f"TD C-01..C-06 probe failed: {exc}") from exc
        state.td_checked = True


PROBES: dict[str, Callable[[Preflight, Any, Any], None]] = {
    "typed_decision": _probe_td,
    "mc_logprob": _probe_logprob,
    "mc_exact_match": _probe_generated_answer,
    "mc_regex_match": _probe_generated_answer,
    "mc_judge": _probe_generated_answer,
    "free_response_exact_match": _probe_generated_answer,
    "free_response_regex_match": _probe_generated_answer,
    "humaneval": _probe_humaneval,
    "ifeval": _probe_instruction,
    "graphwalks_exact_match": _probe_instruction,
    "safety_judge": _probe_judged_generation,
    "document_qa": _probe_judged_generation,
    "bfcl_ast": _probe_function_call,
    "mtbench_judge": _probe_mtbench,
    "livecodebench": _probe_livecodebench,
    "oneruler_tr": _probe_instruction,
}
assert set(PROBES) == set(SCORERS), "every scorer needs a preflight probe"


def run_preflight_checks(
    selected_runs: list,
    resolved_run_configs: dict[str, object],
    model_client=None,
    candidate_generation_client=None,
    tool_call_client=None,
    judge_client=None,
    verifier_client=None,
    code_executor=None,
    code_execution_timeout_seconds: float = 3.0,
    external_responses_enabled: bool = False,
    show_progress: bool = False,
    td_client=None,
) -> None:
    state = Preflight(
        td_client=td_client,
        logprob_client=model_client,
        generation_client=candidate_generation_client,
        tool_call_client=tool_call_client,
        judge_client=judge_client,
        verifier_client=verifier_client,
        code_executor=code_executor,
        code_execution_timeout_seconds=code_execution_timeout_seconds,
        external_responses_enabled=external_responses_enabled,
        show_progress=show_progress,
    )
    for run in selected_runs:
        config = resolved_run_configs[run.run_name]
        PROBES[config.scorer_id](state, run, config)
