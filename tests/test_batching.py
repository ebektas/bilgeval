"""Snapshot of how every evaluator reports a failed generation.

The scenarios drive each evaluator with a client that raises, returns the
wrong number of results, returns errors, or returns empty text, and compare
the error strings, the result-row and sample keys (in order), the progress
callbacks and the request counts with `fixtures/batching_snapshot.json`,
which was captured before the generation loops were unified. Regenerate the
fixture only for a deliberate change:

    BILGEVAL_UPDATE_SNAPSHOT=1 pytest tests/test_batching.py
"""

import json
import os
from pathlib import Path

import pytest

from bilgeval import livecodebench
from bilgeval.code_execution import CodeExecutionResult
from bilgeval.evaluators import (
    GeneratedTextResult,
    GeneratedToolCallResult,
    TextGenerationModelClient,
    evaluate_document_qa,
    evaluate_exact_match_mc,
    evaluate_free_response,
    evaluate_function_calls,
    evaluate_graphwalks,
    evaluate_humaneval,
    evaluate_ifeval,
    evaluate_judge_mc,
    evaluate_livecodebench,
    evaluate_multi_turn_judge,
    evaluate_official_ifeval,
    evaluate_oneruler_tr,
    evaluate_safety_behavior,
)
from bilgeval.reader import (
    Choice,
    DocumentQAExample,
    Example,
    FreeResponseExample,
    FunctionCallExample,
    GraphWalksExample,
    HumanEvalExample,
    IFEvalExample,
    LiveCodeBenchExample,
    MultiTurnJudgeExample,
    OfficialIFEvalExample,
    OneRulerExample,
    SafetyBehaviorExample,
)

SNAPSHOT = Path(__file__).parent / "fixtures" / "batching_snapshot.json"
MODES = ("raise", "mismatch", "error", "empty")


class Client(TextGenerationModelClient):
    generation_api = "chat"
    model = "m"

    def __init__(self, mode):
        self.mode = mode

    def _answer(self, count):
        if self.mode == "raise":
            raise RuntimeError("boom")
        if self.mode == "mismatch":
            return [GeneratedTextResult(text="x")] * (count + 1)
        if self.mode == "error":
            return [GeneratedTextResult(error="upstream 504") for _ in range(count)]
        if self.mode == "empty":
            return [GeneratedTextResult(text="   ", finish_reason="stop") for _ in range(count)]
        return [GeneratedTextResult(text="Cevap: A", finish_reason="stop") for _ in range(count)]

    def generate_texts(self, prompts, **kwargs):
        return self._answer(len(prompts))

    def generate_conversations(self, conversations, **kwargs):
        return self._answer(len(conversations))


class ToolClient:
    def __init__(self, mode):
        self.mode = mode

    def generate_tool_calls(self, examples, **kwargs):
        if self.mode == "raise":
            raise RuntimeError("boom")
        if self.mode == "mismatch":
            return [GeneratedToolCallResult(tool_calls=[])] * (len(examples) + 1)
        if self.mode == "error":
            return [GeneratedToolCallResult(error="upstream 504") for _ in examples]
        return [GeneratedToolCallResult(tool_calls=[], content="") for _ in examples]


class Executor:
    name = "fixture"

    def execute(self, program, timeout_seconds, **kwargs):
        return CodeExecutionResult(status="passed", elapsed_ms=1, exit_code=42)


MC = Example(id="m1", question="Q?", choices=[Choice("A", "x"), Choice("B", "y")], answer_label="A", answer_text="x", raw={})
FREE = FreeResponseExample(id="f1", question="2+2?", answer="4", answer_text="4", raw={})
HUMANEVAL = HumanEvalExample(id="h1", prompt="def f():\n", test="def check(c):\n    pass\n", entry_point="f", canonical_solution=None, raw={})
LCB = LiveCodeBenchExample(id="l1", question_title="t", question_content="c", starter_code="", fn_name=None, num_tests=1, tests=livecodebench.encode_tests(["1"], ["1"], None), raw={})
GRAPH = GraphWalksExample(id="g1", prompt="p", answer_nodes=["a"], problem_type="bfs", raw={})
ONERULER = OneRulerExample(id="r1", prompt="p", outputs=["1234567"], task="niah_single", raw={})
IFEVAL = IFEvalExample(id="i1", prompt="p", instruction_ids=[26], verifier_kwargs={"26": {}}, raw={})
OFFICIAL = OfficialIFEvalExample(id="o1", prompt="p", instruction_ids=["punctuation:no_comma"], kwargs=[{}], raw={})
SAFETY = SafetyBehaviorExample(id="s1", prompt="p", subcategory="Dangerous or sensitive topics", expected="must_refuse", raw={})
DOC = DocumentQAExample(id="d1", document="doc", question="q", answer="a", answer_format="free_form", task_type="open_qa", evidence_spans=["e"], raw={})
MTBENCH = MultiTurnJudgeExample(id="t1", task_id="SI", task_name="SI", topic="x", turns=[{"role": "human", "content": "a"}, {"role": "assistant", "content": "b"}], eval_turn_positions=[0], evaluation_initial="e", evaluation_prompt="p", scoring_format="Rating: [[score]]", raw={})
FUNCTION = FunctionCallExample(id="c1", messages=[{"role": "user", "content": "hi"}], tools=[], expected_calls=[], should_call=False, call_order="any", scoring_mode="call_presence", category="irrelevance", raw={})

SCENARIOS = {
    "exact_match_mc": lambda m, cb: evaluate_exact_match_mc([MC], Client(m), generation_prompt="turkish_mc_v1", append=True, score_target="label", match_mode="regex_match", progress_callback=cb),
    "free_response": lambda m, cb: evaluate_free_response([FREE], Client(m), generation_prompt="turkish_integer_answer_v1", progress_callback=cb),
    "judge_mc(candidate)": lambda m, cb: evaluate_judge_mc([MC], candidate_client=Client(m), judge_client=Client("ok"), generation_prompt="none", append=True, judge_prompt="turkish_mc_correctness_v2", progress_callback=cb),
    "judge_mc(judge)": lambda m, cb: evaluate_judge_mc([MC], candidate_client=Client("ok"), judge_client=Client(m), generation_prompt="none", append=True, judge_prompt="turkish_mc_correctness_v2", progress_callback=cb),
    "humaneval": lambda m, cb: evaluate_humaneval([HUMANEVAL], Client(m), Executor(), progress_callback=cb),
    "livecodebench": lambda m, cb: evaluate_livecodebench([LCB], Client(m), Executor(), progress_callback=cb),
    "graphwalks": lambda m, cb: evaluate_graphwalks([GRAPH], Client(m), progress_callback=cb),
    "oneruler_tr": lambda m, cb: evaluate_oneruler_tr([ONERULER], Client(m), progress_callback=cb),
    "ifeval": lambda m, cb: evaluate_ifeval([IFEVAL], Client(m), progress_callback=cb),
    "official_ifeval": lambda m, cb: evaluate_official_ifeval([OFFICIAL], Client(m), package="google_ifeval", progress_callback=cb),
    "safety(candidate)": lambda m, cb: evaluate_safety_behavior([SAFETY], candidate_client=Client(m), judge_client=Client("ok"), progress_callback=cb),
    "safety(judge)": lambda m, cb: evaluate_safety_behavior([SAFETY], candidate_client=Client("ok"), judge_client=Client(m), progress_callback=cb),
    "document_qa(candidate)": lambda m, cb: evaluate_document_qa([DOC], candidate_client=Client(m), judge_client=Client("ok"), progress_callback=cb),
    "document_qa(judge)": lambda m, cb: evaluate_document_qa([DOC], candidate_client=Client("ok"), judge_client=Client(m), progress_callback=cb),
    "mtbench(candidate)": lambda m, cb: evaluate_multi_turn_judge([MTBENCH], candidate_client=Client(m), judge_client=Client("ok"), progress_callback=cb),
    "mtbench(judge)": lambda m, cb: evaluate_multi_turn_judge([MTBENCH], candidate_client=Client("ok"), judge_client=Client(m), progress_callback=cb),
    "function_calls": lambda m, cb: evaluate_function_calls([FUNCTION], ToolClient(m), progress_callback=cb),
}


def observe(name: str, mode: str) -> dict:
    progress = []
    try:
        report = SCENARIOS[name](mode, lambda a, b, c: progress.append([a, b, c]))
    except Exception as exc:
        return {"raised": f"{type(exc).__name__}: {exc}"}
    row = report["results"][0]
    observed = {
        "failed": row.get("failed"),
        "error": row.get("error"),
        "errors": row.get("errors"),
        "keys": list(row.keys()),
        "progress": progress,
        "num_requests": report["num_requests"],
        "num_failed_examples": report["num_failed_examples"],
        "num_scored_examples": report["num_scored_examples"],
    }
    if "samples" in row:
        observed["sample_error"] = row["samples"][0].get("error")
        observed["sample_keys"] = list(row["samples"][0].keys())
    if "turns" in row:
        observed["turn_error"] = row["turns"][0].get("error")
        observed["turn_keys"] = list(row["turns"][0].keys())
    if "judge_checks" in row:
        observed["judge_check_keys"] = [list(check.keys()) for check in row["judge_checks"]]
    return observed


def snapshot() -> dict:
    return {f"{name}/{mode}": observe(name, mode) for name in SCENARIOS for mode in MODES}


@pytest.mark.skipif(not os.environ.get("BILGEVAL_UPDATE_SNAPSHOT"), reason="writes the snapshot")
def test_update_snapshot():
    SNAPSHOT.write_text(json.dumps(snapshot(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_failure_reporting_matches_the_snapshot(name):
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for mode in MODES:
        assert observe(name, mode) == expected[f"{name}/{mode}"], (name, mode)
