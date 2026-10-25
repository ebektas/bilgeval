"""AIME 2025, GPQA Diamond, GSM8K, HumanEval and LiveCodeBench v6 in English: prompt
forms are compared with the upstream text, settings with the upstream
defaults, and the LiveCodeBench grader runs in the real sandbox."""

import json
import shutil
import socket
import threading
from collections import Counter

import pytest

from bilgeval import livecodebench
from bilgeval import main as main_module
from bilgeval.code_execution import (
    BubblewrapCodeExecutor,
    CodeExecutionResult,
    UnixSocketCodeExecutor,
    _handle_executor_connection,
    validate_attachments,
)
from bilgeval.evaluators import (
    GeneratedTextResult,
    TextGenerationModelClient,
    evaluate_humaneval,
    evaluate_livecodebench,
    mean_sample_report,
)
from bilgeval.generation_prompts import build_generation_prompt
from bilgeval.reader import (
    Choice,
    Example,
    HumanEvalExample,
    LiveCodeBenchExample,
    read_free_response_dataset,
    read_humaneval_dataset,
    read_jsonl_dataset,
    read_livecodebench_dataset,
)
from bilgeval.registry import DATASETS
from bilgeval.registry_bundle import load_registry_bundle


REGISTRY = main_module.DEFAULT_REGISTRY_PATH if hasattr(main_module, "DEFAULT_REGISTRY_PATH") else None
needs_sandbox = pytest.mark.skipif(shutil.which("bwrap") is None, reason="needs bubblewrap")


@pytest.fixture(scope="module")
def registry():
    from pathlib import Path

    path = Path(main_module.__file__).with_name("builtin_registry.json")
    return load_registry_bundle(path)


def question(text="Which one?", gold=1, explanation=None):
    choices = [Choice(label=label, text=f"choice {label}") for label in "ABCD"]
    return Example(
        id=text,
        question=text,
        choices=choices,
        answer_label="ABCD"[gold],
        answer_text=choices[gold].text,
        raw={"explanation": explanation} if explanation else {},
    )


CHOICES = "(A) choice A\n(B) choice B\n(C) choice C\n(D) choice D"


def test_matharena_prompt_is_instruction_blank_line_problem():
    prompt = build_generation_prompt(question("Find $n$."), "matharena_aime_v1")
    assert prompt == (
        "Put your final answer within \\boxed{}.\n"
        "The answer is an integer between 0 and 999 inclusive.\n\nFind $n$."
    )






def test_gpqa_zero_shot_prompt_matches_the_authors_baseline():
    prompt = build_generation_prompt(question(), "gpqa_zero_shot_v1", append_choices=True)
    assert prompt == (
        "What is the correct answer to this question: Which one?\n\n"
        f"Choices:\n{CHOICES}\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)"'
    )


def test_gpqa_few_shot_prompts_match_the_authors_baselines():
    demonstrations = [question("Demo?", gold=3, explanation="Because D.")]
    five_shot = build_generation_prompt(
        question(),
        "gpqa_5shot_v1",
        append_choices=True,
        few_shot_examples=demonstrations,
        score_target="label",
    )
    assert five_shot == (
        "Here are some example questions from experts. Answer the final question "
        "yourself, following the format of the previous questions exactly.\n"
        f"Question: Demo?\nChoices:\n{CHOICES}\nThe correct answer is (D)\n"
        f"Question: Which one?\nChoices:\n{CHOICES}\n"
        "When you're ready to answer, please use the format "
        '"The correct answer is (insert answer here).'
    )
    chain_of_thought = build_generation_prompt(
        question(),
        "gpqa_cot_5shot_v1",
        append_choices=True,
        few_shot_examples=demonstrations,
        score_target="label",
    )
    assert chain_of_thought == (
        "Here are some example questions from experts. An explanation is given before "
        "the final answer. Answer the final question yourself, giving your reasoning "
        "beforehand.\n"
        f"Question: Demo?\nChoices:\n{CHOICES}\n"
        "Let's think step by step: \nBecause D.\nThe correct answer is (D)\n"
        f"Question: Which one?\nChoices:\n{CHOICES}\n"
        "Give step by step reasoning before you answer, and when you're ready to answer, "
        'please use the format "The correct answer is (insert answer here)":\n'
    )


def test_simple_evals_multiple_choice_prompts():
    english = build_generation_prompt(question(), "simple_evals_mc_v1", append_choices=True)
    turkish = build_generation_prompt(
        question("Hangisi?"), "simple_evals_mc_tr_v1", append_choices=True
    )

    assert english == (
        "Answer the following multiple choice question. The last line of your response "
        "should be of the following format: 'Answer: $LETTER' (without quotes) where "
        "LETTER is one of ABCD. Think step by step before answering.\n\n"
        "Which one?\n\nA) choice A\nB) choice B\nC) choice C\nD) choice D"
    )
    assert turkish == (
        "Aşağıdaki çoktan seçmeli soruyu yanıtla. Yanıtının son satırı, HARF'in A, B, C "
        "veya D harflerinden biri olduğu 'Yanıt: $HARF' (tırnak işaretleri olmadan) "
        "biçiminde olmalıdır. Yanıtlamadan önce adım adım düşün.\n\n"
        "Hangisi?\n\nA) choice A\nB) choice B\nC) choice C\nD) choice D"
    )








def test_livecodebench_prompts_and_extraction_follow_the_official_runner():
    stdin = livecodebench.chat_messages("Print 1.", "")
    assert stdin[0] == {"role": "system", "content": livecodebench.SYSTEM_MESSAGE_GENERIC}
    assert stdin[1]["content"] == (
        "### Question:\nPrint 1.\n\n"
        f"### Format: {livecodebench.FORMATTING_WITHOUT_STARTER_CODE}\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )
    functional = livecodebench.user_prompt("Add.", "class Solution:\n    def add(self): ")
    assert (
        f"### Format: {livecodebench.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        "```python\nclass Solution:\n    def add(self): \n```\n\n"
    ) in functional

    response = "First try:\n```python\nprint(0)\n```\nBetter:\n```python\nprint(1)\n```\nDone."
    assert livecodebench.extract_code(response) == "print(1)"
    assert livecodebench.extract_code("print(1)") == ""
    assert livecodebench.extract_code(" print(1) \n", base_model=True) == "print(1)"


def test_upstream_private_tests_decode_without_unpickling_objects():
    import base64
    import pickle
    import zlib

    tests = [{"input": "1", "output": "2", "testtype": "stdin"}]
    encoded = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
    assert livecodebench.decode_upstream_private_tests(encoded) == tests
    assert livecodebench.decode_upstream_private_tests(json.dumps(tests)) == tests

    hostile = base64.b64encode(zlib.compress(pickle.dumps(print))).decode()
    with pytest.raises(pickle.UnpicklingError):
        livecodebench.decode_upstream_private_tests(hostile)


def grade(code, inputs, outputs, fn_name=None, timeout=1):
    execution = BubblewrapCodeExecutor().execute(
        livecodebench.build_program(code, timeout),
        livecodebench.execution_timeout_seconds(len(inputs), timeout),
        attachments={
            livecodebench.TESTS_ATTACHMENT: livecodebench.tests_attachment(
                livecodebench.encode_tests(inputs, outputs, fn_name)
            )
        },
        memory_limit_mb=1024,
    )
    if execution.passed:
        return "passed"
    return livecodebench.VERDICT_BY_EXIT_CODE.get(execution.exit_code, execution.status)


@needs_sandbox
def test_grader_stdin_problems():
    solution = "import sys\nn = int(sys.stdin.readline())\nprint(n * 2)\n"
    assert grade(solution, ["3\n", "10"], ["6\n", "20"]) == "passed"
    assert grade("print(int(input()) * 2)\n", ["3"], ["6"]) == "passed"
    guarded = "def main():\n    print(int(input()) * 2)\nif __name__ == '__main__':\n    main()\n"
    assert grade(guarded, ["3"], ["6"]) == "passed"
    assert grade("print(int(input()) * 2)\nexit()\n", ["3"], ["6"]) == "passed"
    assert grade("print('6.0   ')\n", ["3"], ["6"]) == "passed"
    assert grade("print(int(input()) * 3)\n", ["3"], ["6"]) == "wrong_answer"
    assert grade("print(6)\nprint(6)\n", ["3"], ["6"]) == "wrong_answer"
    assert grade("print(int(input()) * 2)\n", ["3", "x"], ["6", "0"]) == "runtime_error"
    assert grade("while True:\n    pass\n", ["3"], ["6"]) == "time_limit_exceeded"
    assert grade("def broken(:\n", ["3"], ["6"]) == "runtime_error"


@needs_sandbox
def test_grader_functional_problems():
    solution = (
        "class Solution:\n"
        "    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
        "        seen = {}\n"
        "        for index, value in enumerate(nums):\n"
        "            if target - value in seen:\n"
        "                return [seen[target - value], index]\n"
        "            seen[value] = index\n"
    )
    inputs, outputs = ["[2,7,11,15]\n9", "[3,2,4]\n6"], ["[0,1]", "[1,2]"]
    assert grade(solution, inputs, outputs, "twoSum") == "passed"
    as_tuple = solution.replace("[seen[target - value], index]", "(seen[target - value], index)")
    assert grade(as_tuple, inputs, outputs, "twoSum") == "passed"
    assert grade(solution.replace("index]", "0]"), inputs, outputs, "twoSum") == "wrong_answer"
    assert grade("class Solution:\n    pass\n", inputs, outputs, "twoSum") == "no_solution"


@needs_sandbox
def test_a_failure_detail_code_is_never_a_pass():
    executor = BubblewrapCodeExecutor()
    for code in (42, 50, 59, 0):
        assert executor.execute(f"raise SystemExit({code})\n", 3.0).status == "failed"
    assert executor.execute("value = 1\n", 3.0).status == "passed"


def lcb_example(example_id, fn_name=None, difficulty="easy"):
    return LiveCodeBenchExample(
        id=example_id,
        question_title=example_id,
        question_content=f"statement of {example_id}",
        starter_code="class Solution:\n    def run(self): " if fn_name else "",
        fn_name=fn_name,
        num_tests=3,
        tests=livecodebench.encode_tests(["1", "2", "3"], ["1", "2", "3"], fn_name),
        raw={"difficulty": difficulty, "platform": "atcoder", "contest_date": "2025-01-04"},
    )


class PassChatClient(TextGenerationModelClient):
    """Answers response(input, sample), sample being how often that input was
    asked before: every input is asked once per sample."""

    generation_api = "chat"

    def __init__(self, response):
        self.response = response
        self.requests = []
        self.asked = Counter()

    def answer(self, item):
        key = json.dumps(item, ensure_ascii=False)
        sample = self.asked[key]
        self.asked[key] += 1
        return GeneratedTextResult(text=self.response(item, sample), finish_reason="stop")

    def generate_conversations(self, conversations, **kwargs):
        self.requests.append((conversations, kwargs))
        return [self.answer(conversation) for conversation in conversations]

    def generate_texts(self, prompts, **kwargs):
        self.requests.append((prompts, kwargs))
        return [self.answer(prompt) for prompt in prompts]


class MarkerExecutor:
    name = "fixture"

    def __init__(self):
        self.calls = []

    def execute(self, program, timeout_seconds, attachments=None, memory_limit_mb=None):
        self.calls.append((program, timeout_seconds, attachments, memory_limit_mb))
        if "GOOD" in program:
            return CodeExecutionResult(status="passed", elapsed_ms=1, exit_code=42)
        return CodeExecutionResult(
            status="failed",
            elapsed_ms=1,
            exit_code=livecodebench.EXIT_WRONG_ANSWER,
            error="Tests failed with exit code 50.",
        )


def test_livecodebench_pass_at_1_is_the_mean_over_samples():
    def respond(conversation, sample):
        statement = conversation[-1]["content"]
        if "statement of solved" in statement:
            return "```python\nGOOD = 1\n```"
        if "statement of half" in statement:
            return "```python\nGOOD = 1\n```" if sample % 2 == 0 else "```python\nBAD = 1\n```"
        return "I cannot solve this."

    client = PassChatClient(respond)
    executor = MarkerExecutor()
    report = evaluate_livecodebench(
        [lcb_example("solved"), lcb_example("half", difficulty="hard"), lcb_example("refused")],
        model_client=client,
        code_executor=executor,
        generation_settings={"temperature": 0.2, "top_p": 0.95, "stop": ["###"]},
        num_samples=4,
        pass_at_k=[1, 2],
        aggregation="mean",
    )

    conversations, kwargs = client.requests[0]
    assert conversations[0][0]["role"] == "system"
    assert kwargs["stop"] is None
    assert kwargs["max_tokens"] == 2000
    assert len(executor.calls) == 3
    program, timeout, attachments, _ = executor.calls[0]
    assert timeout == (6 + 1) * 3 + 5
    assert set(attachments) == {livecodebench.TESTS_ATTACHMENT}
    assert "TEST_TIMEOUT = 6" in program

    metrics = report["multi_sample_metrics"]
    assert metrics["pass_at_k"]["1"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert metrics["pass_at_k"]["2"] == pytest.approx((1.0 + (1 - 1 / 6) + 0.0) / 3)
    assert report["livecodebench_metrics"]["pass_at_1"] == pytest.approx(0.5)
    assert report["livecodebench_metrics"]["per_difficulty"]["hard"]["pass_at_1"] == 0.5
    assert report["code_execution"]["sample_verdict_counts"] == {
        "no_code": 4,
        "passed": 6,
        "wrong_answer": 2,
    }
    assert report["num_failed_examples"] == 0

    headline = mean_sample_report(report, "pass@1")
    assert "accuracy" not in headline and "correct" not in headline
    assert headline["metric_name"] == "pass@1"
    assert headline["score"] == pytest.approx(0.5)
    assert headline["first_sample_accuracy"] == pytest.approx(2 / 3)
    assert headline["num_samples_total"] == 12


def test_livecodebench_base_model_uses_the_one_shot_prompt_and_the_stop_sequence():
    class CompletionClient(PassChatClient):
        generation_api = "completions"

    client = CompletionClient(lambda prompt, sample: "GOOD = 1\n")
    demonstrations = {
        "stdin": {"question": "Q0", "answer": "print(0)"},
        "functional": {"question": "Q1", "sample_code": "class Solution: ", "answer": "pass"},
    }
    report = evaluate_livecodebench(
        [lcb_example("stdin-problem"), lcb_example("call-problem", fn_name="run")],
        model_client=client,
        code_executor=MarkerExecutor(),
        generation_settings={"temperature": 0.2, "stop": ["###"]},
        base_model_demonstrations=demonstrations,
    )

    prompts, kwargs = client.requests[0]
    assert kwargs["stop"] == ["###"]
    assert prompts[0] == (
        "### Question\nQ0\n\n### Answer\n\nprint(0)\n\n"
        "### Question\nstatement of stdin-problem\n\n### Answer\n\n"
    )
    assert prompts[1] == (
        "### Question\nQ1\n\n### Starter Code\nclass Solution: \n\n### Answer\n\npass\n\n"
        "### Question\nstatement of call-problem\n\n"
        "### Starter Code\nclass Solution:\n    def run(self): \n\n### Answer\n\n"
    )
    assert report["correct"] == 2


def test_humaneval_chat_instruction_is_english_for_the_english_task():
    example = HumanEvalExample(
        id="HumanEval/0",
        prompt="def add_one(value):\n",
        test="def check(candidate):\n    assert candidate(1) == 2\n",
        entry_point="add_one",
        canonical_solution="    return value + 1\n",
        raw={},
    )

    class Executor:
        name = "fixture"

        def execute(self, program, timeout_seconds):
            return CodeExecutionResult(status="passed", elapsed_ms=1, exit_code=42)

    def run(chat_prompt):
        client = PassChatClient(lambda prompt, sample: "def add_one(value):\n    return value + 1")
        report = evaluate_humaneval(
            [example],
            model_client=client,
            code_executor=Executor(),
            chat_prompt=chat_prompt,
        )
        return client.requests[0][0][0], report

    english, report = run("simple_evals_humaneval_v1")
    assert english == (
        "Read the following function signature and docstring, and fully implement the "
        "function described. Your response should only contain the code for this "
        "function.\ndef add_one(value):\n"
    )
    assert report["code_execution"]["prompt_mode"] == "chat:simple_evals_humaneval_v1"
    assert report["results"][0]["generation_prompt_text"] == english
    turkish, _ = run("none")
    assert turkish.startswith("Aşağıdaki Python fonksiyonunu tamamla.")


def test_executor_service_accepts_attachments_and_rejects_bad_requests():
    class Recorder:
        def __init__(self):
            self.requests = []

        def execute(self, program, timeout_seconds, attachments=None, memory_limit_mb=None):
            self.requests.append((program, timeout_seconds, attachments, memory_limit_mb))
            return CodeExecutionResult(status="passed", elapsed_ms=1, exit_code=42)

    def roundtrip(tmp_path, recorder, **request):
        path = str(tmp_path / f"executor-{len(recorder.requests)}.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(1)

        def serve():
            connection, _ = server.accept()
            with connection:
                _handle_executor_connection(connection, recorder, memory_ceiling_mb=1024)

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            return UnixSocketCodeExecutor(path).execute(**request)
        finally:
            thread.join(timeout=10)
            server.close()

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        recorder = Recorder()
        payload = b"x" * 3_000_000
        result = roundtrip(
            Path(directory),
            recorder,
            program="value = 1\n",
            timeout_seconds=300.0,
            attachments={"tests.bin": payload},
            memory_limit_mb=4096,
        )
        assert result.status == "passed"
        program, timeout, attachments, memory_limit_mb = recorder.requests[0]
        assert (program, timeout) == ("value = 1\n", 300.0)
        assert attachments == {"tests.bin": payload}
        assert memory_limit_mb == 1024

        too_long = roundtrip(
            Path(directory), recorder, program="value = 1\n", timeout_seconds=601.0
        )
        assert too_long.status == "executor_error"
        assert len(recorder.requests) == 1

    with pytest.raises(ValueError):
        validate_attachments({"../escape": b""})
    with pytest.raises(ValueError):
        validate_attachments({"runner.py": b""})




def test_streamed_chat_completion_is_collected_into_one_response():
    from types import SimpleNamespace as N

    from bilgeval.evaluators import OpenAICompatibleTextGenerationClient

    def chunk(content=None, reasoning=None, finish=None, usage=None, choices=True):
        delta = N(content=content, reasoning_content=reasoning)
        return N(
            id="chatcmpl-1",
            usage=usage,
            choices=[N(index=0, delta=delta, finish_reason=finish)] if choices else [],
        )

    collected = OpenAICompatibleTextGenerationClient._collect_chat_stream(
        [
            chunk(reasoning="think "),
            chunk(reasoning="more"),
            chunk(content="```python\n"),
            chunk(content="print(1)\n```"),
            chunk(finish="stop"),
            chunk(usage={"completion_tokens": 12}, choices=False),
        ]
    )
    message = collected["choices"][0]["message"]
    assert message["content"] == "```python\nprint(1)\n```"
    assert message["reasoning_content"] == "think more"
    assert collected["choices"][0]["finish_reason"] == "stop"
    assert collected["usage"] == {"completion_tokens": 12}

    with pytest.raises(ValueError, match="without a finish_reason"):
        OpenAICompatibleTextGenerationClient._collect_chat_stream([chunk(content="cut off")])


def test_a_fully_failed_batch_is_resent_before_the_run_aborts(monkeypatch):
    from bilgeval.evaluators import BatchRequestError, OpenAICompatibleTextGenerationClient

    monkeypatch.setenv("BILGEVAL_BATCH_RETRY_DELAYS", "0,0")
    client = OpenAICompatibleTextGenerationClient(
        base_url="http://candidate.invalid/v1", model="m", generation_api="chat"
    )
    attempts = []

    def flaky(prompt, *args):
        attempts.append(prompt)
        if len(attempts) <= 2:
            return GeneratedTextResult(error="504 Gateway Time-out")
        return GeneratedTextResult(text="ok", finish_reason="stop")

    monkeypatch.setattr(client, "generate_chat_text", flaky)
    results = client.generate_texts(["a", "b"], max_tokens=8)
    assert [result.text for result in results] == ["ok", "ok"]

    always_down = OpenAICompatibleTextGenerationClient(
        base_url="http://candidate.invalid/v1", model="m", generation_api="chat"
    )
    monkeypatch.setattr(
        always_down, "generate_chat_text", lambda *args: GeneratedTextResult(error="down")
    )
    with pytest.raises(BatchRequestError, match="all 2 requests"):
        always_down.generate_texts(["a", "b"], max_tokens=8)


def test_turkish_ifeval_ignores_a_leaked_reasoning_block():
    from bilgeval.evaluators import evaluate_ifeval
    from bilgeval.reader import IFEvalExample

    example = IFEvalExample(
        id="p1",
        prompt="Yanıtında virgül kullanma.",
        instruction_ids=[26],
        verifier_kwargs={"26": {}},
        raw={},
    )
    client = PassChatClient(lambda prompt, sample: "<think>plan, with, commas</think>\nVirgülsüz yanıt")
    report = evaluate_ifeval([example], model_client=client)
    assert report["ifeval_metrics"]["num_reasoning_blocks_stripped"] == 1
    assert report["results"][0]["candidate_response"] == "Virgülsüz yanıt"
