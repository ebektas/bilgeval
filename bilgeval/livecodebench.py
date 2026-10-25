"""LiveCodeBench code generation: prompts, code extraction and the grader.

Port of the official runner (LiveCodeBench/LiveCodeBench @ 28fef95):
`lcb_runner/prompts/code_generation.py`, `lcb_runner/utils/extraction_utils.py`
and `lcb_runner/evaluation/testing_util.py`. The grader runs inside the
bilgeval sandbox as one self-contained program per candidate; the test cases
travel next to it as the attachment `tests.bin`.
"""

import base64
import io
import json
import pickle
import zlib
from typing import Any


UPSTREAM_REVISION = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
SCHEMA_VERSION = "livecodebench.codegen/v1"
TESTS_ATTACHMENT = "tests.bin"

OFFICIAL_NUM_SAMPLES = 10
OFFICIAL_TEMPERATURE = 0.2
OFFICIAL_TOP_P = 0.95
OFFICIAL_MAX_TOKENS = 2000
OFFICIAL_TEST_TIMEOUT_SECONDS = 6

DEFAULT_MEMORY_LIMIT_MB = 4096

EXIT_WRONG_ANSWER = 50
EXIT_TIME_LIMIT = 51
EXIT_RUNTIME_ERROR = 52
EXIT_NO_SOLUTION = 53
EXIT_HARNESS_ERROR = 54
VERDICT_BY_EXIT_CODE = {
    EXIT_WRONG_ANSWER: "wrong_answer",
    EXIT_TIME_LIMIT: "time_limit_exceeded",
    EXIT_RUNTIME_ERROR: "runtime_error",
    EXIT_NO_SOLUTION: "no_solution",
    EXIT_HARNESS_ERROR: "harness_error",
}

SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)
FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads the "
    "inputs, runs the algorithm and writes output to STDOUT."
)


def user_prompt(question_content: str, starter_code: str) -> str:
    """`get_generic_question_template_answer`."""
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def chat_messages(question_content: str, starter_code: str) -> list[dict[str, str]]:
    """The `OpenAIChat` / `LLaMa3` form: generic system message + question."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": user_prompt(question_content, starter_code)},
    ]


def base_model_prompt(
    question_content: str,
    starter_code: str,
    demonstration: dict[str, str],
) -> str:
    """`get_base_model_question_template_answer` (`GenericBase`): one solved
    example of the same kind, then the question."""

    def block(question: str, sample_code: str, answer: str) -> str:
        text = "### Question\n" + question + "\n\n"
        if starter_code:
            text += "### Starter Code\n" + sample_code + "\n\n"
        text += "### Answer\n\n" + answer
        if answer:
            text += "\n\n"
        return text

    return block(
        demonstration["question"],
        demonstration.get("sample_code", ""),
        demonstration["answer"],
    ) + block(question_content, starter_code, "")


def extract_code(model_output: str, base_model: bool = False) -> str:
    """`extract_code`: the LAST fenced block; a base model's output as is."""
    if base_model:
        return model_output.strip()
    lines = model_output.split("\n")
    fences = [index for index, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1 : fences[-1]])


class _StringOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from a dataset.")


def decode_upstream_private_tests(value: str) -> list[dict[str, Any]]:
    """`private_test_cases` of the upstream dataset: JSON, or
    base64(zlib(pickle(JSON string))). The pickle may only hold a string."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    data = zlib.decompress(base64.b64decode(value.encode("utf-8")))
    decoded = _StringOnlyUnpickler(io.BytesIO(data)).load()
    if not isinstance(decoded, str):
        raise ValueError("LiveCodeBench private tests did not decode to a JSON string.")
    return json.loads(decoded)


def encode_tests(inputs: list[str], outputs: list[str], fn_name: str | None) -> str:
    """Test cases as stored in the bilgeval dataset: base64(zlib(JSON))."""
    if len(inputs) != len(outputs) or not inputs:
        raise ValueError("LiveCodeBench needs one expected output per test input.")
    payload = json.dumps(
        {"inputs": inputs, "outputs": outputs, "fn_name": fn_name},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return base64.b64encode(zlib.compress(payload.encode("utf-8"), 9)).decode("ascii")


def tests_attachment(encoded_tests: str) -> bytes:
    return base64.b64decode(encoded_tests.encode("ascii"))


def decode_tests(encoded_tests: str) -> dict[str, Any]:
    return json.loads(zlib.decompress(tests_attachment(encoded_tests)).decode("utf-8"))


def execution_timeout_seconds(num_tests: int, test_timeout_seconds: int) -> float:
    """The upstream global timeout of one candidate."""
    return float((test_timeout_seconds + 1) * num_tests + 5)


def build_program(code: str, test_timeout_seconds: int = OFFICIAL_TEST_TIMEOUT_SECONDS) -> str:
    if not isinstance(test_timeout_seconds, int) or test_timeout_seconds < 1:
        raise ValueError("The LiveCodeBench per-test timeout must be a positive integer.")
    return (
        f"CANDIDATE = {json.dumps(code)}\n"
        f"TEST_TIMEOUT = {test_timeout_seconds}\n"
        f"TESTS_ATTACHMENT = {json.dumps(TESTS_ATTACHMENT)}\n"
        + HARNESS_SOURCE
    )


HARNESS_SOURCE = r'''
import ast
import faulthandler
import json
import os
import signal
import sys
import zlib
from decimal import Decimal
from io import StringIO
from types import ModuleType
from unittest.mock import patch, mock_open

sys.set_int_max_str_digits(50000)

EXIT_WRONG_ANSWER = 50
EXIT_TIME_LIMIT = 51
EXIT_RUNTIME_ERROR = 52
EXIT_NO_SOLUTION = 53
EXIT_HARNESS_ERROR = 54

import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockBuffer:
    def __init__(self, inputs):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


class MockStdinWithBuffer:
    def __init__(self, inputs):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


def clean_if_name(code):
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
    except:
        pass
    return code


def make_function(code):
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)
        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        return import_string + "\n" + ast.unparse(import_stmts) + "\n" + ast.unparse(function_ast)
    except Exception:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    inputs_line_iterator = iter(inputs.split("\n"))
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass

    return _inner_call_method(method)


def compile_code(code, timeout):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
    finally:
        signal.alarm(0)
    return compiled_sol


def convert_line_to_decimals(line):
    try:
        return True, [Decimal(elem) for elem in line.split()]
    except:
        return False, []


def get_stripped_lines(val):
    val = val.strip()
    return [val_line.strip() for val_line in val.split("\n")]


def failure_code(error):
    if "timeoutexception" in repr(error).lower():
        return EXIT_TIME_LIMIT
    return EXIT_RUNTIME_ERROR


def grade_call_based(code, all_inputs, all_outputs, fn_name, timeout):
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    method = getattr(compiled_sol, fn_name, None)
    if method is None:
        return EXIT_NO_SOLUTION
    all_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
    all_outputs = [json.loads(output) for output in all_outputs]
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            prediction = method(*gt_inp)
            signal.alarm(0)
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            if not prediction == gt_out:
                return EXIT_WRONG_ANSWER
        except Exception as error:
            signal.alarm(0)
            return failure_code(error)
        finally:
            signal.alarm(0)
            faulthandler.disable()
    return 0


def grade_stdio(code, all_inputs, all_outputs, timeout):
    code = clean_if_name(code)
    code = make_function(code)
    compiled_sol = compile_code(code, timeout)
    method = getattr(compiled_sol, "wrapped_function", None)
    if method is None:
        return EXIT_NO_SOLUTION
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        signal.alarm(timeout)
        with Capturing() as captured_output:
            try:
                call_method(method, gt_inp)
                signal.alarm(0)
            except Exception as error:
                signal.alarm(0)
                return failure_code(error)
            finally:
                signal.alarm(0)
                faulthandler.disable()
        prediction = captured_output[0]
        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)
        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            return EXIT_WRONG_ANSWER
        for stripped_prediction_line, stripped_gt_out_line in zip(
            stripped_prediction_lines, stripped_gt_out_lines
        ):
            if stripped_prediction_line == stripped_gt_out_line:
                continue
            success, decimal_prediction_line = convert_line_to_decimals(stripped_prediction_line)
            if not success:
                return EXIT_WRONG_ANSWER
            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                return EXIT_WRONG_ANSWER
            if decimal_prediction_line == decimal_gtout_line:
                continue
            return EXIT_WRONG_ANSWER
    return 0


def reliability_guard():
    faulthandler.disable()
    import builtins

    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid",
        "fork", "forkpty", "killpg", "rename", "renames", "truncate", "replace", "unlink",
        "fchmod", "fchown", "chmod", "chown", "chroot", "lchflags", "lchmod", "lchown",
        "getcwd", "chdir",
    ):
        setattr(os, name, None)
    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    import subprocess

    subprocess.Popen = None
    builtins.help = None
    for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[name] = None


def run_tests():
    directory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(directory, TESTS_ATTACHMENT), "rb") as file:
        in_outs = json.loads(zlib.decompress(file.read()).decode("utf-8"))
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard()
    fn_name = in_outs.get("fn_name")
    signal.alarm(TEST_TIMEOUT)
    try:
        if fn_name is None:
            return grade_stdio(CANDIDATE, in_outs["inputs"], in_outs["outputs"], TEST_TIMEOUT)
        return grade_call_based(
            CANDIDATE, in_outs["inputs"], in_outs["outputs"], fn_name, TEST_TIMEOUT
        )
    except Exception:
        return EXIT_RUNTIME_ERROR
    finally:
        signal.alarm(0)


try:
    verdict = run_tests()
except SystemExit:
    verdict = EXIT_RUNTIME_ERROR
except BaseException:
    verdict = EXIT_HARNESS_ERROR
if verdict:
    raise SystemExit(verdict)
'''
