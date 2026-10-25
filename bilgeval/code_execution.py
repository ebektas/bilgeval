import base64
import json
import os
import sys
import re
import resource
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


MAX_PROGRAM_BYTES = 1_000_000
MAX_PROTOCOL_BYTES = 2_000_000
MAX_ATTACHMENT_BYTES = 48_000_000
MAX_REQUEST_BYTES = 80_000_000
MAX_EXECUTION_TIMEOUT_SECONDS = 600.0
DEFAULT_MEMORY_LIMIT_MB = 512
MAX_MEMORY_LIMIT_MB = 8192
PASSED_EXIT_CODE = 42
FAILURE_DETAIL_EXIT_CODES = range(50, 60)
_ATTACHMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RUNNER_EPILOGUE = (
    "except SystemExit as exit_request:\n"
    "    code = exit_request.code\n"
    "    raise SystemExit(code if isinstance(code, int) and 50 <= code <= 59 else 1)\n"
    "except BaseException:\n"
    "    raise SystemExit(1)\n"
    f"raise SystemExit({PASSED_EXIT_CODE})\n"
)


def validate_attachments(attachments: dict[str, bytes] | None) -> dict[str, bytes]:
    attachments = attachments or {}
    for name, content in attachments.items():
        if not _ATTACHMENT_NAME.match(name) or name in {"candidate.py", "runner.py"}:
            raise ValueError(f"Invalid attachment name: {name!r}.")
        if not isinstance(content, (bytes, bytearray)):
            raise ValueError(f"Attachment {name!r} must be bytes.")
    if sum(len(content) for content in attachments.values()) > MAX_ATTACHMENT_BYTES:
        raise ValueError("Attachments exceed the 48 MB execution limit.")
    return attachments


def validate_memory_limit(memory_limit_mb: int | None) -> int:
    value = DEFAULT_MEMORY_LIMIT_MB if memory_limit_mb is None else int(memory_limit_mb)
    if not 64 <= value <= MAX_MEMORY_LIMIT_MB:
        raise ValueError(f"Memory limit must be between 64 and {MAX_MEMORY_LIMIT_MB} MB.")
    return value


@dataclass(frozen=True)
class CodeExecutionResult:
    status: str
    elapsed_ms: int
    error: str | None = None
    exit_code: int | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"


class CodeExecutor(Protocol):
    name: str

    def execute(
        self,
        program: str,
        timeout_seconds: float,
        attachments: dict[str, bytes] | None = None,
        memory_limit_mb: int | None = None,
    ) -> CodeExecutionResult:
        """Run `program`; attachments are read-only files next to it."""
        ...


def extract_python_code(candidate_response: str) -> str:
    value = re.sub(
        r"<think>.*?</think>",
        "",
        candidate_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    fenced = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        value = max(fenced, key=len)
    lines = value.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def build_humaneval_program(
    prompt: str,
    candidate_response: str,
    test: str,
    entry_point: str,
) -> tuple[str, str]:
    completion = extract_python_code(candidate_response)
    if not completion:
        raise ValueError("The model response did not contain Python code.")

    definition_pattern = rf"(?m)^(?:async\s+)?def\s+{re.escape(entry_point)}\s*\("
    full_definition = re.search(definition_pattern, completion)
    if full_definition:
        prompt_definition = re.search(definition_pattern, prompt)
        prefix = prompt[: prompt_definition.start()] if prompt_definition else prompt
        solution = prefix + completion
    else:
        solution = prompt + completion
    program = f"{solution.rstrip()}\n{test.rstrip()}\n\ncheck({entry_point})\n"
    if len(program.encode("utf-8")) > MAX_PROGRAM_BYTES:
        raise ValueError("The generated HumanEval program exceeds the 1 MB execution limit.")
    return program, completion


def _limit_candidate_process(
    timeout_seconds: float,
    run_as_uid: int | None = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> None:
    cpu_seconds = max(1, int(timeout_seconds) + 1)
    memory_bytes = memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 * 1024 * 1024, 1 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.setsid()
    if run_as_uid is not None:
        os.setgroups([])
        os.setgid(run_as_uid)
        os.setuid(run_as_uid)


def _run_command(
    command: list[str],
    timeout_seconds: float,
    *,
    preexec_fn=None,
    environment: dict[str, str] | None = None,
    success_exit_code: int = 0,
) -> CodeExecutionResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
            close_fds=True,
            preexec_fn=preexec_fn,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return CodeExecutionResult(
            status="timed_out",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            error=f"Candidate exceeded the {timeout_seconds:g}s execution limit.",
        )
    except OSError as exc:
        return CodeExecutionResult(
            status="executor_error",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            error=f"Could not start the code sandbox: {exc}",
        )

    elapsed_ms = int((time.monotonic() - started) * 1000)
    if completed.returncode == success_exit_code:
        return CodeExecutionResult(
            status="passed",
            elapsed_ms=elapsed_ms,
            exit_code=completed.returncode,
        )
    return CodeExecutionResult(
        status="failed",
        elapsed_ms=elapsed_ms,
        error=f"Tests failed with exit code {completed.returncode}.",
        exit_code=completed.returncode,
    )


class BubblewrapCodeExecutor:
    name = "bubblewrap"

    def __init__(self, executable: str = "bwrap"):
        resolved = shutil.which(executable)
        if resolved is None:
            raise RuntimeError(
                "HumanEval requires Bubblewrap. Install 'bwrap', or configure the "
                "networkless Unix-socket executor."
            )
        self.executable = resolved

    def execute(
        self,
        program: str,
        timeout_seconds: float,
        attachments: dict[str, bytes] | None = None,
        memory_limit_mb: int | None = None,
    ) -> CodeExecutionResult:
        attachments = validate_attachments(attachments)
        memory_bytes = validate_memory_limit(memory_limit_mb) * 1024 * 1024
        with tempfile.TemporaryDirectory(prefix="bilgeval-sandbox-") as directory:
            workspace = Path(directory)
            candidate = workspace / "candidate.py"
            candidate.write_text(program, encoding="utf-8")
            for name, content in attachments.items():
                (workspace / name).write_bytes(content)
            runner = workspace / "runner.py"
            cpu_seconds = max(1, int(timeout_seconds) + 1)
            runner.write_text(
                "import resource, runpy\n"
                f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
                f"resource.setrlimit(resource.RLIMIT_AS, ({memory_bytes}, {memory_bytes}))\n"
                "resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))\n"
                "resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))\n"
                "resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))\n"
                "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
                "try:\n"
                "    runpy.run_path('/workspace/candidate.py', run_name='__main__')\n"
                + _RUNNER_EPILOGUE,
                encoding="utf-8",
            )
            interpreter = os.environ.get("BILGEVAL_SANDBOX_PYTHON") or os.path.realpath(sys.executable)
            prefix = os.path.realpath(sys.base_prefix)
            extra_binds: list[str] = []
            for root in {prefix, os.path.dirname(os.path.dirname(interpreter))}:
                if root not in ("/", "/usr") and not root.startswith("/usr/"):
                    extra_binds += ["--ro-bind", root, root]
            command = [
                self.executable,
                "--unshare-all",
                "--die-with-parent",
                "--new-session",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind",
                "/lib64",
                "/lib64",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--ro-bind",
                str(workspace),
                "/workspace",
                *extra_binds,
                "--chdir",
                "/workspace",
                interpreter,
                "-I",
                "-S",
                "/workspace/runner.py",
            ]
            return _run_command(
                command,
                timeout_seconds,
                environment={
                    "HOME": "/tmp",
                    "LANG": "C.UTF-8",
                    "PATH": "/usr/bin",
                    "PYTHONHASHSEED": "0",
                },
                success_exit_code=PASSED_EXIT_CODE,
            )


class DirectIsolatedContainerExecutor:
    name = "networkless-container"

    def __init__(self, run_as_uid: int = 65534):
        if os.getenv("BILGEVAL_EXECUTOR_ISOLATED") != "1":
            raise RuntimeError(
                "Direct execution is restricted to the dedicated networkless executor container."
            )
        self.run_as_uid = run_as_uid

    def execute(
        self,
        program: str,
        timeout_seconds: float,
        attachments: dict[str, bytes] | None = None,
        memory_limit_mb: int | None = None,
    ) -> CodeExecutionResult:
        attachments = validate_attachments(attachments)
        memory_limit = validate_memory_limit(memory_limit_mb)
        with tempfile.TemporaryDirectory(prefix="bilgeval-sandbox-") as directory:
            workspace = Path(directory)
            os.chmod(workspace, 0o711)
            candidate = workspace / "candidate.py"
            candidate.write_text(program, encoding="utf-8")
            os.chmod(candidate, 0o444)
            for name, content in attachments.items():
                attachment = workspace / name
                attachment.write_bytes(content)
                os.chmod(attachment, 0o444)
            runner = workspace / "runner.py"
            runner.write_text(
                "import runpy\n"
                "try:\n"
                f"    runpy.run_path({str(candidate)!r}, run_name='__main__')\n"
                + _RUNNER_EPILOGUE,
                encoding="utf-8",
            )
            os.chmod(runner, 0o444)
            return _run_command(
                ["/usr/local/bin/python3", "-I", "-S", str(runner)],
                timeout_seconds,
                preexec_fn=lambda: _limit_candidate_process(
                    timeout_seconds,
                    run_as_uid=self.run_as_uid,
                    memory_limit_mb=memory_limit,
                ),
                environment={
                    "HOME": "/tmp",
                    "LANG": "C.UTF-8",
                    "PATH": "/usr/local/bin:/usr/bin",
                    "PYTHONHASHSEED": "0",
                },
                success_exit_code=PASSED_EXIT_CODE,
            )


def _receive_message(connection: socket.socket, limit: int = MAX_PROTOCOL_BYTES) -> dict:
    header = _receive_exact(connection, 4)
    length = struct.unpack("!I", header)[0]
    if length < 1 or length > limit:
        raise ValueError("Invalid code-executor message size.")
    return json.loads(_receive_exact(connection, length).decode("utf-8"))


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("Code-executor socket closed before the message completed.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(
    connection: socket.socket,
    value: dict,
    limit: int = MAX_PROTOCOL_BYTES,
) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    if len(payload) > limit:
        raise ValueError("Code-executor message exceeds the protocol limit.")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


class UnixSocketCodeExecutor:
    name = "unix-socket"

    def __init__(self, socket_path: str, max_parallel: int | None = None):
        self.socket_path = socket_path
        self.max_parallel = max_parallel or int(
            os.environ.get("BILGEVAL_CODE_EXECUTOR_PARALLEL", "2")
        )
        if self.max_parallel < 1:
            raise ValueError("The code executor needs at least one parallel slot.")
        self._slots = threading.BoundedSemaphore(self.max_parallel)

    def execute(
        self,
        program: str,
        timeout_seconds: float,
        attachments: dict[str, bytes] | None = None,
        memory_limit_mb: int | None = None,
    ) -> CodeExecutionResult:
        started = time.monotonic()
        try:
            request: dict = {"program": program, "timeout_seconds": timeout_seconds}
            if attachments:
                request["attachments"] = {
                    name: base64.b64encode(content).decode("ascii")
                    for name, content in validate_attachments(attachments).items()
                }
            if memory_limit_mb is not None:
                request["memory_limit_mb"] = validate_memory_limit(memory_limit_mb)
            with self._slots:
                started = time.monotonic()
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.settimeout(timeout_seconds + 30)
                    connection.connect(self.socket_path)
                    _send_message(connection, request, limit=MAX_REQUEST_BYTES)
                    response = _receive_message(connection)
            return CodeExecutionResult(**response)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return CodeExecutionResult(
                status="executor_error",
                elapsed_ms=int((time.monotonic() - started) * 1000),
                error=f"Code-executor service failed: {exc}",
            )


def create_code_executor(
    mode: str = "auto",
    socket_path: str | None = None,
) -> CodeExecutor:
    if mode == "auto":
        mode = "unix" if socket_path else "bubblewrap"
    if mode == "unix":
        if not socket_path:
            raise ValueError("--code-executor-socket is required for the Unix executor.")
        return UnixSocketCodeExecutor(socket_path)
    if mode == "bubblewrap":
        return BubblewrapCodeExecutor()
    raise ValueError(f"Unsupported code executor: {mode}")


def _handle_executor_connection(connection: socket.socket, executor, memory_ceiling_mb: int) -> None:
    try:
        request = _receive_message(connection, limit=MAX_REQUEST_BYTES)
        program = request.get("program")
        timeout_seconds = float(request.get("timeout_seconds", 3.0))
        if not isinstance(program, str):
            raise ValueError("Code-executor request is missing program text.")
        if len(program.encode("utf-8")) > MAX_PROGRAM_BYTES:
            raise ValueError("Program exceeds the 1 MB execution limit.")
        if not 0.1 <= timeout_seconds <= MAX_EXECUTION_TIMEOUT_SECONDS:
            raise ValueError(
                "Execution timeout must be between 0.1 and "
                f"{MAX_EXECUTION_TIMEOUT_SECONDS:g} seconds."
            )
        raw_attachments = request.get("attachments") or {}
        if not isinstance(raw_attachments, dict):
            raise ValueError("Code-executor attachments must be an object.")
        attachments = {
            str(name): base64.b64decode(str(content), validate=True)
            for name, content in raw_attachments.items()
        }
        memory_limit_mb = min(
            validate_memory_limit(request.get("memory_limit_mb")),
            memory_ceiling_mb,
        )
        result = executor.execute(
            program,
            timeout_seconds,
            attachments=attachments,
            memory_limit_mb=memory_limit_mb,
        )
        _send_message(connection, asdict(result))
    except Exception as exc:
        _send_message(
            connection,
            asdict(
                CodeExecutionResult(
                    status="executor_error",
                    elapsed_ms=0,
                    error=f"Executor request failed: {exc}",
                )
            ),
        )


def serve_unix_executor(socket_path: str, workers: int | None = None) -> None:
    """Serve execution requests, each in its own forked worker process.

    The parent stays single-threaded (fork and preexec_fn are only safe that
    way) and never runs more than `workers` requests at once.
    """
    executor = DirectIsolatedContainerExecutor()
    workers = workers or int(os.environ.get("BILGEVAL_EXECUTOR_WORKERS", "2"))
    if workers < 1:
        raise ValueError("The code executor needs at least one worker.")
    memory_ceiling_mb = validate_memory_limit(
        int(os.environ.get("BILGEVAL_EXECUTOR_MAX_MEMORY_MB", "1024"))
    )
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    active: set[int] = set()

    def reap(block: bool) -> None:
        while active:
            try:
                pid, _ = os.waitpid(-1, 0 if block else os.WNOHANG)
            except ChildProcessError:
                active.clear()
                return
            if pid == 0:
                return
            active.discard(pid)
            block = False

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(socket_path)
        os.chmod(socket_path, 0o666)
        server.listen(32)
        server.settimeout(5.0)
        while True:
            try:
                connection, _ = server.accept()
            except socket.timeout:
                reap(block=False)
                continue
            connection.settimeout(None)
            reap(block=False)
            while len(active) >= workers:
                reap(block=True)
            pid = os.fork()
            if pid == 0:
                status = 0
                try:
                    server.close()
                    with connection:
                        _handle_executor_connection(connection, executor, memory_ceiling_mb)
                except BaseException:
                    status = 1
                finally:
                    os._exit(status)
            active.add(pid)
            connection.close()
