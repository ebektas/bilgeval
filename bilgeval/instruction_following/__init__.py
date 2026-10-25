"""Official instruction-following verifiers: Google IFEval and AllenAI IFBench.

The checker code in `google_ifeval/` and `ifbench/` is upstream's (NOTICE.md);
this module is the evaluation loop of their `evaluation_lib.py`: build each
instruction with the row's arguments, then check the response as it is
(strict) and in eight cleaned-up forms (loose).
"""

import importlib
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


PROTOCOLS = {
    "google.ifeval/v1": "google_ifeval",
    "allenai.ifbench/v1": "ifbench",
}
SOURCES = {
    "google_ifeval": "google-research/google-research instruction_following_eval @ e6890f8",
    "ifbench": "allenai/IFBench @ 1c40f0c",
}
REQUIRED_PACKAGES = ("nltk", "langdetect", "immutabledict", "emoji", "syllapy")
NLTK_DATA = Path(__file__).resolve().parent / "nltk_data"
_THINKING = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)


class MissingVerifierDependency(RuntimeError):
    pass


@lru_cache(maxsize=None)
def _registry(package: str) -> dict[str, Any]:
    missing = []
    for name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise MissingVerifierDependency(
            "The English IFEval / IFBench verifiers need "
            + ", ".join(missing)
            + ". Install them with: pip install 'bilgeval[ifeval]'"
        )
    import langdetect
    import nltk

    if str(NLTK_DATA) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA))
    langdetect.DetectorFactory.seed = 0
    module = importlib.import_module(f"{__name__}.{package}.instructions_registry")
    return module.INSTRUCTION_DICT


def verifier_info(package: str) -> dict[str, Any]:
    from importlib import metadata

    _registry(package)
    return {
        "source": SOURCES[package],
        "num_instruction_types": len(_registry(package)),
        "packages": {name: metadata.version(name) for name in REQUIRED_PACKAGES},
        "langdetect_seed": 0,
    }


def strip_reasoning(response: str) -> tuple[str, bool]:
    """The answer without a leading reasoning block.

    IFBench: "we process the output to extract the answer without the
    reasoning chains". A block that never closes means the answer was never
    written.
    """
    match = _THINKING.match(response)
    if match:
        return response[match.end() :].lstrip(), True
    if response.lstrip().startswith("<think>"):
        return "", True
    return response, False


def loose_variants(response: str) -> list[str]:
    """`test_instruction_following_loose`: the response, without its first
    line, its last line or both, each with and without asterisks."""
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def verify(
    package: str,
    prompt: str,
    instruction_ids: list[str],
    kwargs: list[dict[str, Any]],
    response: str,
) -> tuple[list[bool], list[bool]]:
    """(strict, loose): one verdict per instruction of the prompt."""
    registry = _registry(package)
    if len(instruction_ids) != len(kwargs):
        raise ValueError("Each instruction needs its own kwargs entry.")
    variants = loose_variants(response)
    strict, loose = [], []
    for instruction_id, arguments in zip(instruction_ids, kwargs):
        instruction = registry[instruction_id](instruction_id)
        random_state = random.getstate()
        random.seed(f"bilgeval:{instruction_id}")
        try:
            instruction.build_description(
                **{key: value for key, value in (arguments or {}).items() if value is not None}
            )
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)
        finally:
            random.setstate(random_state)
        strict.append(bool(response.strip() and instruction.check_following(response)))
        loose.append(
            any(bool(variant.strip() and instruction.check_following(variant)) for variant in variants)
        )
    return strict, loose
