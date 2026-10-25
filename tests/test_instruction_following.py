"""English IFEval and IFBench: the vendored upstream checkers must give the
verdicts of the unmodified upstream evaluation on published responses."""

import json
from pathlib import Path

import pytest

pytest.importorskip("nltk", reason="pip install 'bilgeval[ifeval]'")
pytest.importorskip("langdetect", reason="pip install 'bilgeval[ifeval]'")

from bilgeval import instruction_following
from bilgeval import main as main_module
from bilgeval.evaluators import (
    GeneratedTextResult,
    TextGenerationModelClient,
    evaluate_official_ifeval,
)
from bilgeval.reader import OfficialIFEvalExample, read_ifeval_dataset
from bilgeval.registry import DATASETS
from bilgeval.registry_bundle import load_registry_bundle


FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "instruction_following_upstream.json").read_text("utf-8")
)


class EchoClient(TextGenerationModelClient):
    generation_api = "chat"

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def generate_texts(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        results = []
        for prompt in prompts:
            value = self.responses[prompt]
            results.append(
                value
                if isinstance(value, GeneratedTextResult)
                else GeneratedTextResult(text=value, finish_reason="stop")
            )
        return results




def test_bundled_nltk_resources_are_enough(monkeypatch):
    import nltk

    instruction_following.verifier_info("ifbench")
    monkeypatch.setattr(nltk.data, "path", [str(instruction_following.NLTK_DATA)])
    assert nltk.sent_tokenize("One sentence. Another one.") == ["One sentence.", "Another one."]
    assert nltk.pos_tag(nltk.word_tokenize("Run the tests"))[0][1].startswith("VB")
    assert "the" in nltk.corpus.stopwords.words("english")


def test_loose_variants_are_the_upstream_eight():
    assert instruction_following.loose_variants("*a*\nb\n*c*") == [
        "*a*\nb\n*c*",
        "a\nb\nc",
        "b\n*c*",
        "*a*\nb",
        "b",
        "b\nc",
        "a\nb",
        "b",
    ]




def test_reasoning_block_is_not_part_of_the_answer():
    assert instruction_following.strip_reasoning("<think>plan, with commas</think>\n\nAnswer") == (
        "Answer",
        True,
    )
    assert instruction_following.strip_reasoning("<think>never finished") == ("", True)
    assert instruction_following.strip_reasoning("No tags here") == ("No tags here", False)


def example(example_id, instruction_ids, kwargs):
    return OfficialIFEvalExample(
        id=example_id,
        prompt=f"prompt {example_id}",
        instruction_ids=instruction_ids,
        kwargs=kwargs,
        raw={},
    )


def test_headline_and_the_four_upstream_accuracies():
    examples = [
        example("no-comma", ["punctuation:no_comma"], [{}]),
        example("ends", ["startend:end_checker"], [{"end_phrase": "Is there anything else?"}]),
        example(
            "two",
            ["punctuation:no_comma", "keywords:existence"],
            [{}, {"keywords": ["ankara"]}],
        ),
        example("lost", ["punctuation:no_comma", "punctuation:no_comma"], [{}, {}]),
    ]
    client = EchoClient(
        {
            "prompt no-comma": "<think>a plan, with commas</think>A reply without that mark.",
            "prompt ends": "Here you go.\n**Is there anything else?**",
            "prompt two": "No comma here but the keyword is missing.",
            "prompt lost": GeneratedTextResult(error="gateway timeout"),
        }
    )

    def run(headline):
        return evaluate_official_ifeval(
            examples,
            model_client=client,
            package="google_ifeval",
            headline=headline,
            generation_settings={"temperature": 0.0},
        )

    strict = run("strict")
    assert client.calls[0][1]["max_tokens"] == 4096
    metrics = strict["ifeval_metrics"]
    assert metrics["prompt_level_strict_accuracy"] == 1 / 4
    assert metrics["prompt_level_loose_accuracy"] == 2 / 4
    assert metrics["instruction_level_strict_accuracy"] == 2 / 6
    assert metrics["instruction_level_loose_accuracy"] == 3 / 6
    assert metrics["num_reasoning_blocks_stripped"] == 1
    assert metrics["per_instruction_group"]["punctuation"]["total"] == 4
    assert metrics["verifier"]["source"].startswith("google-research")
    assert strict["ifeval_instruction_scores"]["startend:end_checker"]["loose"]["accuracy"] == 1.0
    assert (strict["correct"], strict["num_failed_examples"]) == (1, 1)
    assert strict["results"][3]["follow_strict"] == [False, False]

    loose = run("loose")
    assert loose["correct"] == 2
    assert loose["ifeval_metrics"]["headline"] == "prompt_level_loose_accuracy"


