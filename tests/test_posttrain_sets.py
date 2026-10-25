"""The post-training sets of posttrain_sets.yaml: strict zero-shot chain of thought
scored by the parser, and the official few-shot protocols over chat with the verifier."""
from pathlib import Path

import pytest

from bilgeval import main as main_module
from bilgeval.generation_prompts import build_generation_prompt
from bilgeval.reader import Choice, Example
from bilgeval.registry_bundle import load_registry_bundle


@pytest.fixture(scope="module")
def registry():
    return load_registry_bundle(Path(main_module.__file__).with_name("builtin_registry.json"))




def test_every_turkish_strict_multiple_choice_task_has_a_verified_zero_shot_twin(registry):
    for name, task in registry.tasks.items():
        if not name.endswith(":posttrain-strict") or task.scorer.id != "mc_regex_match" or not name.startswith("tr_"):
            continue
        twin = registry.tasks[name.replace(":posttrain-strict", ":posttrain-chat")]
        assert (twin.dataset, twin.formatter.id, twin.scorer.answer_verification) == (
            task.dataset, "turkish_mc_v1", "llm_as_regex"), name




def test_strict_prompt_ends_with_the_choices():
    choices = [Choice(label=label, text=f"seçenek {label}") for label in "ABCDE"]
    example = Example(id="q", question="Hangisi?", choices=choices, answer_label="B", answer_text="seçenek B", raw={})
    prompt = build_generation_prompt(example, "turkish_mc_cot_strict_v1", append_choices=True)
    assert prompt == (
        "Aşağıdaki çoktan seçmeli soruyu adım adım çöz. Yanıtının son satırı `Cevap: X` "
        "biçiminde olmalı; X doğru seçeneğin harfidir.\n\nSoru: Hangisi?\n\nSeçenekler:\n"
        "A) seçenek A\nB) seçenek B\nC) seçenek C\nD) seçenek D\nE) seçenek E\n"
    )




def test_lm_eval_arc_replica_prompt():
    example = Example(id="q", question="Which one?", choices=[], answer_label="A", answer_text="x", raw={})
    assert build_generation_prompt(example, "lm_eval_arc_v1") == "Question: Which one?\nAnswer:"
