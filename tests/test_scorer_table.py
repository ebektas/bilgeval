"""The scorer table reproduces the literals it replaced. These dicts are
frozen copies of the maps that used to live in main.py and
registry_bundle.py; a change here is a change of task identity."""

import typing

import pytest

from bilgeval import scorer_table
from bilgeval.registry_bundle import DatasetSpec, ScorerSpec


EVAL_MODE_BY_SCORER = {
    "typed_decision": "typed_decision",
    "mc_logprob": "logprob",
    "mc_exact_match": "exact_match",
    "mc_regex_match": "regex_match",
    "mc_judge": "judge",
    "free_response_exact_match": "exact_match",
    "free_response_regex_match": "regex_match",
    "humaneval": "humaneval",
    "ifeval": "ifeval",
    "graphwalks_exact_match": "exact_match",
    "safety_judge": "safety_judge",
    "document_qa": "document_qa",
    "bfcl_ast": "bfcl_ast",
    "mtbench_judge": "mtbench_judge",
    "livecodebench": "livecodebench",
    "oneruler_tr": "oneruler_tr",
}
DATASET_TYPE_BY_SCORER = {
    "typed_decision": "typed_decision",
    "mc_logprob": "multiple_choice",
    "mc_exact_match": "multiple_choice",
    "mc_regex_match": "multiple_choice",
    "mc_judge": "multiple_choice",
    "free_response_exact_match": "free_response",
    "free_response_regex_match": "free_response",
    "humaneval": "code_generation",
    "ifeval": "ifeval",
    "graphwalks_exact_match": "graphwalks",
    "safety_judge": "safety_behavior",
    "document_qa": "document_qa",
    "bfcl_ast": "function_call",
    "mtbench_judge": "multi_turn_judge",
    "livecodebench": "livecodebench",
    "oneruler_tr": "oneruler",
}
INTERNAL_BY_PUBLIC = {
    "typed_decision": "typed_decision",
    "multiple_choice": "mc",
    "free_response": "free_response",
    "code_generation": "humaneval",
    "ifeval": "ifeval",
    "graphwalks": "graphwalks",
    "safety_behavior": "safety_behavior",
    "document_qa": "document_qa",
    "function_call": "function_call",
    "multi_turn_judge": "multi_turn_judge",
    "livecodebench": "livecodebench",
    "oneruler": "oneruler",
}
METRIC_FAMILIES = {
    "typed_decision": "typed_decision",
    "mc": "multiple_choice",
    "mc_logprob": "multiple_choice",
    "free_response": "free_response",
    "humaneval": "code_generation",
    "livecodebench": "code_generation",
    "ifeval": "instruction_following",
    "graphwalks": "graph_reasoning",
    "safety_behavior": "safety",
    "document_qa": "document_qa",
    "function_call": "function_calling",
    "oneruler": "long_context",
}
DEFAULT_MAX_TOKENS = {
    "typed_decision": None,
    "mc_logprob": None,
    "mc_exact_match": 256,
    "mc_regex_match": 256,
    "mc_judge": 256,
    "free_response_exact_match": 64,
    "free_response_regex_match": 64,
    "humaneval": 1024,
    "ifeval": 1024,
    "graphwalks_exact_match": 512,
    "safety_judge": 1024,
    "document_qa": 1024,
    "bfcl_ast": 256,
    "mtbench_judge": 2048,
    "livecodebench": 2000,
    "oneruler_tr": 4096,
}


def test_scorer_identities_are_the_frozen_literals():
    assert {name: item.eval_mode for name, item in scorer_table.SCORERS.items()} == EVAL_MODE_BY_SCORER
    assert {name: item.dataset_type for name, item in scorer_table.SCORERS.items()} == DATASET_TYPE_BY_SCORER
    assert {name: item.default_max_tokens for name, item in scorer_table.SCORERS.items()} == DEFAULT_MAX_TOKENS
    assert scorer_table.ANSWER_PARSED_EVAL_MODES == {"regex_match", "exact_match"}
    assert scorer_table.LOGPROB_EVAL_MODE == "logprob"


def test_dataset_types_are_the_frozen_literals():
    assert {name: item.internal for name, item in scorer_table.DATASET_TYPES.items()} == INTERNAL_BY_PUBLIC
    families = {
        internal: scorer_table.dataset_type(internal).family for internal in METRIC_FAMILIES
    }
    assert families == METRIC_FAMILIES
    assert scorer_table.dataset_type("multi_turn_judge").family == "multi_turn_judge"
    assert scorer_table.dataset_type("mc_logprob") is scorer_table.DATASET_TYPES["multiple_choice"]


def test_registry_literals_come_from_the_table():
    assert typing.get_args(ScorerSpec.model_fields["id"].annotation) == scorer_table.SCORER_IDS
    assert typing.get_args(DatasetSpec.model_fields["task_type"].annotation) == scorer_table.PUBLIC_TASK_TYPES


def test_every_scorer_has_a_runtime_binding():
    scorers = pytest.importorskip("bilgeval.scorers")

    assert set(scorers.SCORER_RUNTIMES) == set(scorer_table.SCORERS)
    assert set(scorers.READERS) >= {item.internal for item in scorer_table.DATASET_TYPES.values()}


def test_scorer_lookup_by_legacy_pair():
    assert scorer_table.scorer_for("exact_match", "graphwalks").id == "graphwalks_exact_match"
    assert scorer_table.scorer_for("regex_match", "free_response").id == "free_response_regex_match"
    assert scorer_table.SCORERS["mc_regex_match"].multi_sample_aggregations == {"first", "consensus", "mean"}
    assert scorer_table.SCORERS["humaneval"].multi_sample_aggregations == {"first", "mean"}
