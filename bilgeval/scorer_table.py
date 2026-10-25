"""The one place that says which scorers and dataset types exist.

Everything that used to be a hand-maintained list — the registry Literals and
cross-field rules, the reader and evaluator dispatch, the preflight branches,
the metric families, the client requirements — derives from these two tables.
The strings are part of run fingerprints and of the web contract, so they
are copied from the code they replace and must not be renamed.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetType:
    """A kind of dataset: how it is read, validated and grouped in reports."""

    public: str
    internal: str
    family: str
    few_shot: bool = False
    validate_records: bool = True
    id_required: bool = False


DATASET_TYPES: dict[str, DatasetType] = {
    item.public: item
    for item in (
        DatasetType("typed_decision", "typed_decision", "typed_decision", validate_records=False, id_required=True),
        DatasetType("multiple_choice", "mc", "multiple_choice", few_shot=True),
        DatasetType("free_response", "free_response", "free_response", few_shot=True),
        DatasetType("code_generation", "humaneval", "code_generation"),
        DatasetType("ifeval", "ifeval", "instruction_following", validate_records=False),
        DatasetType("graphwalks", "graphwalks", "graph_reasoning", validate_records=False),
        DatasetType("safety_behavior", "safety_behavior", "safety"),
        DatasetType("document_qa", "document_qa", "document_qa"),
        DatasetType("function_call", "function_call", "function_calling"),
        DatasetType("multi_turn_judge", "multi_turn_judge", "multi_turn_judge", id_required=True),
        DatasetType("livecodebench", "livecodebench", "code_generation", id_required=True),
        DatasetType("oneruler", "oneruler", "long_context", validate_records=False),
    )
}
PUBLIC_TASK_TYPES: tuple[str, ...] = tuple(DATASET_TYPES)
INTERNAL_ALIASES = {"mc_logprob": "mc"}
INTERNAL_DATASET_TYPES: dict[str, DatasetType] = {
    **{item.internal: item for item in DATASET_TYPES.values()},
    **{alias: DATASET_TYPES["multiple_choice"] for alias in INTERNAL_ALIASES},
}


def dataset_type(internal: str) -> DatasetType:
    try:
        return INTERNAL_DATASET_TYPES[internal]
    except KeyError:
        raise ValueError(f"Unsupported task type: {internal}") from None


@dataclass(frozen=True)
class ScorerDef:
    """What a scorer needs and promises; the evaluate function lives in scorers.py."""

    id: str
    eval_mode: str
    dataset_type: str
    request_type: str = "generate"
    generation_api: str | None = None
    generation_api_reason: str = ""
    multi_sample: bool = False
    consensus: bool = False
    judge_prompt: str = "forbidden"
    answer_verification: bool = False
    expected_metric: str = "accuracy"
    requires: frozenset = field(default_factory=lambda: frozenset({"generation"}))
    default_max_tokens: int | None = None
    mean_metric_name: str = "mean_sample_accuracy"

    @property
    def multi_sample_aggregations(self) -> frozenset:
        return frozenset({"first", "mean"} | ({"consensus"} if self.consensus else set()))


def _scorers(*items: ScorerDef) -> dict[str, ScorerDef]:
    return {item.id: item for item in items}


GENERATED_MC_MAX_TOKENS = 256
SCORERS: dict[str, ScorerDef] = _scorers(
    ScorerDef("typed_decision", "typed_decision", "typed_decision", request_type="decide", requires=frozenset({"td"})),
    ScorerDef(
        "mc_logprob",
        "logprob",
        "multiple_choice",
        request_type="logprob",
        requires=frozenset({"logprob"}),
    ),
    ScorerDef(
        "mc_exact_match",
        "exact_match",
        "multiple_choice",
        multi_sample=True,
        consensus=True,
        default_max_tokens=GENERATED_MC_MAX_TOKENS,
    ),
    ScorerDef(
        "mc_regex_match",
        "regex_match",
        "multiple_choice",
        multi_sample=True,
        consensus=True,
        answer_verification=True,
        default_max_tokens=GENERATED_MC_MAX_TOKENS,
    ),
    ScorerDef(
        "mc_judge",
        "judge",
        "multiple_choice",
        judge_prompt="required",
        requires=frozenset({"generation", "judge"}),
        default_max_tokens=GENERATED_MC_MAX_TOKENS,
    ),
    ScorerDef(
        "free_response_exact_match",
        "exact_match",
        "free_response",
        multi_sample=True,
        consensus=True,
        default_max_tokens=64,
    ),
    ScorerDef(
        "free_response_regex_match",
        "regex_match",
        "free_response",
        multi_sample=True,
        consensus=True,
        answer_verification=True,
        default_max_tokens=64,
    ),
    ScorerDef(
        "humaneval",
        "humaneval",
        "code_generation",
        multi_sample=True,
        requires=frozenset({"generation", "executor"}),
        default_max_tokens=1024,
        mean_metric_name="pass@1",
    ),
    ScorerDef("ifeval", "ifeval", "ifeval", default_max_tokens=1024),
    ScorerDef("graphwalks_exact_match", "exact_match", "graphwalks", default_max_tokens=512),
    ScorerDef(
        "safety_judge",
        "safety_judge",
        "safety_behavior",
        requires=frozenset({"generation", "judge"}),
        default_max_tokens=1024,
    ),
    ScorerDef(
        "document_qa",
        "document_qa",
        "document_qa",
        requires=frozenset({"generation", "judge"}),
        default_max_tokens=1024,
    ),
    ScorerDef(
        "bfcl_ast",
        "bfcl_ast",
        "function_call",
        generation_api="chat",
        generation_api_reason="native tools are carried by chat completions",
        requires=frozenset({"tool"}),
        default_max_tokens=256,
    ),
    ScorerDef(
        "mtbench_judge",
        "mtbench_judge",
        "multi_turn_judge",
        generation_api="chat",
        generation_api_reason="the benchmark evaluates multi-turn conversations",
        expected_metric="mtbench_score",
        requires=frozenset({"generation", "judge"}),
        default_max_tokens=2048,
    ),
    ScorerDef(
        "livecodebench",
        "livecodebench",
        "livecodebench",
        multi_sample=True,
        requires=frozenset({"generation", "executor"}),
        default_max_tokens=2000,
        mean_metric_name="pass@1",
    ),
    ScorerDef("oneruler_tr", "oneruler_tr", "oneruler", default_max_tokens=4096),
)
SCORER_IDS: tuple[str, ...] = tuple(SCORERS)
EVAL_MODES: frozenset = frozenset(item.eval_mode for item in SCORERS.values())
LOGPROB_EVAL_MODE = "logprob"
ANSWER_PARSED_EVAL_MODES: frozenset = frozenset({"regex_match", "exact_match"})
EVAL_MODE_BY_DATASET_TYPE: dict[str, str] = {
    "ifeval": "ifeval",
    "graphwalks": "exact_match",
    "code_generation": "humaneval",
    "livecodebench": "livecodebench",
    "safety_behavior": "safety_judge",
    "document_qa": "document_qa",
    "function_call": "bfcl_ast",
    "multi_turn_judge": "mtbench_judge",
    "oneruler": "oneruler_tr",
    "typed_decision": "typed_decision",
}
ONERULER_TR_RULES_VERSION = "1"


def scorer(scorer_id: str) -> ScorerDef:
    try:
        return SCORERS[scorer_id]
    except KeyError:
        raise ValueError(f"Unsupported scorer: {scorer_id}") from None


def scorer_for(eval_mode: str, dataset_public_type: str) -> ScorerDef:
    """The scorer a (legacy eval_mode, dataset type) pair means."""
    for item in SCORERS.values():
        if item.eval_mode == eval_mode and item.dataset_type == dataset_public_type:
            return item
    raise ValueError(f"No scorer evaluates {dataset_public_type!r} data with eval_mode {eval_mode!r}.")
