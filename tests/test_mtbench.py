from bilgeval.evaluators import (
    GeneratedTextResult,
    TextGenerationModelClient,
    evaluate_multi_turn_judge,
)
from bilgeval import main as main_module
from bilgeval.reader import MultiTurnJudgeExample
from bilgeval.specialized_tasks import (
    mtbench_candidate_messages,
    mtbench_judge_messages,
    parse_mtbench_rating,
)


def example(
    dialogue_id: str,
    task_id: str,
    ratings_turns: int = 3,
    skip_first: bool = False,
) -> MultiTurnJudgeExample:
    turns = []
    for index in range(ratings_turns):
        turns.extend(
            [
                {"role": "human", "content": f"question {index}"},
                {"role": "assistant", "content": f"gold answer {index}"},
            ]
        )
    return MultiTurnJudgeExample(
        id=dialogue_id,
        task_id=task_id,
        task_name=task_id,
        topic="test",
        turns=turns,
        eval_turn_positions=list(range(ratings_turns))[1 if skip_first else 0 :],
        evaluation_initial="Evaluate only the last assistant response.",
        evaluation_prompt=f"Criteria for {task_id}.",
        scoring_format="DIALOGUE\nRating: [[score]]",
        raw={"id": dialogue_id, "task_id": task_id, "topic": "test"},
    )


class ConversationClient(TextGenerationModelClient):
    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.conversations = []
        self.request_index = 0

    def generate_conversations(self, conversations, **kwargs):
        self.conversations.extend(conversations)
        results = []
        for _ in conversations:
            self.request_index += 1
            results.append(
                GeneratedTextResult(
                    text=next(self.outputs),
                    request_id=f"request-{self.request_index}",
                )
            )
        return results


def test_candidate_context_uses_golden_history_before_target_turn():
    item = example("CM__test__001", "CM", ratings_turns=3, skip_first=True)

    assert mtbench_candidate_messages(item, 1) == [
        {"role": "user", "content": "question 0"},
        {"role": "assistant", "content": "gold answer 0"},
        {"role": "user", "content": "question 1"},
    ]


def test_judge_prompt_replaces_target_gold_with_candidate_response():
    item = example("SI__test__001", "SI", ratings_turns=2)
    messages = mtbench_judge_messages(item, 1, "candidate answer")

    assert messages[0]["role"] == "system"
    assert "candidate answer" in messages[1]["content"]
    assert "gold answer 1" not in messages[1]["content"]
    assert "gold answer 0" in messages[1]["content"]
    assert "Criteria for SI." in messages[1]["content"]


def test_rating_parser_uses_last_valid_in_range_rating():
    assert parse_mtbench_rating("Draft [[4]].\nRating: [[8.5]]") == 8.5
    assert parse_mtbench_rating("Rating: [[0]]") is None
    assert parse_mtbench_rating("Rating: [[11]]") is None
    assert parse_mtbench_rating("Rating: 8") is None


def test_evaluator_uses_minimum_turn_then_macro_task_mean():
    examples = [
        example("CM__test__001", "CM", ratings_turns=3, skip_first=True),
        example("SI__test__001", "SI", ratings_turns=2),
    ]
    candidate = ConversationClient(
        ["candidate cm1", "candidate cm2", "candidate si0", "candidate si1"]
    )
    judge = ConversationClient(
        [
            "Rating: [[8]]",
            "Rating: [[6]]",
            "Rating: [[9]]",
            "Rating: [[7]]",
        ]
    )

    report = evaluate_multi_turn_judge(
        examples,
        candidate_client=candidate,
        judge_client=judge,
        request_batch_size=8,
    )

    assert report["score"] == 6.5
    assert report["coverage"] == 1.0
    assert report["num_requests"] == 8
    assert report["results"][0]["score"] == 6.0
    assert report["results"][1]["score"] == 7.0
    assert report["mtbench_metrics"]["per_task"]["CM"]["mean"] == 6.0
    assert report["mtbench_metrics"]["per_task"]["SI"]["mean"] == 7.0


def test_unparseable_judge_turn_fails_the_whole_dialogue():
    item = example("SI__test__001", "SI", ratings_turns=2)
    candidate = ConversationClient(["candidate 0", "candidate 1"])
    judge = ConversationClient(["Rating: [[8]]", "no structured rating"])

    report = evaluate_multi_turn_judge(
        [item],
        candidate_client=candidate,
        judge_client=judge,
    )

    assert report["num_scored_examples"] == 0
    assert report["num_failed_examples"] == 1
    assert report["mtbench_metrics"]["num_scored_turns"] == 1
    assert report["mtbench_metrics"]["num_failed_turns"] == 1


