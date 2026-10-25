from bilgeval.evaluators import (
    graphwalks_breakdown,
    graphwalks_size_tokens,
    parse_graphwalks_nodes,
)
from bilgeval.reader import GraphWalksExample


def test_graphwalks_parser_accepts_bare_node_ids_from_final_answer():
    response = """
The earlier candidates were [wrong_node, another_wrong_node].
After traversing the graph, these are the direct parents.
Final Answer: [cfcd208495, a87ff679a2, e4da3b7fbb]
"""

    assert parse_graphwalks_nodes(response) == [
        "cfcd208495",
        "a87ff679a2",
        "e4da3b7fbb",
    ]


def test_graphwalks_parser_keeps_support_for_quoted_python_lists():
    assert parse_graphwalks_nodes(
        "Final Answer: ['cfcd208495', 'a87ff679a2']"
    ) == ["cfcd208495", "a87ff679a2"]


def test_graphwalks_parser_accepts_an_empty_final_answer():
    assert parse_graphwalks_nodes("Reasoning.\nFinal Answer: []") == []


def test_graphwalks_parser_scores_unstructured_text_as_empty_set():
    assert parse_graphwalks_nodes("No structured answer\nwas emitted") == []


def test_graphwalks_parser_accepts_the_turkish_marker():
    assert parse_graphwalks_nodes("Adım adım...\nCevap: [cfcd208495, a87ff679a2]") == [
        "cfcd208495",
        "a87ff679a2",
    ]
    assert parse_graphwalks_nodes("cevap: []") == []
    assert parse_graphwalks_nodes("Final Answer: [a]\nDüzeltme:\nCevap: [b]") == ["b"]
    assert parse_graphwalks_nodes("Cevap: [a]\nFinal Answer: [b]") == ["b"]


def test_graphwalks_size_bins_follow_openai_nominal_sizes():
    assert graphwalks_size_tokens({"prompt_chars": 2638}) == 1024
    assert graphwalks_size_tokens({"prompt_chars": 4425}) == 2048
    assert graphwalks_size_tokens({"prompt_chars": 55626}) == 32768
    assert graphwalks_size_tokens({"prompt_chars": 219451}) == 131072
    assert graphwalks_size_tokens({"prompt_chars": 437413}) == 262144
    assert graphwalks_size_tokens({"prompt_chars": 1748226}) == 2**20
    assert graphwalks_size_tokens({"prompt_chars": 3100, "size_tokens": 1024}) == 1024
    assert graphwalks_size_tokens({}) is None


def test_graphwalks_breakdown_counts_unscored_rows_as_zero_per_size():
    def example(idx, chars, problem_type):
        return GraphWalksExample(
            id=str(idx),
            prompt="p",
            answer_nodes=["a"],
            problem_type=problem_type,
            raw={"prompt_chars": chars} if chars else {},
        )

    examples = [
        example(0, 2638, "bfs"),
        example(1, 2663, "parents"),
        example(2, 55538, "parents"),
        example(3, None, "parents"),
    ]
    results = [
        {"failed": False, "correct": True, "f1": 1.0},
        {"failed": False, "correct": False, "f1": 0.5},
        {"failed": True},
        {"failed": False, "correct": False, "f1": 0.0},
    ]

    metrics = graphwalks_breakdown(examples, results)

    assert metrics["by_size"] == [
        {
            "size": "1k",
            "size_tokens": 1024,
            "num_examples": 2,
            "num_failed_examples": 0,
            "mean_f1": 0.75,
            "exact_match_accuracy": 0.5,
        },
        {
            "size": "32k",
            "size_tokens": 32768,
            "num_examples": 1,
            "num_failed_examples": 1,
            "mean_f1": 0.0,
            "exact_match_accuracy": 0.0,
        },
        {
            "size": "unknown",
            "size_tokens": None,
            "num_examples": 1,
            "num_failed_examples": 0,
            "mean_f1": 0.0,
            "exact_match_accuracy": 0.0,
        },
    ]
    assert [(row["problem_type"], row["num_examples"], row["mean_f1"]) for row in metrics["by_problem_type"]] == [
        ("bfs", 1, 1.0),
        ("parents", 3, 0.5 / 3),
    ]
