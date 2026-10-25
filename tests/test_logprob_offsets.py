"""Server-added special tokens must never shift scoring into the prompt."""

from types import SimpleNamespace

import pytest

from bilgeval.evaluators import OpenAICompatibleLogprobClient


def score_echo(*, prompt="Prompt", completion=" A", prefix="<bos>",
               offsets=None, known_prefix="<bos>", probe_marker=None,
               add_bos_token=False, echoed_prefix="", token_suffix=None):
    echoed = echoed_prefix + prompt + completion + "\n"
    tokens = [prefix, echoed_prefix + prompt, completion, "\n"]
    if token_suffix is not None:
        tokens[2] = token_suffix
    if offsets is None:
        offsets = [0, len(prefix), len(prefix) + len(echoed_prefix + prompt),
                   len(prefix) + len(echoed_prefix + prompt + completion)]
    response = SimpleNamespace(choices=[SimpleNamespace(
        index=0, text=echoed, logprobs=SimpleNamespace(
            tokens=tokens, token_logprobs=[None, -9.0, -0.25, -0.5], text_offset=offsets,
        ),
    )])
    client = OpenAICompatibleLogprobClient(
        model="fixture", bos_token=known_prefix, add_bos_token=add_bos_token,
    )
    client.client.close()
    requests = []
    client.client = SimpleNamespace(completions=SimpleNamespace(
        create=lambda **request: requests.append(request) or response,
    ))
    if probe_marker is not None:
        client.bos_diagnostics["sequence_start_probe"] = {
            "chat_template_first_token": probe_marker,
            "bos_token": None,
        }
    result = client.score_prompt_completion_pairs([(prompt, completion)])[0]
    return result, requests


def test_known_hidden_bos_preserves_context_and_scores_only_completion():
    result, requests = score_echo()
    assert result.error is None
    assert result.score == -0.25
    assert result.num_completion_tokens == 1
    assert requests[0]["prompt"] == ["Prompt A"]
    assert "add_special_tokens" not in requests[0]


def test_probed_hidden_chat_marker_can_be_auto_added_as_a_sequence_prefix():
    result, _ = score_echo(prefix="<|im_start|>", probe_marker="<|im_start|>")
    assert result.error is None
    assert result.score == -0.25


def test_hidden_prefix_cannot_make_a_prompt_token_look_like_the_completion():
    response = SimpleNamespace(choices=[SimpleNamespace(
        index=0, text="First prompt: A\n", logprobs=SimpleNamespace(
            tokens=["<bos>", "First pr", "ompt:", " A", "\n"],
            token_logprobs=[None, -1.0, -9.0, -0.25, -0.5],
            text_offset=[0, 5, 13, 18, 20],
        ),
    )])
    client = OpenAICompatibleLogprobClient(model="fixture")
    client.client.close()
    client.client = SimpleNamespace(completions=SimpleNamespace(create=lambda **request: response))
    result = client.score_prompt_completion_pairs([("First prompt:", " A")])[0]
    assert result.error is None
    assert result.score == -0.25
    assert result.num_completion_tokens == 1


@pytest.mark.parametrize("kwargs", [
    {"prefix": "<unknown>"},
    {"prefix": "hidden text"},
    {"prefix": "<|im_start|>"},
    {"offsets": [0, 5, 6, 8]},
    {"offsets": [-1, -1, -1, -1]},
    {"token_suffix": " B"},
])
def test_unverified_hidden_prefix_or_offsets_never_produce_a_score(kwargs):
    result, _ = score_echo(**kwargs)
    assert result.error is not None
    assert result.score is None


def test_explicit_client_bos_and_verified_server_prefix_are_both_accounted_for():
    result, requests = score_echo(add_bos_token=True, echoed_prefix="<bos>")
    assert result.error is None
    assert result.score == -0.25
    assert requests[0]["prompt"] == ["<bos>Prompt A"]


def test_ordinary_positive_offsets_remain_authoritative_for_split_unicode_tokens():
    response = SimpleNamespace(choices=[SimpleNamespace(
        index=0, text="Emoji 😊. A\n", logprobs=SimpleNamespace(
            tokens=["Emoji ", "�", "�", ".", " A", "\n"],
            token_logprobs=[None, -3.0, -4.0, -5.0, -0.25, -0.5],
            text_offset=[0, 6, 6, 7, 8, 10],
        ),
    )])
    client = OpenAICompatibleLogprobClient(model="fixture")
    client.client.close()
    client.client = SimpleNamespace(completions=SimpleNamespace(create=lambda **request: response))
    result = client.score_prompt_completion_pairs([("Emoji 😊.", " A")])[0]
    assert result.error is None
    assert result.score == -0.25
    assert result.num_completion_tokens == 1


@pytest.mark.parametrize("extra_prefix", [False, True])
def test_explicit_marker_with_lossy_unicode_rejects_only_an_extra_hidden_marker(extra_prefix):
    tokens = ["<bos>", "First pr", "ompt:", " A", "�", "�"]
    offsets = [0, 5, 13, 18, 20, 20]
    logprobs = [None, -1.0, -9.0, -0.25, -0.5, -0.5]
    if extra_prefix:
        tokens.insert(0, "<bos>")
        offsets = [0] + [offset + 5 for offset in offsets]
        logprobs = [None, -2.0] + logprobs[1:]
    response = SimpleNamespace(choices=[SimpleNamespace(
        index=0, text="<bos>First prompt: A😊", logprobs=SimpleNamespace(
            tokens=tokens, token_logprobs=logprobs, text_offset=offsets,
        ),
    )])
    client = OpenAICompatibleLogprobClient(model="fixture")
    client.client.close()
    client.client = SimpleNamespace(completions=SimpleNamespace(create=lambda **request: response))
    result = client.score_prompt_completion_pairs([("<bos>First prompt:", " A")])[0]
    if extra_prefix:
        assert result.error is not None
        assert result.score is None
    else:
        assert result.error is None
        assert result.score == -0.25
