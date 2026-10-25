import httpx
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bilgeval.evaluators import (
    BatchRequestError,
    CompletionScoreResult,
    GeneratedTextResult,
    LogprobModelClient,
    OpenAICompatibleLogprobClient,
    OpenAICompatibleTextGenerationClient,
    SGLangNativeLogprobClient,
    THINKING_DISABLED_EXTRA_BODY,
    create_logprob_client,
    evaluate_logprob_mc,
    generated_request_count,
    model_requires_bos,
    probe_sequence_start,
    resolve_bos_behavior,
)
from bilgeval.reader import Choice, Example
from bilgeval.registry_bundle import PromptRendererSpec


class RecordingTraceCollector:
    active = True

    def __init__(self):
        self.calls = []

    def capture(self, call):
        self.calls.append(call)
        return {"trace_id": f"trace-{len(self.calls)}"}


class LogprobClientTests(unittest.TestCase):
    def test_bos_auto_adds_one_token_when_gemma_server_omits_it(self):
        effective, diagnostics = resolve_bos_behavior(
            mode="auto",
            bos_required=True,
            bos_token="<bos>",
            plain_tokens=["Bilgeval", " BOS"],
            explicit_tokens=["<bos>", "Bilgeval", " BOS"],
        )

        self.assertTrue(effective)
        self.assertEqual(diagnostics["source"], "bilgeval_prefix")

    def test_bos_auto_does_not_duplicate_server_added_token(self):
        effective, diagnostics = resolve_bos_behavior(
            mode="auto",
            bos_required=True,
            bos_token="<bos>",
            plain_tokens=["<bos>", "Bilgeval"],
            explicit_tokens=["<bos>", "Bilgeval"],
        )

        self.assertFalse(effective)
        self.assertEqual(diagnostics["source"], "server_tokenizer")

    def test_bos_probe_rejects_duplicate_prefix(self):
        with self.assertRaisesRegex(ValueError, "leading BOS"):
            resolve_bos_behavior(
                mode="auto",
                bos_required=True,
                bos_token="<bos>",
                plain_tokens=["<bos>", "<bos>", "Bilgeval"],
                explicit_tokens=["<bos>", "<bos>", "Bilgeval"],
            )

    def test_model_family_detection_uses_sglang_model_info(self):
        self.assertTrue(
            model_requires_bos(
                "checkpoint-43",
                {"model_type": "gemma4", "tokenizer_path": "/models/gemma-4-31b"},
            )
        )
        self.assertFalse(model_requires_bos("meta-llama/Llama-3.1-8B"))

    def test_client_factory_selects_native_sglang_and_resolves_gemma_bos(self):
        with (
            patch(
                "bilgeval.evaluators.get_sglang_model_info",
                return_value={"model_path": "/models/gemma", "model_type": "gemma4"},
            ),
            patch.object(
                SGLangNativeLogprobClient,
                "probe_input_tokens",
                return_value=[
                    ["Bil", "ge", "val"],
                    ["<bos>", "Bil", "ge", "val"],
                ],
            ),
        ):
            client = create_logprob_client(
                base_url="https://example.test/v1",
                model="checkpoint",
            )

        self.assertIsInstance(client, SGLangNativeLogprobClient)
        self.assertTrue(client.effective_add_bos_token)
        self.assertEqual(client.bos_diagnostics["transport"], "sglang_generate")

    def test_non_gemma_auto_does_not_probe_or_add_bos(self):
        with (
            patch("bilgeval.evaluators.get_sglang_model_info", return_value=None),
            patch("bilgeval.evaluators.probe_sequence_start", return_value=None),
            patch.object(
                OpenAICompatibleLogprobClient,
                "probe_input_tokens",
                side_effect=AssertionError("non-Gemma auto must not probe"),
            ),
        ):
            client = create_logprob_client(
                base_url="https://example.test/v1",
                model="meta-llama/Llama-3.1-8B",
            )

        self.assertFalse(client.effective_add_bos_token)
        self.assertEqual(client.bos_diagnostics["source"], "model_does_not_require_bos")

    def test_sequence_start_probe_reads_the_chat_template_and_the_plain_tokenizer(self):
        servers = {
            "gemma": (["<bos>", "<|turn>", "user", "a", "<turn|>", "<|turn>", "model", "b"], [2, 105, 7, 9, 106, 105, 8, 10],
                      [83547, 787], ("<bos>", False)),
            "qwen": (["<|im_start|>", "user", "a", "<|im_end|>", "<|im_start|>", "assistant", "b"], [1, 2, 3, 4, 1, 5, 6],
                     [33, 319], (None, False)),
            "deepseek": (["<｜begin▁of▁sentence｜>", "<｜User｜>", "a", "<｜Assistant｜>", "b"], [0, 11, 12, 13, 14],
                         [36, 321], ("<｜begin▁of▁sentence｜>", False)),
            "llama": (["<|begin_of_text|>", "<|start_header_id|>", "user", "<|start_header_id|>", "assistant"], [128000, 6, 7, 6, 8],
                      [128000, 33], ("<|begin_of_text|>", True)),
        }
        for name, (chat_strs, chat_ids, plain_ids, (bos, server_adds)) in servers.items():
            def post(url, json, **_kwargs):
                if "messages" in json:
                    body = {"tokens": chat_ids, "token_strs": chat_strs}
                else:
                    body = {"tokens": plain_ids, "token_strs": None}
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

            with patch("bilgeval.evaluators.httpx.post", side_effect=post):
                probe = probe_sequence_start(
                    base_url="https://example.test/v1", model=name, api_key="EMPTY", insecure_skip_verify=False
                )
            self.assertEqual((probe["bos_token"], probe["server_adds_bos"]), (bos, server_adds), name)

    def test_sequence_start_probe_is_none_without_tokenize(self):
        def post(url, json, **_kwargs):
            raise httpx.HTTPStatusError("404", request=None, response=None)

        with patch("bilgeval.evaluators.httpx.post", side_effect=post):
            self.assertIsNone(
                probe_sequence_start(base_url="https://example.test/v1", model="m", api_key="", insecure_skip_verify=False)
            )

    def test_auto_bos_uses_the_probed_token_for_a_model_without_gemma_in_its_name(self):
        probe = {"source": "vllm_tokenize_chat_template", "bos_token": "<｜begin▁of▁sentence｜>",
                 "bos_token_id": 0, "server_adds_bos": False, "chat_template_first_token": "<｜begin▁of▁sentence｜>"}
        with (
            patch("bilgeval.evaluators.get_sglang_model_info", return_value=None),
            patch("bilgeval.evaluators.probe_sequence_start", return_value=probe),
            patch.object(
                OpenAICompatibleLogprobClient,
                "probe_input_tokens",
                return_value=[["B", "il"], ["<｜begin▁of▁sentence｜>", "B", "il"]],
            ),
        ):
            client = create_logprob_client(base_url="https://example.test/v1", model="deepseek-v4-flash")

        self.assertTrue(client.effective_add_bos_token)
        self.assertEqual(client.bos_token, "<｜begin▁of▁sentence｜>")
        self.assertEqual(client.bos_diagnostics["sequence_start_probe"], probe)

    def test_sglang_negative_offsets_are_reconstructed_and_generated_token_is_ignored(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    text="<bos>Prompt A\n",
                    logprobs=SimpleNamespace(
                        tokens=["<bos>", "Prompt", " A", "\n"],
                        token_logprobs=[None, -1.0, -0.25, -0.5],
                        text_offset=[-1, -1, -1, -1],
                    ),
                )
            ]
        )
        client = OpenAICompatibleLogprobClient(
            model="gemma",
            add_bos_token=True,
        )
        client.client = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **request: response)
        )

        result = client.score_prompt_completion_pairs([("Prompt", " A")])[0]

        self.assertIsNone(result.error)
        self.assertEqual(result.score, -0.25)
        self.assertEqual(result.num_completion_tokens, 1)

    def test_a_long_prompt_list_goes_out_in_requests_the_server_accepts(self):
        sizes = []

        def create(**request):
            prompts = request["prompt"]
            if len(prompts) > 2:
                raise AssertionError(f"prompt list length {len(prompts)} exceeds 2")
            sizes.append(len(prompts))
            return SimpleNamespace(choices=[
                SimpleNamespace(index=i, text=text + "\n", logprobs=SimpleNamespace(
                    tokens=["P", text[1:], "\n"], token_logprobs=[None, -float(text[2:]), -0.5],
                    text_offset=[0, 1, len(text)]))
                for i, text in enumerate(prompts)
            ])

        client = OpenAICompatibleLogprobClient(model="m")
        client.client = SimpleNamespace(completions=SimpleNamespace(create=create))
        pairs = [("P", f" {i}") for i in range(5)]
        with patch.dict("os.environ", {"BILGEVAL_MAX_PROMPTS_PER_REQUEST": "2"}):
            results = client.score_prompt_completion_pairs(pairs)

        self.assertEqual(sorted(sizes), [1, 2, 2])
        self.assertEqual([r.score for r in results], [-0.0, -1.0, -2.0, -3.0, -4.0])

    def test_logprob_request_uses_positive_max_tokens(self):
        requests = []
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    text="Prompt A\n",
                    logprobs=SimpleNamespace(
                        tokens=["Prompt", " A", "\n"],
                        token_logprobs=[None, -0.25, -0.5],
                        text_offset=[0, 6, 8],
                    ),
                )
            ]
        )
        client = OpenAICompatibleLogprobClient(model="gemma")
        client.client = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **request: requests.append(request) or response
            )
        )

        result = client.score_prompt_completion_pairs([("Prompt", " A")])[0]

        self.assertIsNone(result.error)
        self.assertEqual(requests[0]["max_tokens"], 1)
        self.assertEqual(result.score, -0.25)
        self.assertEqual(result.num_completion_tokens, 1)

    def test_logprob_serialization_is_applied_before_bos(self):
        requests = []
        serialized_prompt = "<bos><user>Prompt</user><model>"
        full_text = serialized_prompt + " A"
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    text=full_text + "\n",
                    logprobs=SimpleNamespace(
                        tokens=["<bos>", "<user>Prompt</user><model>", " A", "\n"],
                        token_logprobs=[None, -1.0, -0.25, -0.5],
                        text_offset=[0, 5, len(serialized_prompt), len(full_text)],
                    ),
                )
            ]
        )
        client = OpenAICompatibleLogprobClient(
            model="gemma",
            add_bos_token=True,
            prompt_renderer=PromptRendererSpec(
                mode="template",
                template_id="fixture",
                template="<user>{prompt}</user><model>",
            ),
        )
        client.client = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **request: requests.append(request) or response
            )
        )

        result = client.score_prompt_completion_pairs([("Prompt", " A")])[0]

        self.assertIsNone(result.error)
        self.assertEqual(requests[0]["prompt"], [full_text])
        self.assertEqual(result.score, -0.25)

    def test_openai_response_indices_must_match_batch(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(index=0, text="Prompt A", logprobs=None),
                SimpleNamespace(index=0, text="Prompt B", logprobs=None),
            ]
        )
        client = OpenAICompatibleLogprobClient(model="gemma")
        client.client = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **request: response)
        )

        with self.assertRaisesRegex(RuntimeError, "choice indices"):
            client.score_prompt_completion_pairs([("Prompt", " A"), ("Prompt", " B")])

    def test_native_sglang_scores_only_the_continuation(self):
        requests = []

        class Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return [
                    {
                        "meta_info": {
                            "input_token_logprobs": [
                                [None, 2, "<bos>"],
                                [-1.0, 1, "Prompt"],
                                [-0.25, 2, " A"],
                            ]
                        }
                    }
                ]

        client = SGLangNativeLogprobClient(
            base_url="https://example.test/v1",
            model="gemma",
            add_bos_token=True,
        )
        client.client = SimpleNamespace(
            post=lambda endpoint, json: requests.append((endpoint, json)) or Response()
        )

        result = client.score_prompt_completion_pairs([("Prompt", " A")])[0]

        self.assertIsNone(result.error)
        self.assertEqual(result.score, -0.25)
        self.assertEqual(result.num_completion_tokens, 1)
        self.assertEqual(requests[0][0], "https://example.test/generate")
        self.assertEqual(requests[0][1]["text"], ["<bos>Prompt A"])

    def test_tied_mc_scores_fail_instead_of_selecting_first_choice(self):
        class TiedClient(LogprobModelClient):
            model = "fixture"

            def score_prompt_completion_pairs(self, pairs, **kwargs):
                return [CompletionScoreResult(score=-1.0, num_completion_tokens=1) for _ in pairs]

        example = Example(
            id="tie-1",
            question="Test?",
            choices=[Choice(label="A", text="One"), Choice(label="B", text="Two")],
            answer_label="A",
            answer_text="One",
            raw={},
        )

        report = evaluate_logprob_mc(
            [example],
            TiedClient(),
            append=True,
            score_target="label",
        )

        self.assertEqual(report["num_scored_examples"], 0)
        self.assertEqual(report["num_failed_examples"], 1)
        self.assertIn("tied", report["results"][0]["error"])

    def test_pmi_normalization_preserves_conditional_and_reference_scores(self):
        class PMIClient(LogprobModelClient):
            model = "fixture"

            def score_prompt_completion_pairs(self, pairs, **kwargs):
                scores = []
                for prompt, completion in pairs:
                    label = completion.strip()
                    if prompt == "Yanıt:":
                        score = -5.0 if label == "A" else 0.0
                    else:
                        score = -0.2 if label == "A" else -0.1
                    scores.append(
                        CompletionScoreResult(score=score, num_completion_tokens=1)
                    )
                return scores

        example = Example(
            id="pmi-1",
            question="Test?",
            choices=[Choice(label="A", text="One"), Choice(label="B", text="Two")],
            answer_label="A",
            answer_text="One",
            raw={},
        )

        report = evaluate_logprob_mc(
            [example],
            PMIClient(),
            append=True,
            score_target="label",
            normalization="pmi",
            pmi_reference_prompt="Yanıt:",
        )

        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["num_candidate_scores"], 2)
        self.assertEqual(report["num_model_scores"], 4)
        first_score = report["results"][0]["scores"][0]
        self.assertEqual(first_score["conditional_score"], -0.2)
        self.assertEqual(first_score["pmi_reference_score"], -5.0)
        self.assertAlmostEqual(first_score["score"], 4.8)

    def test_pmi_normalization_requires_explicit_reference_prompt(self):
        with self.assertRaisesRegex(ValueError, "pmi_reference_prompt"):
            evaluate_logprob_mc(
                [],
                LogprobModelClient(),
                append=True,
                normalization="pmi",
            )


class TextGenerationClientTests(unittest.TestCase):
    def test_chat_generation_uses_configured_concurrency(self):
        worker_counts = []

        class ImmediateExecutor:
            def __init__(self, max_workers):
                worker_counts.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def map(self, function, values):
                return [function(value) for value in values]

        client = OpenAICompatibleTextGenerationClient(
            generation_api="chat",
            max_concurrency=512,
        )
        with (
            patch("bilgeval.evaluators.ThreadPoolExecutor", ImmediateExecutor),
            patch.object(
                client,
                "generate_chat_text",
                return_value=GeneratedTextResult(text="ok"),
            ),
        ):
            results = client.generate_texts(["prompt"] * 1024)

        self.assertEqual(worker_counts, [512])
        self.assertEqual(len(results), 1024)

    def test_candidate_generation_retries_failures_with_smaller_pools(self):
        worker_counts = []
        attempts = {}

        class ImmediateExecutor:
            def __init__(self, max_workers):
                worker_counts.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def map(self, function, values):
                return [function(value) for value in values]

        def generate(prompt, *args, **kwargs):
            attempts[prompt] = attempts.get(prompt, 0) + 1
            if int(prompt) < 200 and attempts[prompt] == 1:
                return GeneratedTextResult(error="Request timed out.")
            return GeneratedTextResult(text="ok")

        client = OpenAICompatibleTextGenerationClient(
            generation_api="chat",
            max_concurrency=512,
            request_role="generation",
        )
        with (
            patch("bilgeval.evaluators.ThreadPoolExecutor", ImmediateExecutor),
            patch.object(client, "generate_chat_text", side_effect=generate),
        ):
            results = client.generate_texts([str(index) for index in range(1024)])

        self.assertEqual(worker_counts, [512, 128])
        self.assertTrue(all(result.error is None for result in results))
        self.assertEqual(sum(attempts.values()), 1224)

    def test_generation_concurrency_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "max_concurrency"):
            OpenAICompatibleTextGenerationClient(max_concurrency=0)

    def test_generation_concurrency_configures_http_connection_pool(self):
        with (
            patch("bilgeval.evaluators.httpx.Client") as http_client,
            patch("bilgeval.evaluators.OpenAI"),
        ):
            OpenAICompatibleTextGenerationClient(
                max_concurrency=512,
                insecure_skip_verify=True,
            )

        limits = http_client.call_args.kwargs["limits"]
        timeout = http_client.call_args.kwargs["timeout"]
        self.assertEqual(limits.max_connections, 512)
        self.assertEqual(limits.max_keepalive_connections, 512)
        self.assertEqual(timeout.read, 3600.0)
        self.assertEqual(timeout.connect, 30.0)
        self.assertFalse(http_client.call_args.kwargs["verify"])

    def test_chat_generation_merges_default_extra_body_with_top_k(self):
        requests = []
        traces = RecordingTraceCollector()
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(content="[[8]]"),
                    finish_reason="stop",
                )
            ]
        )
        client = OpenAICompatibleTextGenerationClient(
            base_url="https://example.test/v1",
            model="judge",
            generation_api="chat",
            trace_collector=traces,
            default_extra_body={"chat_template_kwargs": {"thinking": False}},
        )
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **request: requests.append(request) or response
                )
            )
        )

        result = client.generate_texts(["Judge"], max_tokens=32, top_k=20)[0]

        self.assertEqual(result.text, "[[8]]")
        self.assertEqual(
            requests[0]["extra_body"],
            {
                "chat_template_kwargs": {"thinking": False},
                "top_k": 20,
            },
        )
        self.assertEqual(
            traces.calls[0]["request"]["chat_template_kwargs"],
            {"thinking": False},
        )
        self.assertEqual(traces.calls[0]["request"]["top_k"], 20)

    def test_chat_generation_raises_when_the_entire_batch_fails(self):
        client = OpenAICompatibleTextGenerationClient(
            base_url="https://example.test/v1",
            model="gemma",
            generation_api="chat",
        )
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_request: (_ for _ in ()).throw(
                        RuntimeError("server overloaded")
                    )
                )
            )
        )

        with self.assertRaisesRegex(BatchRequestError, "all 3 requests.*server overloaded"):
            client.generate_texts(["one", "two", "three"])

    def test_completion_generation_serializes_before_bos(self):
        requests = []
        response = SimpleNamespace(choices=[SimpleNamespace(index=0, text="B")])
        client = OpenAICompatibleTextGenerationClient(
            base_url="https://example.test/v1",
            model="gemma",
            add_bos_token=True,
            prompt_renderer=PromptRendererSpec(
                mode="template",
                template_id="fixture",
                template="<user>{prompt}</user><model>",
            ),
        )
        client.client = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **request: requests.append(request) or response
            )
        )

        result = client.generate_texts(["Prompt"], max_tokens=8)[0]

        self.assertIsNone(result.error)
        self.assertEqual(requests[0]["prompt"], ["<bos><user>Prompt</user><model>"])

    def test_completion_generation_forwards_sampling_parameters(self):
        requests = []
        traces = RecordingTraceCollector()
        response = SimpleNamespace(
            choices=[SimpleNamespace(index=0, text="42")]
        )
        client = OpenAICompatibleTextGenerationClient(
            base_url="https://example.test/v1",
            model="reasoning-model",
            trace_collector=traces,
        )
        client.client = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **request: requests.append(request) or response
            )
        )

        result = client.generate_texts(
            ["Prompt"],
            max_tokens=128,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop=["</answer>"],
        )[0]

        self.assertIsNone(result.error)
        self.assertEqual(result.text, "42")
        self.assertEqual(requests[0]["temperature"], 0.7)
        self.assertEqual(requests[0]["top_p"], 0.9)
        self.assertNotIn("seed", requests[0])
        self.assertEqual(requests[0]["stop"], ["</answer>"])
        self.assertEqual(requests[0]["extra_body"], {"top_k": 40})
        self.assertEqual(result.api_call, {"trace_id": "trace-1"})
        self.assertEqual(traces.calls[0]["request"]["top_k"], 40)
        self.assertNotIn("extra_body", traces.calls[0]["request"])

    def test_chat_generation_uses_messages_and_server_chat_template(self):
        requests = []
        traces = RecordingTraceCollector()
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(content="B) Dogru"),
                    finish_reason="length",
                )
            ]
        )
        client = OpenAICompatibleTextGenerationClient(
            base_url="https://example.test/v1",
            model="gemma",
            add_bos_token=True,
            generation_api="chat",
            trace_collector=traces,
        )
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **request: requests.append(request) or response
                )
            )
        )

        result = client.generate_texts(["Prompt"], max_tokens=32)[0]

        self.assertIsNone(result.error)
        self.assertEqual(result.text, "B) Dogru")
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(requests[0]["messages"], [{"role": "user", "content": "Prompt"}])
        self.assertNotIn("<bos>", requests[0]["messages"][0]["content"])
        self.assertEqual(result.api_call, {"trace_id": "trace-1"})
        self.assertEqual(traces.calls[0]["endpoint"], "chat")
        self.assertIn("/v1/chat/completions", traces.calls[0]["curl"])

    def test_chat_generation_flags_reasoning_only_responses(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(content="", reasoning_content="let me think" * 10),
                    finish_reason="length",
                )
            ]
        )
        client = OpenAICompatibleTextGenerationClient(
            base_url="https://example.test/v1",
            model="qwen",
            generation_api="chat",
        )
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_request: response)
            )
        )

        result = client.generate_chat_text("Prompt", 32, None, 0.0, None, None)

        self.assertIsNone(result.text)
        self.assertIn("--disable-thinking", result.error)
        self.assertEqual(result.reasoning_chars, 120)
        self.assertEqual(result.finish_reason, "length")

    def test_chat_generation_with_thinking_off_reads_an_answer_filed_as_reasoning(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(content="", reasoning_content="391"),
                    finish_reason="stop",
                )
            ]
        )
        client = OpenAICompatibleTextGenerationClient(
            base_url="https://example.test/v1",
            model="deepseek-v4-flash",
            generation_api="chat",
            default_extra_body=THINKING_DISABLED_EXTRA_BODY,
        )
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_request: response)
            )
        )

        result = client.generate_chat_text("Prompt", 32, None, 0.0, None, None)

        self.assertIsNone(result.error)
        self.assertEqual(result.text, "391")
        self.assertEqual(result.finish_reason, "stop")

    def test_chat_generation_counts_requests_lost_to_the_endpoint_not_to_reasoning(self):
        def create(**request):
            prompt = request["messages"][-1]["content"]
            if prompt == "lost":
                raise RuntimeError("Connection error.")
            message = (
                SimpleNamespace(content="", reasoning_content="x" * 50)
                if prompt == "thought"
                else SimpleNamespace(content="42", reasoning_content=None)
            )
            return SimpleNamespace(choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")])

        client = OpenAICompatibleTextGenerationClient(
            base_url="https://example.test/v1",
            model="reasoning-model",
            generation_api="chat",
        )
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        results = client.generate_texts(["ok", "lost", "thought"], max_tokens=8)

        self.assertEqual(client.request_tally.snapshot(), (3, 1))
        self.assertTrue(results[1].request_failed)
        self.assertTrue(results[2].error.startswith("Empty content after 50 characters"))
        self.assertFalse(results[2].request_failed)

    def test_streamed_chat_waits_at_most_the_idle_timeout_for_a_chunk(self):
        streamed = OpenAICompatibleTextGenerationClient(
            base_url="https://example.test/v1", model="m", generation_api="chat",
            stream=True, request_timeout=1800,
        )
        plain = OpenAICompatibleTextGenerationClient(
            base_url="https://example.test/v1", model="m", generation_api="chat", request_timeout=1800,
        )
        self.assertEqual(streamed.client._client.timeout.read, 600.0)
        self.assertEqual(plain.client._client.timeout.read, 1800.0)

    def test_a_dead_stream_is_sent_again(self):
        chunk = SimpleNamespace(
            id="c1", usage=None,
            choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content="391"), finish_reason="stop")],
        )
        calls = []

        def create(**_request):
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ReadTimeout("no chunk for 600 s")
            return iter([chunk])

        client = OpenAICompatibleTextGenerationClient(
            base_url="https://example.test/v1", model="m", generation_api="chat", stream=True,
        )
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        result = client.generate_chat_text("Prompt", 32, None, 0.0, None, None)

        self.assertIsNone(result.error)
        self.assertEqual(result.text, "391")
        self.assertEqual(len(calls), 2)

    def test_a_stream_that_stays_dead_fails_the_answer(self):
        def create(**_request):
            raise httpx.ReadTimeout("no chunk for 600 s")

        client = OpenAICompatibleTextGenerationClient(
            base_url="https://example.test/v1", model="m", generation_api="chat", stream=True,
        )
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        result = client.generate_chat_text("Prompt", 32, None, 0.0, None, None)

        self.assertIn("Chat completion request failed", result.error)

    def test_generated_request_count_deduplicates_batched_completion_calls(self):
        shared_call = {"request": "batch"}
        batched = [
            GeneratedTextResult(text="A", api_call=shared_call),
            GeneratedTextResult(text="B", api_call=shared_call),
        ]
        individual = [
            GeneratedTextResult(text="A", api_call={"request": "one"}),
            GeneratedTextResult(text="B", api_call={"request": "two"}),
        ]

        self.assertEqual(generated_request_count(batched), 1)
        self.assertEqual(generated_request_count(individual), 2)

if __name__ == "__main__":
    unittest.main()
