# Copyright © 2026 Apple Inc.
#
# Serving delegation to mlx-vlm (mlx_lm/vlm_delegate.py): registry/detection,
# the ModelProvider._load routing, the GenerationResult adapter, the
# _serve_delegated_diffusion lane, and the x_stream_draft_blocks wire format —
# all with mocked engines, no checkpoint or mlx_vlm import needed.

import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import requests

from mlx_lm.generate import TextStateMachine
from mlx_lm.server import (
    APIHandler,
    GenerationContext,
    ModelProvider,
    Response,
    ResponseGenerator,
)
from mlx_lm.vlm_delegate import (
    DELEGATED_VLM_FAMILIES,
    DelegatedResponse,
    VlmDelegate,
    _adapt_results,
    _repair_glm5_next_namespaces,
    is_delegated_model_type,
    load_delegate,
)


def make_cli_args(**overrides):
    args = SimpleNamespace(
        model=None,
        adapter_path=None,
        draft_model=None,
        pipeline=False,
        trust_remote_code=False,
        chat_template="",
        use_default_chat_template=False,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def raw_result(text="", token=0, finish=None, block=False, draft=None, cached=0):
    """A stand-in for mlx-vlm's GenerationResult (only the fields we read)."""
    return SimpleNamespace(
        text=text,
        token=token,
        finish_reason=finish,
        diffusion_block_complete=block,
        is_draft=draft is not None,
        draft_blocks=draft,
        logprobs=None,
        cached_tokens=cached,
    )


class TestDelegateRegistry(unittest.TestCase):
    def test_detection(self):
        self.assertIn("diffusion_gemma", DELEGATED_VLM_FAMILIES)
        self.assertIn("glm5_next", DELEGATED_VLM_FAMILIES)
        self.assertIn("qwen4_exp", DELEGATED_VLM_FAMILIES)
        self.assertTrue(is_delegated_model_type("diffusion_gemma"))
        self.assertTrue(is_delegated_model_type("glm5_next"))
        self.assertTrue(is_delegated_model_type("qwen4_exp"))
        self.assertFalse(is_delegated_model_type("gemma4"))
        self.assertFalse(is_delegated_model_type("llama"))
        self.assertFalse(is_delegated_model_type(None))

    def test_model_provider_load_routes_to_delegate(self):
        for model_type in ("diffusion_gemma", "glm5_next", "qwen4_exp"):
            with self.subTest(model_type=model_type):
                provider = ModelProvider(make_cli_args())
                fake = SimpleNamespace(model="MODEL", tokenizer="TOKENIZER")
                with tempfile.TemporaryDirectory() as d:
                    (Path(d) / "config.json").write_text(
                        json.dumps({"model_type": model_type})
                    )
                    with mock.patch(
                        "mlx_lm.server.load_delegate", return_value=fake
                    ) as load_delegate:
                        provider._load(d)
                    load_delegate.assert_called_once_with(
                        Path(d),
                        prompt_cache_bytes=None,
                        prompt_cache_disk_bytes=None,
                    )
                self.assertIs(provider.delegate, fake)
                self.assertEqual(provider.model, "MODEL")
                self.assertEqual(provider.tokenizer, "TOKENIZER")
                self.assertFalse(provider.is_batchable)
                self.assertIsNone(provider.vision_encoder)

    def test_model_provider_rejects_adapters_and_drafts(self):
        provider = ModelProvider(make_cli_args())
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(
                json.dumps({"model_type": "diffusion_gemma"})
            )
            with self.assertRaises(ValueError):
                provider._load(d, adapter_path="adapter")
            with self.assertRaises(ValueError):
                provider._load(d, draft_model_path="draft")

    def test_non_delegated_model_type_untouched(self):
        provider = ModelProvider(make_cli_args())
        model = SimpleNamespace(layers=[])
        tokenizer = SimpleNamespace(chat_template=None)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(json.dumps({"model_type": "llama"}))
            with (
                mock.patch(
                    "mlx_lm.server.load", return_value=(model, tokenizer)
                ) as load,
                mock.patch("mlx_lm.server.load_delegate") as load_delegate,
            ):
                provider._load(d)
            load.assert_called_once()
            load_delegate.assert_not_called()
        self.assertIsNone(provider.delegate)
        self.assertIs(provider.model, model)

    def test_delegate_apc_uses_admitted_ram_ceiling(self):
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="qwen4_exp", eos_token_id=[])
        )
        processor = SimpleNamespace()
        manager = mock.Mock()
        with (
            mock.patch("mlx_lm.vlm_delegate._repair_diffusion_gemma_processor"),
            mock.patch("mlx_vlm.utils.load", return_value=(model, processor)),
            mock.patch("mlx_lm.utils.load_tokenizer", return_value="TOKENIZER"),
            mock.patch("mlx_vlm.apc.apc_disk_namespace", return_value="namespace"),
            mock.patch("mlx_vlm.apc.from_env", return_value=manager) as from_env,
        ):
            delegate = load_delegate(
                "/model",
                prompt_cache_bytes=4 * 1024**3,
                prompt_cache_disk_bytes=0,
            )

        from_env.assert_called_once_with(
            "namespace",
            overrides={"enabled": True, "max_resident_bytes": 4 * 1024**3},
        )
        delegate.close()
        manager.close.assert_called_once()

    def test_glm5_next_checkpoint_namespaces_are_remapped_once(self):
        class FakeModel:
            calls = []

            def sanitize(self, weights):
                self.calls.append(weights)
                return weights

        with mock.patch.dict("sys.modules", {}):
            import mlx_vlm.models.glm5_next.glm5_next as glm5

            original_model = glm5.Model
            glm5.Model = FakeModel
            try:
                _repair_glm5_next_namespaces()
                _repair_glm5_next_namespaces()
                model = FakeModel()
                marker = object()
                out = model.sanitize(
                    {
                        "vision_tower.post_layernorm.weight": marker,
                        "language_model.lm_head.weight": marker,
                        "language_model.model.layers.0.self_attn.f_a_proj.weight": marker,
                        "language_model.model.layers.0.self_attn.f_a_proj.scales": marker,
                        "language_model.model.layers.0.self_attn.f_a_proj.biases": marker,
                        "language_model.model.layers.0.self_attn.f_b_proj.weight": marker,
                        "language_model.model.layers.0.self_attn.f_b_proj.scales": marker,
                        "language_model.model.layers.0.self_attn.f_b_proj.biases": marker,
                    }
                )
                aliases = model.quantization_path_aliases(
                    "language_model.model.layers.0.self_attn."
                    "forget_gate.f_a_proj"
                )
            finally:
                glm5.Model = original_model

        self.assertIs(out["vision_model.post_layernorm.weight"], marker)
        self.assertIs(out["language_model.lm_head.weight"], marker)
        self.assertNotIn("vision_tower.post_layernorm.weight", out)
        for projection in ("f_a_proj", "f_b_proj"):
            for suffix in ("weight", "scales", "biases"):
                self.assertIs(
                    out[
                        "language_model.model.layers.0.self_attn."
                        f"forget_gate.{projection}.{suffix}"
                    ],
                    marker,
                )
        self.assertEqual(len(FakeModel.calls), 1)
        self.assertEqual(
            aliases,
            (
                "language_model.model.layers.0.self_attn.f_a_proj",
                "model.layers.0.self_attn.f_a_proj",
            ),
        )

    def test_glm5_next_quantization_uses_checkpoint_projection_path(self):
        class FakeModel:
            def sanitize(self, weights):
                return weights

        with mock.patch.dict("sys.modules", {}):
            import mlx_vlm.models.glm5_next.glm5_next as glm5
            from mlx_vlm.utils import _quantization_for_module_path

            original_model = glm5.Model
            glm5.Model = FakeModel
            try:
                _repair_glm5_next_namespaces()
                model = FakeModel()
                path = (
                    "language_model.model.layers.0.self_attn."
                    "forget_gate.f_a_proj"
                )
                expected = {"bits": 8, "group_size": 64, "mode": "affine"}
                quantization = {
                    "bits": 4,
                    "group_size": 64,
                    "mode": "affine",
                    "language_model.model.layers.0.self_attn.f_a_proj": expected,
                }
                actual = _quantization_for_module_path(quantization, path, model)
            finally:
                glm5.Model = original_model

        self.assertEqual(actual, expected)

    def test_glm5_next_load_bypasses_processor_model_config_parse(self):
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="glm5_next", eos_token_id=[2])
        )
        processor = SimpleNamespace()
        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d)
            (model_path / "config.json").write_text(
                json.dumps({"model_type": "glm5_next"})
            )
            with (
                mock.patch("mlx_lm.vlm_delegate._repair_glm5_next_namespaces"),
                mock.patch("mlx_vlm.utils.load") as combined_load,
                mock.patch("mlx_vlm.utils.load_model", return_value=model) as load_model,
                mock.patch(
                    "transformers.PreTrainedTokenizerFast.from_pretrained",
                    return_value=processor,
                ) as load_processor,
                mock.patch("mlx_lm.utils.load_tokenizer", return_value="TOKENIZER"),
            ):
                delegate = load_delegate(model_path)

        combined_load.assert_not_called()
        load_model.assert_called_once_with(model_path)
        load_processor.assert_called_once_with(model_path)
        self.assertIs(delegate.model, model)
        self.assertIs(delegate.processor, processor)
        self.assertEqual(delegate.tokenizer, "TOKENIZER")


class TestAdaptResults(unittest.TestCase):
    """_adapt_results mirrors mlx_vlm.server's _diffusion_block_chunks
    semantics under mlx_lm's per-token accounting."""

    def test_two_blocks_with_drafts(self):
        raw = [
            raw_result(draft=["░░"]),
            raw_result(draft=["░░"]),  # unchanged → deduped
            raw_result(draft=["a░"]),
            raw_result(text="a", token=1),
            raw_result(text="b", token=2),
            raw_result(block=True, token=2),
            raw_result(draft=["░░"]),  # canvas 2
            raw_result(text="c", token=3),
            raw_result(block=True, token=3),
            raw_result(text="!", token=3, finish="length"),  # finalize leftovers
        ]
        out = list(_adapt_results(raw))

        drafts = [r for r in out if r.draft_blocks is not None]
        committed = [r for r in out if r.draft_blocks is None]
        self.assertEqual([d.draft_blocks for d in drafts], [["░░"], ["a░"], ["░░"]])
        self.assertTrue(all(d.text == "" for d in drafts))
        # One response per committed token; the boundary markers and the final
        # finish result folded into each block's last token.
        self.assertEqual(
            [(r.text, r.block_complete, r.finish_reason) for r in committed],
            [("a", False, None), ("b", True, None), ("c!", True, "length")],
        )
        # Block 2's draft must come after block 1's committed flush
        # (identity, not ==: dataclass equality would find the earlier twin).
        positions = {id(r): i for i, r in enumerate(out)}
        self.assertLess(positions[id(committed[1])], positions[id(drafts[2])])

    def test_committed_text_identical_without_drafts(self):
        raw = [
            raw_result(text="a", token=1),
            raw_result(text="b", token=2),
            raw_result(block=True, token=2),
            raw_result(text="c", token=3),
            raw_result(block=True, token=3),
            raw_result(text="!", token=3, finish="length"),
        ]
        out = list(_adapt_results(raw))
        self.assertTrue(all(r.draft_blocks is None for r in out))
        self.assertEqual("".join(r.text for r in out), "abc!")
        self.assertEqual(len(out), 3)
        self.assertEqual(out[-1].finish_reason, "length")

    def test_duplicate_draft_across_canvases_deduped(self):
        raw = [
            raw_result(draft=["░"]),
            raw_result(text="a", token=1),
            raw_result(block=True, token=1),
            raw_result(draft=["░"]),  # identical fresh canvas → deduped
            raw_result(text="b", token=2),
            raw_result(block=True, token=2),
            raw_result(finish="stop", token=2),
        ]
        out = list(_adapt_results(raw))
        self.assertEqual(len([r for r in out if r.draft_blocks is not None]), 1)
        self.assertEqual("".join(r.text for r in out), "ab")

    def test_immediate_eos_yields_single_terminal_response(self):
        raw = [
            raw_result(block=True, token=None),
            raw_result(finish="stop", token=None),
        ]
        out = list(_adapt_results(raw))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].finish_reason, "stop")
        self.assertTrue(out[0].block_complete)
        self.assertEqual(out[0].text, "")

    def test_autoregressive_results_stream_per_token_and_fold_terminal_tail(self):
        raw = [
            raw_result(text="Hel", token=1),
            raw_result(text="lo", token=2),
            raw_result(text="!", token=2, finish="stop"),
        ]
        out = list(_adapt_results(raw, diffusion=False))

        self.assertEqual([r.text for r in out], ["Hel", "lo!"])
        self.assertTrue(all(r.block_complete for r in out))
        self.assertIsNone(out[0].finish_reason)
        self.assertEqual(out[1].finish_reason, "stop")

    def test_cached_tokens_survive_autoregressive_adaptation(self):
        out = list(
            _adapt_results(
                [
                    raw_result(text="x", token=1, cached=17),
                    raw_result(finish="stop", token=1, cached=17),
                ],
                diffusion=False,
            )
        )
        self.assertEqual(out[-1].cached_tokens, 17)


class TestExplicitCheckpointBoundary(unittest.TestCase):
    def test_marker_selects_template_stable_prefix_without_generation_suffix(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen4_exp"))
        delegate = VlmDelegate(model, SimpleNamespace(), SimpleNamespace())
        messages = [
            {"role": "system", "content": "rules"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "anchor",
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ],
            },
            {"role": "user", "content": "dynamic"},
        ]

        def fake_template(
            _processor, _config, rendered, add_generation_prompt=True, **_
        ):
            text = "".join(f"<{m['role']}>{m.get('content', '')}|" for m in rendered)
            return text + ("<assistant>" if add_generation_prompt else "")

        def fake_prepare(prompt, _images):
            return {"input_ids": mx.array([[ord(c) for c in prompt]])}

        delegate.prepare = fake_prepare
        with mock.patch(
            "mlx_vlm.prompt_utils.apply_chat_template", side_effect=fake_template
        ):
            full_prompt, images = delegate.render_chat(messages)
            full_inputs = delegate.prepare(full_prompt, images)
            boundary = delegate.explicit_checkpoint_len(messages, full_inputs)
            prefix_prompt, _ = delegate.render_chat(
                messages[:2], add_generation_prompt=False
            )

        self.assertEqual(boundary, len(prefix_prompt))
        self.assertNotIn("prompt_cache_breakpoint", prefix_prompt)

    def test_delegate_forwards_apc_manager_tenant_and_boundary(self):
        stopping = mock.Mock()
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(stopping_criteria=stopping)
        )
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="qwen4_exp", eos_token_id=[])
        )
        manager = mock.Mock()
        delegate = VlmDelegate(
            model,
            processor,
            SimpleNamespace(),
            apc_manager=manager,
        )
        terminal = raw_result(finish="length", cached=0)
        with mock.patch(
            "mlx_vlm.generate.stream_generate",
            return_value=(result for result in [terminal]),
        ) as stream_generate:
            list(
                delegate.stream(
                    {"input_ids": mx.array([[1, 2, 3]])},
                    max_tokens=0,
                    temperature=0.0,
                    apc_tenant="shared-prefix",
                    apc_checkpoint_len=2,
                    apc_ttl_seconds=1800,
                )
            )

        kwargs = stream_generate.call_args.kwargs
        self.assertIs(kwargs["apc_manager"], manager)
        self.assertEqual(kwargs["apc_tenant"], "shared-prefix")
        self.assertEqual(kwargs["apc_checkpoint_len"], 2)
        self.assertEqual(kwargs["apc_ttl_seconds"], 1800)


class FakeDelegate:
    def __init__(self, script):
        self.script = script
        self.calls = {}
        self.tokenizer = SimpleNamespace(
            has_thinking=False,
            has_tool_calling=False,
            tool_parser=None,
        )

    def render_chat(self, messages, tools=None, **template_kwargs):
        self.calls["render_chat"] = (messages, tools, template_kwargs)
        return "PROMPT", []

    def prepare(self, prompt, images):
        self.calls["prepare"] = (prompt, images)
        return {"input_ids": mx.array([[1, 2, 3]])}

    def explicit_checkpoint_len(
        self, messages, inputs, *, tools=None, **template_kwargs
    ):
        self.calls["explicit_checkpoint_len"] = (
            messages,
            inputs,
            tools,
            template_kwargs,
        )
        return 2

    def stream(self, inputs, *, max_tokens, temperature, draft_blocks=False, **kwargs):
        self.calls["stream"] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "draft_blocks": draft_blocks,
            **kwargs,
        }
        yield from self.script


class TestServeDelegated(unittest.TestCase):
    """_serve_single routes delegated models to _serve_delegated_diffusion,
    driven directly (no HTTP)."""

    def serve(
        self,
        script,
        request=None,
        stream_draft_blocks=True,
        stop_words=(),
        prompt_cache_key=None,
        prompt_cache_options=None,
    ):
        rg = ResponseGenerator.__new__(ResponseGenerator)  # skip the worker thread
        fake = FakeDelegate(script)
        rg.model_provider = SimpleNamespace(
            delegate=fake,
            model="model",
            tokenizer="tokenizer",
            draft_model=None,
            cli_args=SimpleNamespace(chat_template_args={}, prefill_step_size=2048),
        )
        request = request or SimpleNamespace(
            request_type="chat",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            vision=None,
        )
        args = SimpleNamespace(
            stop_words=list(stop_words),
            max_tokens=32,
            seed=None,
            chat_template_kwargs=None,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_options=prompt_cache_options,
            stream_draft_blocks=stream_draft_blocks,
            sampling=SimpleNamespace(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0),
            logits=SimpleNamespace(
                repetition_penalty=1.0,
                repetition_context_size=20,
                presence_penalty=0.0,
                presence_context_size=20,
                frequency_penalty=0.0,
                frequency_context_size=20,
            ),
        )
        rqueue = Queue()
        rg._serve_single((rqueue, request, args))
        items = []
        while (item := rqueue.get_nowait()) is not None:
            if isinstance(item, Exception):
                raise item
            items.append(item)
        self.assertTrue(rqueue.empty())
        return fake, items

    def script(self):
        return [
            DelegatedResponse("", 0, 0.0, False, None, ["░░"]),
            DelegatedResponse("Hel", 5, 0.0, False, None),
            DelegatedResponse("lo", 6, 0.0, True, None),
            DelegatedResponse("", 0, 0.0, False, None, ["!░"]),
            DelegatedResponse("!", 7, 0.0, True, "stop"),
        ]

    def test_chat_request_streams_blocks_and_drafts(self):
        fake, items = self.serve(self.script())

        ctx = items[0]
        self.assertIsInstance(ctx, GenerationContext)
        self.assertEqual(ctx.prompt, [1, 2, 3])  # from prepare(), for usage

        # The prompt was rendered and prepared by the delegate (mlx-vlm path).
        self.assertEqual(fake.calls["prepare"], ("PROMPT", []))
        self.assertTrue(fake.calls["stream"]["draft_blocks"])

        responses = items[1:]
        self.assertTrue(all(isinstance(r, Response) for r in responses))
        drafts = [r for r in responses if r.draft_blocks is not None]
        committed = [r for r in responses if r.draft_blocks is None]
        self.assertEqual([d.draft_blocks for d in drafts], [["░░"], ["!░"]])
        # Text only on block boundaries; the draft precedes its block's chunk.
        self.assertEqual([r.text for r in committed], ["", "Hello", "!"])
        self.assertEqual(committed[-1].finish_reason, "stop")
        self.assertLess(responses.index(drafts[1]), responses.index(committed[2]))

    def test_flag_off_disables_drafts(self):
        fake, _ = self.serve(
            [DelegatedResponse("x", 1, 0.0, True, "stop")], stream_draft_blocks=False
        )
        self.assertFalse(fake.calls["stream"]["draft_blocks"])

    def test_text_completion_skips_chat_rendering_and_drafts(self):
        request = SimpleNamespace(request_type="text", prompt="raw prompt", vision=None)
        fake, items = self.serve(
            [DelegatedResponse("ok", 1, 0.0, True, "stop")], request=request
        )
        self.assertNotIn("render_chat", fake.calls)
        self.assertEqual(fake.calls["prepare"], ("raw prompt", []))
        # Drafts are chat-only even when the flag is set.
        self.assertFalse(fake.calls["stream"]["draft_blocks"])
        self.assertEqual(items[1].text, "ok")

    def test_stop_word_truncates_block(self):
        _, items = self.serve(
            [
                DelegatedResponse("foo ", 1, 0.0, False, None),
                DelegatedResponse("STOP tail", 2, 0.0, True, None),
            ],
            stop_words=["STOP"],
        )
        committed = [r for r in items[1:] if r.draft_blocks is None]
        self.assertEqual("".join(r.text for r in committed), "foo ")
        self.assertEqual(committed[-1].finish_reason, "stop")

    def test_explicit_prime_forwards_boundary_tenant_and_cached_usage(self):
        request = SimpleNamespace(
            request_type="chat",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "stable",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                }
            ],
            tools=None,
            vision=None,
        )
        script = [DelegatedResponse("", 0, 0.0, True, "length", cached_tokens=2)]
        fake, items = self.serve(
            script,
            request=request,
            prompt_cache_key="shared-prefix",
            prompt_cache_options={"mode": "explicit", "ttl": "30m"},
        )
        self.assertIn("explicit_checkpoint_len", fake.calls)
        self.assertEqual(fake.calls["stream"]["apc_tenant"], "shared-prefix")
        self.assertEqual(fake.calls["stream"]["apc_checkpoint_len"], 2)
        self.assertEqual(fake.calls["stream"]["apc_ttl_seconds"], 1800)
        self.assertEqual(items[0].prompt_cache_count, 2)


class StubResponseGenerator:
    """The ResponseGenerator surface APIHandler touches, with a scripted
    response stream — exercises request parsing + wire formatting only."""

    def __init__(self):
        self.cli_args = SimpleNamespace(
            num_draft_tokens=3,
            max_tokens=64,
            temp=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            allowed_origins=["*"],
        )
        self.script = []
        self.captured = []

    def generate(self, request, args, progress_callback=None):
        self.captured.append((request, args))
        ctx = GenerationContext(
            has_tool_calling=False,
            has_thinking=False,
            tool_parser=None,
            text_sm=TextStateMachine(),
            initial_state="normal",
            prompt=[1, 2, 3],
        )
        return ctx, iter(list(self.script))


class TestDraftWireFormat(unittest.TestCase):
    """x_stream_draft_blocks parsing and the delta.x_draft_blocks chunk shape
    (byte-compatible with the patched mlx_vlm.server)."""

    @classmethod
    def setUpClass(cls):
        cls.response_generator = StubResponseGenerator()
        cls.httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join()

    def setUp(self):
        self.response_generator.script = [
            Response("", 0, 0.0, None, (), draft_blocks=["░░░░"]),
            Response("", 0, 0.0, None, (), draft_blocks=["Hel░"]),
            Response("Hel", 5, 0.0, None, ()),
            Response("", 0, 0.0, None, (), draft_blocks=["lo░"]),
            Response("lo", 6, 0.0, "stop", ()),
        ]
        self.response_generator.captured = []

    def post(self, body, path="/v1/chat/completions"):
        return requests.post(f"http://localhost:{self.port}{path}", json=body)

    def sse_chunks(self, response):
        response.encoding = "utf-8"  # SSE has no charset header; the wire is UTF-8
        chunks = []
        for line in response.text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[len("data: ") :]))
        return chunks

    def test_streaming_chat_emits_draft_chunks(self):
        response = self.post(
            {
                "model": "delegated",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "x_stream_draft_blocks": True,
                "stream_options": {"include_usage": True},
            }
        )
        self.assertEqual(response.status_code, 200)
        chunks = self.sse_chunks(response)

        _, args = self.response_generator.captured[-1]
        self.assertTrue(args.stream_draft_blocks)

        drafts = [
            c
            for c in chunks
            if c["choices"] and "x_draft_blocks" in c["choices"][0].get("delta", {})
        ]
        self.assertEqual(len(drafts), 3)
        for draft in drafts:
            choice = draft["choices"][0]
            # The exact shape the patched mlx_vlm.server emits.
            self.assertEqual(draft["object"], "chat.completion.chunk")
            self.assertIsNone(choice["finish_reason"])
            self.assertIsNone(choice["logprobs"])
            self.assertNotIn("content", choice["delta"])
            self.assertEqual(list(choice["delta"]), ["x_draft_blocks"])
            self.assertIsNone(draft["usage"])
        self.assertEqual(
            [d["choices"][0]["delta"]["x_draft_blocks"] for d in drafts],
            [["░░░░"], ["Hel░"], ["lo░"]],
        )

        # Committed content is untouched, and drafts precede it.
        content = [
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
            if c["choices"] and "x_draft_blocks" not in c["choices"][0].get("delta", {})
        ]
        self.assertEqual("".join(content), "Hello")
        self.assertEqual(chunks.index(drafts[0]), 0)

        # Draft chunks carry no token accounting: 2 committed tokens only.
        usage = [c for c in chunks if not c["choices"]][-1]["usage"]
        self.assertEqual(usage["completion_tokens"], 2)
        self.assertEqual(usage["prompt_tokens"], 3)

    def test_stream_without_flag_has_no_draft_chunks(self):
        self.response_generator.script = [
            r for r in self.response_generator.script if r.draft_blocks is None
        ]
        response = self.post(
            {
                "model": "delegated",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
        )
        self.assertEqual(response.status_code, 200)
        _, args = self.response_generator.captured[-1]
        self.assertFalse(args.stream_draft_blocks)
        self.assertNotIn("x_draft_blocks", response.text)
        content = [
            c["choices"][0]["delta"].get("content", "")
            for c in self.sse_chunks(response)
            if c["choices"]
        ]
        self.assertEqual("".join(content), "Hello")

    def test_non_stream_ignores_flag_and_drafts(self):
        # stream=False: the flag must not reach generation, and any draft
        # responses are invisible in the aggregate (no accounting either).
        response = self.post(
            {
                "model": "delegated",
                "messages": [{"role": "user", "content": "hi"}],
                "x_stream_draft_blocks": True,
            }
        )
        self.assertEqual(response.status_code, 200)
        _, args = self.response_generator.captured[-1]
        self.assertFalse(args.stream_draft_blocks)
        body = response.json()
        self.assertNotIn("x_draft_blocks", response.text)
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(body["usage"]["completion_tokens"], 2)

    def test_text_completions_tolerate_flag_without_draft_chunks(self):
        response = self.post(
            {
                "model": "delegated",
                "prompt": "hi",
                "stream": True,
                "x_stream_draft_blocks": True,
            },
            path="/v1/completions",
        )
        self.assertEqual(response.status_code, 200)
        # The flag parses (stream-only gating happens at generation; the wire
        # never carries drafts on a text_completion object).
        _, args = self.response_generator.captured[-1]
        self.assertTrue(args.stream_draft_blocks)
        self.assertNotIn("x_draft_blocks", response.text)
        text = [
            c["choices"][0].get("text", "")
            for c in self.sse_chunks(response)
            if c["choices"]
        ]
        self.assertEqual("".join(text), "Hello")

    def test_prompt_cache_extensions_reach_generation_unchanged(self):
        response = self.post(
            {
                "model": "delegated",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "stable",
                                "prompt_cache_breakpoint": {"mode": "explicit"},
                            }
                        ],
                    }
                ],
                "max_tokens": 0,
                "prompt_cache_key": "shared-prefix",
                "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            }
        )
        self.assertEqual(response.status_code, 200)
        request, args = self.response_generator.captured[-1]
        self.assertEqual(args.prompt_cache_key, "shared-prefix")
        self.assertEqual(args.prompt_cache_options, {"mode": "explicit", "ttl": "30m"})
        self.assertEqual(
            request.messages[0]["content"][0]["prompt_cache_breakpoint"],
            {"mode": "explicit"},
        )

    def test_prompt_cache_extensions_fail_closed(self):
        response = self.post(
            {
                "model": "delegated",
                "messages": [{"role": "user", "content": "hi"}],
                "prompt_cache_options": {"mode": "automatic"},
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("implicit or explicit", response.text)


if __name__ == "__main__":
    unittest.main()
