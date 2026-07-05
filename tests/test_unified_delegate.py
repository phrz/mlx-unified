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
    _adapt_results,
    is_delegated_model_type,
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


def raw_result(text="", token=0, finish=None, block=False, draft=None):
    """A stand-in for mlx-vlm's GenerationResult (only the fields we read)."""
    return SimpleNamespace(
        text=text,
        token=token,
        finish_reason=finish,
        diffusion_block_complete=block,
        is_draft=draft is not None,
        draft_blocks=draft,
    )


class TestDelegateRegistry(unittest.TestCase):
    def test_detection(self):
        self.assertIn("diffusion_gemma", DELEGATED_VLM_FAMILIES)
        self.assertTrue(is_delegated_model_type("diffusion_gemma"))
        self.assertFalse(is_delegated_model_type("gemma4"))
        self.assertFalse(is_delegated_model_type("llama"))
        self.assertFalse(is_delegated_model_type(None))

    def test_model_provider_load_routes_to_delegate(self):
        provider = ModelProvider(make_cli_args())
        fake = SimpleNamespace(model="MODEL", tokenizer="TOKENIZER")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(
                json.dumps({"model_type": "diffusion_gemma"})
            )
            with mock.patch(
                "mlx_lm.server.load_delegate", return_value=fake
            ) as load_delegate:
                provider._load(d)
            load_delegate.assert_called_once()
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
            with mock.patch(
                "mlx_lm.server.load", return_value=(model, tokenizer)
            ) as load, mock.patch(
                "mlx_lm.server.load_delegate"
            ) as load_delegate:
                provider._load(d)
            load.assert_called_once()
            load_delegate.assert_not_called()
        self.assertIsNone(provider.delegate)
        self.assertIs(provider.model, model)


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
        raw = [raw_result(block=True, token=None), raw_result(finish="stop", token=None)]
        out = list(_adapt_results(raw))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].finish_reason, "stop")
        self.assertTrue(out[0].block_complete)
        self.assertEqual(out[0].text, "")


class FakeDelegate:
    def __init__(self, script):
        self.script = script
        self.calls = {}

    def render_chat(self, messages, tools=None, **template_kwargs):
        self.calls["render_chat"] = (messages, tools, template_kwargs)
        return "PROMPT", []

    def prepare(self, prompt, images):
        self.calls["prepare"] = (prompt, images)
        return {"input_ids": mx.array([[1, 2, 3]])}

    def stream(self, inputs, *, max_tokens, temperature, draft_blocks=False):
        self.calls["stream"] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "draft_blocks": draft_blocks,
        }
        yield from self.script


class TestServeDelegated(unittest.TestCase):
    """_serve_single routes delegated models to _serve_delegated_diffusion,
    driven directly (no HTTP)."""

    def serve(self, script, request=None, stream_draft_blocks=True, stop_words=()):
        rg = ResponseGenerator.__new__(ResponseGenerator)  # skip the worker thread
        fake = FakeDelegate(script)
        rg.model_provider = SimpleNamespace(
            delegate=fake,
            model="model",
            tokenizer="tokenizer",
            draft_model=None,
            cli_args=SimpleNamespace(chat_template_args={}),
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
            stream_draft_blocks=stream_draft_blocks,
            sampling=SimpleNamespace(temperature=0.0),
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
            sequences={(0,): ""},
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
            Response("", 0, "normal", None, 0.0, None, (), draft_blocks=["░░░░"]),
            Response("", 0, "normal", None, 0.0, None, (), draft_blocks=["Hel░"]),
            Response("Hel", 5, "normal", None, 0.0, None, ()),
            Response("", 0, "normal", None, 0.0, None, (), draft_blocks=["lo░"]),
            Response("lo", 6, "normal", None, 0.0, "stop", ()),
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


if __name__ == "__main__":
    unittest.main()
