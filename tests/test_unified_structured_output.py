# Copyright © 2026 Apple Inc.
#
# mlx-unified: grammar-enforced response_format (mlx_lm/structured_output.py).
# Covers request validation (400 path), vocabulary building, the guided logits
# processor's masks against a tiny fake vocabulary, thinking passthrough, the
# index cache, and delegate forwarding — no checkpoint, no GPU.

import http.server
import json
import sys
import threading
import types
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import numpy as np
import requests

from mlx_lm.generate import TextStateMachine
from mlx_lm.server import (
    STRUCTURED_OUTPUT_CONTRACT_VERSION,
    APIHandler,
    GenerationContext,
    Response,
)
from mlx_lm.structured_output import (
    OUTLINES_AVAILABLE,
    GuidedLogitsProcessor,
    ResponseFormatSpec,
    StructuredIndexCache,
    StructuredOutputError,
    build_vocabulary,
    make_guided_processor,
    parse_response_format,
    token_bytes,
)

mx.set_default_device(mx.cpu)

VOCAB = {
    "{": 0,
    "}": 1,
    '"a"': 2,
    ":": 3,
    "1": 4,
    "2": 5,
    " ": 6,
    "<eos>": 7,
    "<think>": 8,
    "</think>": 9,
    "x": 10,
    "12": 11,
    '"': 12,
    "a": 13,
    "-": 14,
    "true": 15,
    "hello": 16,
}
EOS = 7
THINK_START = 8
THINK_END = 9
V = len(VOCAB) + 3  # logits wider than the tokenizer (padded lm_head)

SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}},
    "required": ["a"],
    "additionalProperties": False,
}


class FakeTokenizer:
    """The surface structured_output reads from mlx_lm's TokenizerWrapper."""

    def __init__(self, has_thinking=True, vocab=VOCAB):
        self._vocab = dict(vocab)
        self.eos_token_id = EOS
        self.eos_token_ids = {EOS}
        self.all_special_ids = [EOS, THINK_START, THINK_END]
        self.has_thinking = has_thinking
        self._think_start_tokens = (THINK_START,) if has_thinking else ()
        self._think_end_tokens = (THINK_END,) if has_thinking else ()

    def get_vocab(self):
        return self._vocab


def allowed_ids(processor, tokens):
    logits = mx.zeros((1, V))
    out = processor(mx.array(tokens), logits)
    finite = np.isfinite(np.array(out[0]))
    return set(np.flatnonzero(finite).tolist()), out


@unittest.skipUnless(OUTLINES_AVAILABLE, "outlines-core not installed")
class TestParseResponseFormat(unittest.TestCase):
    def test_absent_and_text_are_unconstrained(self):
        self.assertIsNone(parse_response_format(None))
        self.assertIsNone(parse_response_format({"type": "text"}))

    def test_json_object(self):
        spec = parse_response_format({"type": "json_object"})
        self.assertEqual(spec.kind, "json_object")
        self.assertTrue(spec.regex.startswith("\\{"))

    def test_json_schema(self):
        spec = parse_response_format(
            {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": SCHEMA, "strict": True},
            }
        )
        self.assertEqual(spec.kind, "json_schema")
        self.assertEqual(spec.name, "answer")
        self.assertIn('"a"', spec.regex)

    def test_rejects_malformed_values(self):
        cases = [
            ("a string", "must be an object"),
            ({"type": "xml"}, "unsupported response_format.type"),
            ({"type": "json_schema"}, "json_schema must be an object"),
            ({"type": "json_schema", "json_schema": {}}, "schema must be a JSON schema"),
            (
                {"type": "json_schema", "json_schema": {"schema": SCHEMA, "strict": "yes"}},
                "strict must be a boolean",
            ),
            (
                {"type": "json_schema", "json_schema": {"schema": SCHEMA, "name": 3}},
                "name must be a string",
            ),
            (
                {
                    "type": "json_schema",
                    "json_schema": {"schema": {"properties": {"a": {"$ref": "#/nope"}}}},
                },
                "unsupported json_schema schema",
            ),
        ]
        for value, message in cases:
            with self.subTest(value=value):
                with self.assertRaises(StructuredOutputError) as raised:
                    parse_response_format(value)
                self.assertIn(message, str(raised.exception))
                self.assertIsInstance(raised.exception, ValueError)


class TestTokenBytes(unittest.TestCase):
    def test_byte_level_alphabet(self):
        self.assertEqual(token_bytes("Ġhello", True), b" hello")
        self.assertEqual(token_bytes("Ċ", True), b"\n")
        self.assertEqual(token_bytes("Ã©", True), "é".encode("utf-8"))

    def test_sentencepiece_and_byte_fallback(self):
        self.assertEqual(token_bytes("▁hi", False), b" hi")
        self.assertEqual(token_bytes("<0x0A>", False), b"\n")
        self.assertEqual(token_bytes("plain", False), b"plain")

    @unittest.skipUnless(OUTLINES_AVAILABLE, "outlines-core not installed")
    def test_vocabulary_skips_special_tokens(self):
        vocab, max_id = build_vocabulary(FakeTokenizer(), EOS)
        self.assertEqual(max_id, max(VOCAB.values()))
        self.assertEqual(vocab.get_eos_token_id(), EOS)
        self.assertEqual(vocab.get(b"{"), [0])
        self.assertIsNone(vocab.get(b"<think>"))
        self.assertIsNone(vocab.get(b"<eos>"))


@unittest.skipUnless(OUTLINES_AVAILABLE, "outlines-core not installed")
class TestGuidedLogitsProcessor(unittest.TestCase):
    def processor(self, tokenizer=None, initial_state="normal", offer_thinking=True):
        tokenizer = tokenizer or FakeTokenizer()
        cache = StructuredIndexCache()
        spec = parse_response_format(
            {"type": "json_schema", "json_schema": {"schema": SCHEMA}}
        )
        (processor,) = make_guided_processor(
            cache,
            ("model", None),
            tokenizer,
            spec,
            initial_state=initial_state,
            offer_thinking=offer_thinking,
        )
        return processor

    def test_masks_follow_the_schema(self):
        p = self.processor(FakeTokenizer(has_thinking=False))
        # First call: only prompt tokens seen; the document must open with "{".
        allowed, out = allowed_ids(p, [10])
        self.assertEqual(allowed, {0})
        self.assertEqual(out.shape, (1, V))
        # After "{": the key (whole or split) or an optional space.
        allowed, _ = allowed_ids(p, [10, 0])
        self.assertEqual(allowed, {2, 6, 12})
        allowed, _ = allowed_ids(p, [10, 0, 2])
        self.assertEqual(allowed, {3, 6})
        allowed, _ = allowed_ids(p, [10, 0, 2, 3])
        self.assertEqual(allowed, {4, 5, 6, 11, 14})
        allowed, _ = allowed_ids(p, [10, 0, 2, 3, 4])
        self.assertEqual(allowed, {1, 4, 5, 6, 11})
        # Closing brace completes the document: only eos remains.
        allowed, _ = allowed_ids(p, [10, 0, 2, 3, 4, 1])
        self.assertEqual(allowed, {EOS})
        self.assertEqual(p.phase, "guided")
        allowed, _ = allowed_ids(p, [10, 0, 2, 3, 4, 1, EOS])
        self.assertEqual(allowed, {EOS})
        self.assertEqual(p.phase, "done")

    def test_mask_is_additive_and_keeps_allowed_logits(self):
        p = self.processor(FakeTokenizer(has_thinking=False))
        logits = mx.arange(V, dtype=mx.float32)[None]
        out = p(mx.array([10]), logits)
        self.assertEqual(float(out[0, 0]), 0.0)
        self.assertTrue(all(not np.isfinite(float(out[0, i])) for i in range(1, V)))

    def test_padded_logits_columns_are_masked(self):
        p = self.processor(FakeTokenizer(has_thinking=False))
        allowed = p.allowed_token_ids(V)
        self.assertEqual(len(allowed), V)
        self.assertFalse(allowed[len(VOCAB):].any())

    def test_thinking_model_may_open_a_think_block_first(self):
        p = self.processor()
        allowed, _ = allowed_ids(p, [10])
        self.assertEqual(allowed, {0, THINK_START})
        logits = mx.zeros((1, V))
        out = p(mx.array([10, THINK_START]), logits)
        self.assertEqual(p.phase, "reasoning")
        self.assertTrue(np.isfinite(np.array(out)).all())
        out = p(mx.array([10, THINK_START, 16]), logits)
        self.assertTrue(np.isfinite(np.array(out)).all())
        allowed, _ = allowed_ids(p, [10, THINK_START, 16, THINK_END])
        self.assertEqual(p.phase, "guided")
        self.assertEqual(allowed, {0})

    def test_a_two_token_think_start_offers_each_token_once(self):
        # Gemma 4 opens a think block with <|channel> then "thought". After the
        # first is generated the mask must offer the second, not the first
        # again: a mask cached on the guide state alone served the first
        # step's mask twice and the repeated opener then "left the grammar".
        tokenizer = FakeTokenizer()
        tokenizer._think_start_tokens = (THINK_START, 16)
        p = self.processor(tokenizer)
        allowed, _ = allowed_ids(p, [10])
        self.assertEqual(allowed, {0, THINK_START})
        allowed, _ = allowed_ids(p, [10, THINK_START])
        self.assertEqual(allowed, {0, 16})
        self.assertEqual(p.phase, "guided")
        allowed, _ = allowed_ids(p, [10, THINK_START, 16])
        self.assertEqual(p.phase, "reasoning")
        self.assertEqual(allowed, set(range(V)))

    def test_thinking_disabled_for_the_request_is_never_offered(self):
        p = self.processor(offer_thinking=False)
        allowed, _ = allowed_ids(p, [10])
        self.assertEqual(allowed, {0})
        with self.assertLogs(level="WARNING"):
            allowed, _ = allowed_ids(p, [10, THINK_START])
        self.assertEqual(allowed, {EOS})

    def test_declining_the_think_block_engages_the_grammar(self):
        p = self.processor()
        allowed_ids(p, [10])
        allowed, _ = allowed_ids(p, [10, 0])
        self.assertEqual(allowed, {2, 6, 12})

    def test_prompt_inside_a_think_block_is_free_until_it_closes(self):
        p = self.processor(initial_state="reasoning")
        logits = mx.zeros((1, V))
        self.assertEqual(p.phase, "reasoning")
        out = p(mx.array([10]), logits)
        self.assertTrue(np.isfinite(np.array(out)).all())
        out = p(mx.array([10, 16]), logits)
        self.assertTrue(np.isfinite(np.array(out)).all())
        allowed, _ = allowed_ids(p, [10, 16, THINK_END])
        self.assertEqual(allowed, {0})

    def test_no_thinking_tokens_means_no_free_phase(self):
        p = self.processor(FakeTokenizer(has_thinking=False), initial_state="reasoning")
        allowed, _ = allowed_ids(p, [10])
        self.assertEqual(allowed, {0})

    def test_token_outside_the_grammar_ends_the_document(self):
        p = self.processor(FakeTokenizer(has_thinking=False))
        allowed_ids(p, [10])
        with self.assertLogs(level="WARNING"):
            allowed, _ = allowed_ids(p, [10, 16])
        self.assertEqual(allowed, {EOS})

    def test_masks_are_cached_per_state(self):
        p = self.processor(FakeTokenizer(has_thinking=False))
        first = p.mask(V)
        self.assertIs(p.mask(V), first)

    def test_index_cache_reuses_compiled_index_per_model(self):
        cache = StructuredIndexCache(max_entries=2)
        spec = parse_response_format({"type": "json_object"})
        tokenizer = FakeTokenizer()
        index, _ = cache.index_for("m1", tokenizer, spec)
        self.assertIs(cache.index_for("m1", tokenizer, spec)[0], index)
        other = parse_response_format(
            {"type": "json_schema", "json_schema": {"schema": SCHEMA}}
        )
        self.assertIsNot(cache.index_for("m1", tokenizer, other)[0], index)
        # A different served model rebuilds the vocabulary and its indexes.
        self.assertIsNot(cache.index_for("m2", tokenizer, spec)[0], index)

    def test_no_format_means_no_processor(self):
        self.assertEqual(
            make_guided_processor(StructuredIndexCache(), "m", FakeTokenizer(), None),
            [],
        )


class StubResponseGenerator:
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
        self.captured = []
        self.raise_on_generate = None

    def generate(self, request, args, progress_callback=None):
        self.captured.append((request, args))
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        ctx = GenerationContext(
            has_tool_calling=False,
            has_thinking=False,
            tool_parser=None,
            text_sm=TextStateMachine(),
            initial_state="normal",
            prompt=[1, 2, 3],
        )
        return ctx, iter([Response('{"a":1}', 5, 0.0, "stop", ())])


@unittest.skipUnless(OUTLINES_AVAILABLE, "outlines-core not installed")
class TestServerWire(unittest.TestCase):
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
        self.response_generator.captured = []
        self.response_generator.raise_on_generate = None

    def post(self, body):
        return requests.post(f"http://localhost:{self.port}/v1/chat/completions", json=body)

    def test_contract_version(self):
        self.assertEqual(STRUCTURED_OUTPUT_CONTRACT_VERSION, 1)

    def test_valid_format_reaches_generation(self):
        response = self.post(
            {
                "model": "default_model",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": SCHEMA},
                },
            }
        )
        self.assertEqual(response.status_code, 200)
        _, args = self.response_generator.captured[-1]
        self.assertIsInstance(args.structured_output, ResponseFormatSpec)
        self.assertEqual(args.structured_output.kind, "json_schema")
        self.assertEqual(response.json()["choices"][0]["message"]["content"], '{"a":1}')

    def test_no_format_leaves_generation_unconstrained(self):
        response = self.post(
            {"model": "default_model", "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(response.status_code, 200)
        _, args = self.response_generator.captured[-1]
        self.assertIsNone(args.structured_output)

    def test_invalid_schema_is_a_400_before_generation(self):
        response = self.post(
            {
                "model": "default_model",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": {"properties": {"a": {"$ref": "#/nope"}}}},
                },
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported json_schema schema", response.json()["error"])
        self.assertEqual(self.response_generator.captured, [])

    def test_unknown_format_type_is_a_400(self):
        response = self.post(
            {
                "model": "default_model",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "xml"},
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported response_format.type", response.json()["error"])

    def test_generation_lane_refusal_is_a_400(self):
        self.response_generator.raise_on_generate = StructuredOutputError(
            "structured output is not supported for diffusion models"
        )
        response = self.post(
            {
                "model": "default_model",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("diffusion", response.json()["error"])


class TestDelegateForwarding(unittest.TestCase):
    def fake_mlx_vlm(self, stream_generate):
        mlx_vlm = types.ModuleType("mlx_vlm")
        generate = types.ModuleType("mlx_vlm.generate")
        generate.stream_generate = stream_generate
        mlx_vlm.generate = generate
        return {"mlx_vlm": mlx_vlm, "mlx_vlm.generate": generate}

    def delegate(self, is_diffusion=False):
        from mlx_lm.vlm_delegate import VlmDelegate

        processor = SimpleNamespace(tokenizer=SimpleNamespace())
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="qwen4_exp", eos_token_id=[])
        )
        delegate = VlmDelegate(model, processor, SimpleNamespace())
        delegate.is_diffusion = is_diffusion
        return delegate

    def test_stream_forwards_logits_processors(self):
        terminal = SimpleNamespace(
            text="",
            token=0,
            finish_reason="length",
            diffusion_block_complete=False,
            is_draft=False,
            draft_blocks=None,
            logprobs=None,
            cached_tokens=0,
        )
        stream_generate = mock.Mock(return_value=(r for r in [terminal]))
        sentinel = [object()]
        with mock.patch.dict(sys.modules, self.fake_mlx_vlm(stream_generate)):
            list(
                self.delegate().stream(
                    {"input_ids": mx.array([[1, 2, 3]])},
                    max_tokens=0,
                    temperature=0.0,
                    logits_processors=sentinel,
                )
            )
        self.assertIs(stream_generate.call_args.kwargs["logits_processors"], sentinel)

    def test_diffusion_refuses_logits_processors(self):
        with self.assertRaises(ValueError):
            list(
                self.delegate(is_diffusion=True).stream(
                    {"input_ids": mx.array([[1]])},
                    max_tokens=0,
                    temperature=0.0,
                    logits_processors=[object()],
                )
            )


if __name__ == "__main__":
    unittest.main()
