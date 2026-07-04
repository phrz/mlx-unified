# Copyright © 2026 Apple Inc.

import copy
import unittest

import mlx.core as mx

from mlx_lm.models import deepseekocr, unlimited_ocr
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.unlimited_ocr import RingSlidingKVCache

WINDOW = 4


def tiny_args(cls, **overrides):
    kwargs = dict(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_shared_experts=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        max_position_embeddings=512,
    )
    kwargs.update(overrides)
    return cls(**kwargs)


class TestUnifiedDsOCR(unittest.TestCase):
    def model_test_runner(self, model, model_type, vocab_size, num_layers):
        self.assertEqual(len(model.layers), num_layers)
        self.assertEqual(model.model_type, model_type)

        inputs = mx.array([[0, 1]])
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (1, 2, vocab_size))

        cache = make_prompt_cache(model)
        outputs = model(inputs, cache=cache)
        self.assertEqual(outputs.shape, (1, 2, vocab_size))

        outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(outputs.shape, (1, 1, vocab_size))

        outputs = model(mx.array([[0, 1], [2, 3]]))
        self.assertEqual(outputs.shape, (2, 2, vocab_size))

        copy.deepcopy(model)

    def test_deepseekocr(self):
        args = tiny_args(deepseekocr.ModelArgs)
        model = deepseekocr.Model(args)
        # first_k_dense_replace=1: layer 0 dense, the rest MoE.
        self.assertIsInstance(model.layers[0].mlp, deepseekocr.MLP)
        self.assertIsInstance(model.layers[1].mlp, deepseekocr.MoE)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_unlimited_ocr(self):
        args = tiny_args(unlimited_ocr.ModelArgs, sliding_window_size=WINDOW)
        model = unlimited_ocr.Model(args)
        cache = make_prompt_cache(model)
        self.assertTrue(all(isinstance(c, RingSlidingKVCache) for c in cache))
        self.assertEqual(cache[0].window_size, WINDOW)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_ring_cache_decode(self):
        args = tiny_args(unlimited_ocr.ModelArgs, sliding_window_size=WINDOW)
        model = unlimited_ocr.Model(args)
        prompt = mx.array([[3, 1, 4, 1, 5, 9]])
        prefill_length = prompt.shape[1]

        cache = make_prompt_cache(model)
        out = model(prompt, cache=cache)
        tok = mx.argmax(out[0, -1:, :], keepdims=True)
        for step in range(WINDOW + 3):
            out = model(tok, cache=cache)
            self.assertEqual(out.shape, (1, 1, args.vocab_size))
            tok = mx.argmax(out[0, -1:, :], keepdims=True)

        c = cache[0]
        self.assertEqual(c.prefill_length, prefill_length)
        self.assertEqual(c.offset, prefill_length + WINDOW + 3)
        # The retained KV never grows past prefill + window...
        keys, _ = c.state
        self.assertEqual(keys.shape[2], prefill_length + WINDOW)
        # ...and steady-state decode is mask-free (ring order is not causal).
        self.assertIsNone(c.make_mask(1))
        self.assertEqual(c._ring_pos, 3)
        self.assertFalse(c.is_trimmable())

    def test_ring_cache_matches_plain_before_window_fills(self):
        # Until the decode ring fills, R-SWA is exact full attention, so logits
        # must match a plain-KVCache run of the deepseekocr body.
        mx.random.seed(7)
        args = tiny_args(unlimited_ocr.ModelArgs, sliding_window_size=WINDOW)
        model = unlimited_ocr.Model(args)

        ring_cache = make_prompt_cache(model)
        plain_cache = [KVCache() for _ in model.layers]

        prompt = mx.array([[3, 1, 4, 1, 5, 9]])
        ring_out = model(prompt, cache=ring_cache)
        plain_out = model(prompt, cache=plain_cache)
        self.assertTrue(mx.allclose(ring_out, plain_out, atol=1e-5))

        tok = mx.argmax(ring_out[0, -1:, :], keepdims=True)
        for _ in range(WINDOW - 1):
            ring_out = model(tok, cache=ring_cache)
            plain_out = model(tok, cache=plain_cache)
            self.assertTrue(mx.allclose(ring_out, plain_out, atol=1e-5))
            tok = mx.argmax(ring_out[0, -1:, :], keepdims=True)

    def test_input_embeddings_injection_and_reset(self):
        # Plain-injection arch: no side state to set. Injecting the model's own
        # embeddings must reproduce the token forward exactly; injecting
        # vision-like embeddings must not disturb later text-only forwards.
        mx.random.seed(0)
        args = tiny_args(deepseekocr.ModelArgs)
        model = deepseekocr.Model(args)

        text_inputs = mx.array([[10, 11, 12, 13]])
        baseline = model(text_inputs)
        injected = model(
            text_inputs, input_embeddings=model.model.embed_tokens(text_inputs)
        )
        self.assertTrue(mx.array_equal(baseline, injected))

        tokens = mx.array([[10, 11, 250, 250, 250, 250, 12, 13]])
        embeddings = model.model.embed_tokens(tokens)
        image_mask = (tokens == 250)[..., None]
        embeddings = mx.where(
            image_mask, mx.random.normal(embeddings.shape) * 0.02, embeddings
        )

        cache = make_prompt_cache(model)
        out = model(tokens, cache=cache, input_embeddings=embeddings)
        self.assertEqual(out.shape, (1, tokens.shape[1], args.vocab_size))
        out = model(mx.argmax(out[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(out.shape, (1, 1, args.vocab_size))

        after = model(text_inputs)
        self.assertTrue(mx.array_equal(baseline, after))

    def test_sanitize(self):
        args = tiny_args(deepseekocr.ModelArgs, num_hidden_layers=2)
        model = deepseekocr.Model(args)

        # mlx-vlm-converted layout: language_model prefix, top-level vision keys.
        converted = {
            "language_model.model.embed_tokens.weight": mx.zeros((1,)),
            "language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros((1,)),
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight": mx.zeros((1,)),
            "language_model.model.norm.weight": mx.zeros((1,)),
            "language_model.lm_head.weight": mx.zeros((1,)),
            "vision_model.embeddings.patch_embedding.weight": mx.zeros((1,)),
            "sam_model.blocks.0.attn.qkv.weight": mx.zeros((1,)),
            "projector.layers.weight": mx.zeros((1,)),
            "image_newline": mx.zeros((1,)),
            "view_separator": mx.zeros((1,)),
        }
        self.assertEqual(
            set(model.sanitize(converted)),
            {
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.1.mlp.switch_mlp.gate_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
            },
        )

        # Raw HF layout: model.-prefixed vision keys, per-expert MoE weights.
        raw = {
            "model.embed_tokens.weight": mx.zeros((1,)),
            "model.norm.weight": mx.zeros((1,)),
            "lm_head.weight": mx.zeros((1,)),
            "model.vision_model.embeddings.patch_embedding.weight": mx.zeros((1,)),
            "model.sam_model.blocks.0.attn.qkv.weight": mx.zeros((1,)),
            "model.projector.layers.weight": mx.zeros((1,)),
            "model.image_newline": mx.zeros((1,)),
            "model.view_seperator": mx.zeros((1,)),
            "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((1,)),
        }
        for e in range(args.n_routed_experts):
            raw[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = mx.zeros((4, 8))
        sanitized = model.sanitize(raw)
        self.assertEqual(
            set(sanitized),
            {
                "model.embed_tokens.weight",
                "model.norm.weight",
                "lm_head.weight",
                "model.layers.1.mlp.switch_mlp.gate_proj.weight",
            },
        )
        self.assertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"].shape,
            (args.n_routed_experts, 4, 8),
        )


if __name__ == "__main__":
    unittest.main()
