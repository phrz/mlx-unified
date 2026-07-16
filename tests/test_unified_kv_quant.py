# Copyright © 2026 Apple Inc.
"""KV-cache quantization on sliding-window (gemma) models: rotating caches are
skipped (bounded at window_size — nothing to win) while the unbounded
full-attention layers quantize, and KV-shared layers dispatch quantized SDPA
via their DONOR's cache when that donor was converted."""

import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step, maybe_quantize_kv_cache
from mlx_lm.models import cache as cache_mod
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache
from mlx_lm.models.gemma4_text import Model

from .test_unified_drafters import tiny_args


def quant_args():
    # mx.quantize needs group_size >= 32 dividing head_dim — the drafter tests'
    # head_dim=8 config can't quantize, so these tests widen the heads.
    return tiny_args(head_dim=32, hidden_size=64, intermediate_size=128)


class TestSlidingWindowKvQuant(unittest.TestCase):
    def test_maybe_quantize_skips_rotating_converts_full(self):
        model = Model(quant_args())
        caches = cache_mod.make_prompt_cache(model)
        model.speculative_verify_hidden(mx.array([[1, 2, 3, 4, 5, 6]]), caches)

        maybe_quantize_kv_cache(caches, quantized_kv_start=1, kv_group_size=32, kv_bits=8)

        kinds = [type(c).__name__ for c in caches]
        self.assertIn("QuantizedKVCache", kinds)  # full_attention converted
        self.assertIn("RotatingKVCache", kinds)  # sliding kept, NOT raised on
        for c, lt in zip(caches, model.args.layer_types):
            if lt == "full_attention":
                self.assertIsInstance(c, QuantizedKVCache)
            else:
                self.assertIsInstance(c, RotatingKVCache)

    def test_generation_with_kv_bits_on_shared_layer_model(self):
        # tiny_args has num_kv_shared_layers=2: layer 3 (full_attention, shared)
        # consumes the quantized donor's packed triples — the dispatch fix's path.
        model = Model(quant_args())
        prompt = mx.array([3, 1, 4, 1, 5, 9, 2, 6])
        toks = []
        for tok, _ in generate_step(
            prompt,
            model,
            max_tokens=12,
            kv_bits=8,
            kv_group_size=32,
            quantized_kv_start=2,
            sampler=lambda x: mx.argmax(x, axis=-1),
        ):
            toks.append(int(tok))
        self.assertEqual(len(toks), 12)  # survived quantization mid-generation

    def test_quantized_matches_fp16_greedy_at_8bit(self):
        # 8-bit KV round-trip is near-lossless on a tiny model — greedy tokens
        # should match the unquantized run.
        model = Model(quant_args())
        prompt = mx.array([3, 1, 4, 1, 5, 9])
        argmax = lambda x: mx.argmax(x, axis=-1)
        base = [int(t) for t, _ in generate_step(prompt, model, max_tokens=8, sampler=argmax)]
        quant = [
            int(t)
            for t, _ in generate_step(
                prompt,
                model,
                max_tokens=8,
                kv_bits=8,
                kv_group_size=32,
                quantized_kv_start=2,
                sampler=argmax,
            )
        ]
        self.assertEqual(base, quant)


if __name__ == "__main__":
    unittest.main()
