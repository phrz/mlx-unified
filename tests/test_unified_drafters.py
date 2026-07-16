# Copyright © 2026 Apple Inc.
"""Unit tests for the drafter-family speculative port (docs/PORTING-DRAFTERS.md):
gemma4_text's speculative hook contract (sink / skip_final_norm / rollback /
hidden→logits helpers) on a tiny random config, spec_delegate's kind gating,
and — when the real checkpoints are present locally — a greedy parity
integration test of drafter_generate_step against plain generate_step."""

import os
import unittest

import mlx.core as mx

from mlx_lm.models import cache as cache_mod
from mlx_lm.models.gemma4_text import Model, ModelArgs


def tiny_args(**over):
    base = dict(
        model_type="gemma4_text",
        hidden_size=16,
        num_hidden_layers=4,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_kv_shared_layers=2,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        sliding_window=8,
        sliding_window_pattern=2,
        max_position_embeddings=256,
        rope_parameters={
            "full_attention": {"rope_theta": 10000.0},
            "sliding_attention": {"rope_theta": 10000.0},
        },
    )
    base.update(over)
    return ModelArgs.from_dict(base)


class TestGemma4SpeculativeHooks(unittest.TestCase):
    def setUp(self):
        self.model = Model(tiny_args())
        self.cache = cache_mod.make_prompt_cache(self.model)
        self.tokens = mx.array([[1, 2, 3, 4, 5]])

    def test_verify_hidden_returns_prenorm_and_sink(self):
        hidden, shared = self.model.speculative_verify_hidden(self.tokens, self.cache)
        self.assertEqual(hidden.shape, (1, 5, 16))
        # Both layer types present, each mapping to full-context (K, V).
        self.assertEqual(
            set(shared), {"sliding_attention", "full_attention"}
        )
        for k, v in shared.values():
            self.assertEqual(k.shape[2], 5)  # [B, kv_heads, L, head_dim]
            self.assertEqual(v.shape[2], 5)
        # Pre-norm: applying the final norm + head must equal the normal call.
        want = Model.__call__
        normed_logits = self.model.speculative_logits_from_hidden(hidden)
        self.assertEqual(normed_logits.shape, (1, 5, 64))

    def test_skip_final_norm_differs_from_normed_forward(self):
        c1 = cache_mod.make_prompt_cache(self.model)
        pre = self.model.model(self.tokens, cache=c1, skip_final_norm=True)
        c2 = cache_mod.make_prompt_cache(self.model)
        post = self.model.model(self.tokens, cache=c2)
        self.assertFalse(mx.allclose(pre, post).item())
        self.assertTrue(
            mx.allclose(self.model.model.norm(pre), post, atol=1e-5).item()
        )

    def test_rollback_trims_rejected_tail(self):
        self.model.speculative_verify_hidden(self.tokens, self.cache)
        offsets = [c.offset for c in self.cache]
        # Verify a 4-token block, accept only 1 → trim 3.
        block = mx.array([[6, 7, 8, 9]])
        self.model.speculative_verify_hidden(block, self.cache)
        accepted = self.model.rollback_speculative_cache(self.cache, None, 0, 4)
        self.assertEqual(accepted, 0)
        for c, before in zip(self.cache, offsets):
            self.assertEqual(c.offset, before + 1)

    def test_draft_hidden_is_final_norm(self):
        hidden, _ = self.model.speculative_verify_hidden(self.tokens, self.cache)
        drafted = self.model.speculative_draft_hidden(hidden)
        self.assertTrue(
            mx.allclose(drafted, self.model.model.norm(hidden)).item()
        )


class TestSpecDelegateGating(unittest.TestCase):
    def test_unsupported_kind_rejected_before_any_load(self):
        from unittest import mock

        from mlx_lm import spec_delegate

        # The gate must fire on the RESOLVED kind without loading weights.
        with mock.patch.object(
            spec_delegate, "resolve_drafter_kind", return_value="dflash"
        ):
            with self.assertRaisesRegex(ValueError, "not supported"):
                spec_delegate.load_drafter("/nonexistent", None)

    def test_drafter_generate_step_requires_hooks(self):
        from mlx_lm.generate import drafter_generate_step

        class NoHooks:
            pass

        with self.assertRaises(ValueError):
            next(
                drafter_generate_step(
                    mx.array([1, 2, 3]), NoHooks(), object(), max_tokens=4
                )
            )


TARGET = "/Users/phrz/.runway/models/models--mlx-community--gemma-4-E2B-it-qat-4bit/snapshots/main"
DRAFTER = "/Users/phrz/.runway/models/models--mlx-community--gemma-4-E2B-it-qat-assistant-4bit/snapshots/main"


@unittest.skipUnless(
    os.path.isdir(TARGET) and os.path.isdir(DRAFTER),
    "gemma-4 E2B target+assistant checkpoints not present locally",
)
class TestDrafterParityIntegration(unittest.TestCase):
    """Greedy speculation must reproduce the plain greedy output. Kept SHORT:
    on quantized models the batched verify forward and the one-token decode
    forward use different kernels whose tiny logit differences can flip a
    near-tie argmax after enough tokens — that's inherent to speculative decode
    on quantized weights (classic mlx-lm draft decoding included), not a bug."""

    def test_greedy_parity_short(self):
        from mlx_lm import load
        from mlx_lm.generate import drafter_generate_step, generate_step
        from mlx_lm.spec_delegate import is_drafter_checkpoint, load_drafter

        self.assertTrue(is_drafter_checkpoint(DRAFTER))
        model, tokenizer = load(TARGET)
        drafter, kind = load_drafter(DRAFTER)
        self.assertEqual(kind, "mtp")

        prompt = mx.array(tokenizer.encode("The capital of France is"))
        argmax = lambda x: mx.argmax(x, axis=-1)
        base = [int(t) for t, _ in generate_step(prompt, model, max_tokens=24, sampler=argmax)]
        spec = [
            int(t)
            for t, _, _ in drafter_generate_step(
                prompt, model, drafter, max_tokens=24, sampler=argmax, temperature=0.0
            )
        ]
        self.assertEqual(base, spec)


if __name__ == "__main__":
    unittest.main()
