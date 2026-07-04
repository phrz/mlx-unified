# Copyright © 2026 Apple Inc.

import copy
import unittest

import mlx.core as mx

from mlx_lm.models import gemma, gemma2, gemma3_text
from mlx_lm.models.cache import make_prompt_cache


def tiny_gemma():
    args = gemma.ModelArgs(
        model_type="gemma",
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        head_dim=32,
        rms_norm_eps=1e-5,
        vocab_size=256,
        num_key_value_heads=2,
    )
    return gemma.Model(args), args


def tiny_gemma2():
    args = gemma2.ModelArgs(
        model_type="gemma2",
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        head_dim=32,
        rms_norm_eps=1e-5,
        vocab_size=256,
        num_key_value_heads=2,
    )
    return gemma2.Model(args), args


def tiny_gemma3():
    # sliding_window_pattern=2 alternates local/global; sliding_window=4 is
    # shorter than the test prompts so the band actually bites.
    args = gemma3_text.ModelArgs(
        model_type="gemma3_text",
        hidden_size=128,
        num_hidden_layers=4,
        intermediate_size=256,
        num_attention_heads=4,
        head_dim=32,
        rms_norm_eps=1e-5,
        vocab_size=256,
        num_key_value_heads=2,
        sliding_window=4,
        sliding_window_pattern=2,
    )
    return gemma3_text.Model(args), args


class TestUnifiedGemmaMask(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)

    def model_runner(self, model, args, model_type):
        self.assertEqual(len(model.layers), args.num_hidden_layers)
        self.assertEqual(model.model_type, model_type)

        inputs = mx.array([[0, 1]])
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (1, 2, args.vocab_size))

        cache = make_prompt_cache(model)
        outputs = model(inputs, cache=cache)
        self.assertEqual(outputs.shape, (1, 2, args.vocab_size))

        outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(outputs.shape, (1, 1, args.vocab_size))

        copy.deepcopy(model)

    def test_models(self):
        for tiny, model_type in (
            (tiny_gemma, "gemma"),
            (tiny_gemma2, "gemma2"),
            (tiny_gemma3, "gemma3_text"),
        ):
            model, args = tiny()
            self.model_runner(model, args, model_type)

    def prefix_mask_runner(self, model, args, prompt):
        inputs = mx.array([prompt])
        L = inputs.shape[1]
        baseline = model(inputs)

        # A causal-triangle prefix mask adds no edges, so the overlay must
        # reproduce the baseline exactly — this pins the row/column slicing.
        causal = mx.tril(mx.ones((L, L)))[None, None]
        model.model.set_visual_state(attention_mask_4d=causal)
        out = model(inputs, input_embeddings=model.model.embed_tokens(inputs))
        self.assertTrue(mx.allclose(out, baseline, atol=1e-5))

        # Whole-prompt bidirectional prefix mask (batch 1, no padding -> ones),
        # injected the server way: side state + merged input_embeddings.
        model.model.set_visual_state(attention_mask_4d=mx.ones((1, 1, L, L)))
        cache = make_prompt_cache(model)
        out = model(
            inputs, cache=cache, input_embeddings=model.model.embed_tokens(inputs)
        )
        self.assertEqual(out.shape, (1, L, args.vocab_size))
        # Future context reaches every position (later layers see changed keys).
        self.assertFalse(mx.allclose(out, baseline, atol=1e-4))
        # Decode steps after the prefill revert to normal causal behavior.
        step = model(mx.argmax(out[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(step.shape, (1, 1, args.vocab_size))
        model.model.reset_visual_state()

        # Reset must restore byte-identical text-only behavior.
        self.assertTrue(mx.array_equal(model(inputs), baseline))

    def test_gemma_prefix_mask(self):
        model, args = tiny_gemma()
        self.prefix_mask_runner(model, args, [3, 1, 4, 1, 5, 9])

    def test_gemma2_prefix_mask(self):
        model, args = tiny_gemma2()
        self.prefix_mask_runner(model, args, [3, 1, 4, 1, 5, 9])

    def test_gemma3_prefix_mask(self):
        # Prompt no longer than the sliding window, so the final-row equality
        # in the runner holds for local layers too.
        model, args = tiny_gemma3()
        self.prefix_mask_runner(model, args, [3, 1, 4, 1])

    def test_gemma3_prefix_mask_supersedes_sliding_window(self):
        # Longer than the sliding window: under the causal band the last query
        # cannot see the first keys; the vision override must let it (mlx-vlm
        # drives local layers with the same bidirectional mask).
        model, args = tiny_gemma3()
        inputs = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
        L = inputs.shape[1]
        baseline = model(inputs)

        model.model.set_visual_state(attention_mask_4d=mx.ones((1, 1, L, L)))
        out = model(inputs, input_embeddings=model.model.embed_tokens(inputs))
        model.model.reset_visual_state()

        self.assertFalse(mx.allclose(out[:, -1], baseline[:, -1], atol=1e-4))
        self.assertTrue(mx.array_equal(model(inputs), baseline))

    def test_paligemma_sanitize(self):
        for tiny in (tiny_gemma, tiny_gemma2):
            model, _ = tiny()
            weights = {
                "language_model.model.embed_tokens.weight": mx.zeros((1,)),
                "language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros(
                    (1,)
                ),
                "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros(
                    (1,)
                ),
                "language_model.lm_head.weight": mx.zeros((1,)),
                "vision_tower.vision_model.embeddings.patch_embedding.weight": mx.zeros(
                    (1,)
                ),
                "multi_modal_projector.linear.weight": mx.zeros((1,)),
            }
            self.assertEqual(
                set(model.sanitize(weights)),
                {
                    "model.embed_tokens.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                },
            )

    def test_paligemma_config_hoists_text_config(self):
        # HF PaliGemma configs omit head_dim/rms_norm_eps from text_config and
        # carry a stale top-level vocab_size — text_config values must win.
        args = gemma.ModelArgs.from_dict(
            {
                "model_type": "paligemma",
                "vocab_size": 257152,
                "image_token_index": 257152,
                "vision_config": {"model_type": "siglip_vision_model"},
                "text_config": {
                    "model_type": "gemma",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_key_value_heads": 1,
                    "vocab_size": 257216,
                },
            }
        )
        self.assertEqual(args.model_type, "paligemma")
        self.assertEqual(args.hidden_size, 2048)
        self.assertEqual(args.vocab_size, 257216)
        self.assertEqual(args.head_dim, 256)
        self.assertEqual(args.rms_norm_eps, 1e-6)

        args2 = gemma2.ModelArgs.from_dict(
            {
                "model_type": "paligemma",
                "text_config": {
                    "model_type": "gemma2",
                    "hidden_size": 2304,
                    "intermediate_size": 9216,
                    "num_attention_heads": 8,
                    "num_hidden_layers": 26,
                    "num_key_value_heads": 4,
                    "query_pre_attn_scalar": 256,
                    "vocab_size": 257216,
                },
            }
        )
        self.assertEqual(args2.hidden_size, 2304)
        self.assertEqual(args2.query_pre_attn_scalar, 256)
        self.assertEqual(args2.vocab_size, 257216)


if __name__ == "__main__":
    unittest.main()
