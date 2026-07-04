# Copyright © 2026 Apple Inc.
import copy
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from mlx_lm.models.cache import make_prompt_cache


def tiny_moondream2_args():
    from mlx_lm.models import moondream2

    return moondream2.ModelArgs(
        model_type="moondream2",
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        vocab_size=1000,
        num_attention_heads=4,
        num_key_value_heads=4,
    )


def tiny_moondream3_args():
    from mlx_lm.models import moondream3

    return moondream3.ModelArgs(
        model_type="moondream3",
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        vocab_size=1000,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        rope_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        moe_start_layer=1,
    )


def prefix_attention_mask(seq_len, prefix_len):
    """The mask moondream's get_input_embeddings emits: bos + image tokens form a
    bidirectional prefix, everything else is causal."""
    mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1)
    mask[:prefix_len, :prefix_len] = 0.0
    return mask.reshape(1, 1, seq_len, seq_len)


class TestUnifiedMoondream(unittest.TestCase):

    def model_test_runner(self, model, model_type, vocab_size, num_layers):
        self.assertEqual(len(model.layers), num_layers)
        self.assertEqual(model.model_type, model_type)

        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs = model(inputs)
            self.assertEqual(outputs.shape, (1, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            cache = make_prompt_cache(model)
            outputs = model(inputs, cache=cache)
            self.assertEqual(outputs.shape, (1, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
            self.assertEqual(outputs.shape, (1, 1, vocab_size))
            self.assertEqual(outputs.dtype, t)

        # Test batch size > 1
        inputs = mx.array([[0, 1], [2, 3]])
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (2, 2, vocab_size))

        # Make sure the model can be copied / pickled
        copy.deepcopy(model)

    def visual_state_test_runner(self, model, vocab_size):
        """set_visual_state(attention_mask_4d=...) must change prefill attention,
        survive cached decode past the mask, and reset to byte-identical text-only
        behavior."""
        inputs = mx.array([[0, 1, 2, 3, 4, 5]])
        seq_len, prefix_len = inputs.shape[1], 3
        baseline = model(inputs)
        embeds = model.model.embed_tokens(inputs)

        model.model.set_visual_state(
            attention_mask_4d=prefix_attention_mask(seq_len, prefix_len)
        )
        masked = model(inputs, input_embeddings=embeds)
        self.assertEqual(masked.shape, (1, seq_len, vocab_size))
        # The bidirectional prefix rewires prefix rows, so logits must diverge.
        self.assertFalse(mx.allclose(masked, baseline))

        # A prefix of just bos is exactly the causal mask — logits must match.
        model.model.set_visual_state(
            attention_mask_4d=prefix_attention_mask(seq_len, 1)
        )
        degenerate = model(inputs, input_embeddings=embeds)
        self.assertTrue(mx.allclose(degenerate, baseline, atol=1e-5))

        model.model.set_visual_state(
            attention_mask_4d=prefix_attention_mask(seq_len, prefix_len)
        )

        # Cached prefill inside the mask, then a decode step past its end (the
        # causal fallback path).
        cache = make_prompt_cache(model)
        cached = model(inputs, cache=cache, input_embeddings=embeds)
        self.assertTrue(mx.allclose(cached, masked, atol=1e-5))
        step = model(mx.array([[6]]), cache=cache)
        self.assertEqual(step.shape, (1, 1, vocab_size))

        model.model.reset_visual_state()
        restored = model(inputs)
        self.assertTrue(mx.array_equal(restored, baseline))

    def sanitize_test_runner(self, model, checkpoint_weights):
        expected = {k for k, _ in tree_flatten(model.parameters())}
        self.assertEqual(set(model.sanitize(checkpoint_weights)), expected)

    def test_moondream2(self):
        from mlx_lm.models import moondream2

        args = tiny_moondream2_args()
        model = moondream2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_moondream2_visual_state(self):
        from mlx_lm.models import moondream2

        model = moondream2.Model(tiny_moondream2_args())
        self.visual_state_test_runner(model, 1000)

    def test_moondream2_sanitize(self):
        from mlx_lm.models import moondream2

        model = moondream2.Model(tiny_moondream2_args())
        params = dict(tree_flatten(model.parameters()))

        # mlx-vlm conversion layout: text.* beside a vision tower.
        vlm = {
            "text.model." + k.removeprefix("model.")
            if k.startswith("model.")
            else "text." + k: v
            for k, v in params.items()
        }
        vlm["vision.encoder.patch_emb.weight"] = mx.zeros((1,))
        self.sanitize_test_runner(model, vlm)

        # Raw vikhyatk/moondream2 layout.
        raw = {}
        for k, v in params.items():
            if k == "model.embed_tokens.weight":
                k = "text_model.transformer.embd.wte.weight"
            elif k.startswith("model.layers."):
                k = "text_model.transformer.h." + k.removeprefix("model.layers.")
                k = k.replace(".attn.qkv.", ".mixer.Wqkv.")
                k = k.replace(".attn.proj.", ".mixer.out_proj.")
            elif k.startswith("model.post_ln."):
                k = "text_model.lm_head.ln." + k.removeprefix("model.post_ln.")
            else:
                k = "text_model.lm_head.linear." + k.removeprefix("lm_head.")
            raw[k] = v
        raw["vision_encoder.encoder.model.visual.norm.weight"] = mx.zeros((1,))
        raw["region_model.coordinate_encoder.weight"] = mx.zeros((1,))
        self.sanitize_test_runner(model, raw)

    def test_moondream3(self):
        from mlx_lm.models import moondream3

        args = tiny_moondream3_args()
        model = moondream3.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_moondream3_visual_state(self):
        from mlx_lm.models import moondream3

        model = moondream3.Model(tiny_moondream3_args())
        self.visual_state_test_runner(model, 1000)

    def test_moondream3_tau_is_position_dependent(self):
        """A nonzero Tau alpha must make attention position-dependent through the
        cache offset (the temperature term sees absolute positions)."""
        from mlx_lm.models import moondream3

        model = moondream3.Model(tiny_moondream3_args())
        params = dict(tree_flatten(model.parameters()))
        alpha_keys = [k for k in params if k.endswith("attn.tau.alpha")]
        self.assertTrue(alpha_keys)
        model.update(
            tree_unflatten([(k, mx.ones_like(params[k]) * 5.0) for k in alpha_keys])
        )

        token = mx.array([[7]])
        cache = make_prompt_cache(model)
        first = model(token, cache=cache)
        second = model(token, cache=cache)
        # Same token, different absolute position: tau_pos differs, so must logits.
        self.assertFalse(mx.allclose(first[:, -1], second[:, -1]))

    def test_moondream3_sanitize(self):
        from mlx_lm.models import moondream3

        model = moondream3.Model(tiny_moondream3_args())
        params = dict(tree_flatten(model.parameters()))

        # mlx-vlm conversion layout.
        vlm = {
            "text.model." + k.removeprefix("model.")
            if k.startswith("model.")
            else "text." + k: v
            for k, v in params.items()
        }
        vlm["vision.encoder.patch_emb.weight"] = mx.zeros((1,))
        self.sanitize_test_runner(model, vlm)

        # Raw moondream3 layout: everything under "model.", bare wte tensor.
        raw = {}
        for k, v in params.items():
            if k == "model.wte.weight":
                k = "model.text.wte"
            elif k.startswith("model."):
                k = "model.text." + k.removeprefix("model.")
            else:
                k = "model.text." + k
            raw[k] = v
        raw["model.vision.blocks.0.attn.qkv.weight"] = mx.zeros((1,))
        raw["model.region.coord_encoder.weight"] = mx.zeros((1,))
        self.sanitize_test_runner(model, raw)


if __name__ == "__main__":
    unittest.main()
