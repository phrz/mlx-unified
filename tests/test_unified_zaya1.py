# Copyright © 2026 Apple Inc.
#
# mlx-unified: zaya1_vl text-side support in models/zaya1_vl.py — CCA attention,
# Mixture-of-Depths MoE with EDA router state, residual scaling, and
# visual_pos_masks-gated vision-LoRA deltas (set_visual_state/reset_visual_state).

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from mlx_lm.models import zaya1_vl
from mlx_lm.models.cache import make_prompt_cache


def tiny_args(**overrides):
    params = {
        "model_type": "zaya1_vl",
        "vocab_size": 256,
        "hidden_size": 128,
        "ffn_hidden_size": 128,
        "num_hidden_layers": 2,
        "num_experts": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_query_groups": 2,
        "head_dim": 32,
        "max_position_embeddings": 512,
        "norm_epsilon": 1e-5,
        "moe_router_topk": 1,
        "zaya_mlp_expansion": 32,
        "zaya_use_mod": True,
        "zaya_use_eda": True,
        "gated_linear_unit": True,
        "scale_residual_merge": True,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "cca_time0": 2,
        "cca_time1": 2,
        "tie_word_embeddings": True,
        "vision_lora": True,
        "vision_lora_rank_attn": 4,
        "vision_lora_rank_mlp": 8,
    }
    params.update(overrides)
    return zaya1_vl.ModelArgs.from_dict(params)


def randomize(model):
    # temp / balancing biases / residual scales initialize to exact zeros and ones;
    # random values make the attention scores and MoD routing non-degenerate.
    mx.random.seed(7)
    model.update(
        tree_map(lambda p: mx.random.normal(p.shape).astype(p.dtype) * 0.1, model.parameters())
    )


class TestUnifiedZaya1(unittest.TestCase):
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

    def test_model(self):
        args = tiny_args()
        model = zaya1_vl.Model(args)
        randomize(model)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_model_topk2(self):
        args = tiny_args(moe_router_topk=2)
        model = zaya1_vl.Model(args)
        randomize(model)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_cached_prefill_matches_uncached(self):
        # The CCA aux cache (conv overhang + delayed hidden state) must make a
        # cached full prefill match the cache-free forward.
        model = zaya1_vl.Model(tiny_args())
        randomize(model)
        inputs = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
        expected = model(inputs)
        outputs = model(inputs, cache=make_prompt_cache(model))
        self.assertTrue(mx.allclose(outputs, expected, atol=1e-5))

    def test_input_embeddings(self):
        # Only the embedding source changes: feeding embed_tokens output through
        # input_embeddings must reproduce the token forward exactly.
        model = zaya1_vl.Model(tiny_args())
        randomize(model)
        inputs = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
        expected = model(inputs)
        embeds = model.model.embed_tokens(inputs)
        outputs = model(inputs, input_embeddings=embeds)
        self.assertTrue(mx.array_equal(outputs, expected))

    def test_visual_lora_gating_and_reset(self):
        model = zaya1_vl.Model(tiny_args())
        randomize(model)
        inner = model.model
        inputs = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
        seq_len = inputs.shape[1]
        baseline = model(inputs)

        # image run at positions 2..5 with spliced (here: random) embeddings
        visual_mask = mx.array([[False, False, True, True, True, True, False, False]])
        embeds = mx.where(
            visual_mask[..., None],
            mx.random.normal((1, seq_len, 128)) * 0.1,
            inner.embed_tokens(inputs),
        )

        inner.set_visual_state(visual_pos_masks=visual_mask)
        gated = model(inputs, input_embeddings=embeds)
        self.assertEqual(gated.shape, baseline.shape)
        self.assertFalse(mx.array_equal(gated, baseline))

        # an all-False mask must leave every LoRA delta inert
        inner.set_visual_state(visual_pos_masks=mx.zeros((1, seq_len), dtype=mx.bool_))
        self.assertTrue(mx.allclose(model(inputs), baseline, atol=1e-6))

        # state cleared must restore byte-identical text-only behavior
        inner.reset_visual_state()
        self.assertTrue(mx.array_equal(model(inputs), baseline))

    def test_visual_lora_chunked_prefill_and_decode(self):
        # The anchor must keep chunked prefill aligned with the full-prompt mask,
        # and decode steps past the prompt must be LoRA-free (prefill-only).
        model = zaya1_vl.Model(tiny_args())
        randomize(model)
        inner = model.model
        inputs = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
        seq_len = inputs.shape[1]

        visual_mask = mx.array([[False, True, True, True, True, False, False, False]])
        embeds = mx.where(
            visual_mask[..., None],
            mx.random.normal((1, seq_len, 128)) * 0.1,
            inner.embed_tokens(inputs),
        )

        inner.set_visual_state(visual_pos_masks=visual_mask)
        cache = make_prompt_cache(model)
        single = model(inputs, cache=cache, input_embeddings=embeds)
        decode_single = model(mx.array([[7]]), cache=cache)

        inner.set_visual_state(visual_pos_masks=visual_mask)
        cache = make_prompt_cache(model)
        model(inputs[:, :5], cache=cache, input_embeddings=embeds[:, :5])
        chunked = model(inputs[:, 5:], cache=cache, input_embeddings=embeds[:, 5:])
        decode_chunked = model(mx.array([[7]]), cache=cache)
        inner.reset_visual_state()

        self.assertTrue(mx.allclose(single[:, -1], chunked[:, -1], atol=1e-4))
        self.assertTrue(mx.allclose(decode_single, decode_chunked, atol=1e-4))

    def test_sanitize_mlx_vlm_layout(self):
        # An mlx-vlm conversion stores the text body under language_model.* with
        # stacked experts and mlx-layout convs, alongside a vision tower; sanitize
        # must strip the vision side, remap, and drop the materialized tied head.
        model = zaya1_vl.Model(tiny_args())
        flat = dict(tree_flatten(model.parameters()))
        weights = {f"language_model.{k}": v for k, v in flat.items()}
        weights["language_model.lm_head.weight"] = flat["model.embed_tokens.weight"]
        weights["vision_tower.blocks.0.attn.qkv.weight"] = mx.zeros((4, 4))
        weights["vision_tower.merger.mlp.0.weight"] = mx.zeros((4, 4))

        sanitized = model.sanitize(weights)
        self.assertEqual(set(sanitized.keys()), set(flat.keys()))
        model.load_weights(list(sanitized.items()))

    def test_sanitize_raw_layout(self):
        # A raw checkpoint keeps model.* at the top level with per-expert
        # local_experts and torch-layout (C_out, C_in/groups, kernel) convs.
        args = tiny_args()
        model = zaya1_vl.Model(args)
        flat = dict(tree_flatten(model.parameters()))
        weights = {}
        for k, v in flat.items():
            if ".experts." in k and k.endswith(".weight"):
                prefix, name = k.split(".experts.")
                name = name[: -len(".weight")]
                for e in range(args.num_experts):
                    weights[f"{prefix}.experts.local_experts.{e}.{name}.weight"] = v[e]
            elif ".conv_qk." in k and k.endswith(".weight"):
                weights[k] = v.transpose(0, 2, 1)
            else:
                weights[k] = v
        weights["lm_head.weight"] = flat["model.embed_tokens.weight"]
        weights["vision_tower.patch_embed.proj.weight"] = mx.zeros((4, 4))

        sanitized = model.sanitize(weights)
        self.assertEqual(set(sanitized.keys()), set(flat.keys()))
        model.load_weights(list(sanitized.items()))


if __name__ == "__main__":
    unittest.main()
