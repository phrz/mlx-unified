# Copyright © 2026 Apple Inc.
"""mlx-unified tests for ernie4_5_moe: 3D multimodal RoPE (set_mrope_state) and
dual-expert token routing (set_visual_state) side state, plus VL-checkpoint
sanitize. The mrope positions are built with the bridge's own qwen-style
builder — ERNIE's position semantics are identical (text spans sequential on
all axes, image runs get (t, h, w) grid positions, cursor advances by
max(t, h, w))."""

import copy
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from mlx_lm.models import ernie4_5_moe
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.multimodal import build_qwen_image_mrope_state

IMAGE_TOKEN_ID = 250


def tiny_model(**overrides):
    config = dict(
        hidden_size=128,
        intermediate_size=128,
        model_type="ernie4_5_moe_vl",
        max_position_embeddings=1000,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=4,
        rms_norm_eps=1e-5,
        vocab_size=256,
        rope_theta=1000,
        use_bias=False,
        tie_word_embeddings=False,
        moe_num_experts=[4, 4],
        moe_intermediate_size=[64, 32],
        moe_k=2,
        moe_layer_interval=1,
        moe_layer_start_index=1,
        moe_num_shared_experts=1,
        # head_dim 32 → dim/2 = 16 rotary pairs, split 6/6/4 like 22/22/20.
        rope_scaling={"type": "default", "mrope_section": [6, 6, 4]},
    )
    config.update(overrides)
    args = ernie4_5_moe.ModelArgs.from_dict(config)
    return ernie4_5_moe.Model(args)


def image_prompt():
    """A 12-token prompt with one 4-token image run (grid 1x4x4, merge 2)."""
    tokens = mx.array([[1, 2, 3, 4] + [IMAGE_TOKEN_ID] * 4 + [5, 6, 7, 8]])
    state = build_qwen_image_mrope_state(
        input_ids=tokens,
        image_grid_thw=mx.array([[1, 4, 4]]),
        image_token_id=IMAGE_TOKEN_ID,
        spatial_merge_size=2,
    )
    token_types = (tokens == IMAGE_TOKEN_ID).astype(mx.int32)
    return tokens, state, token_types


class TestUnifiedErnie(unittest.TestCase):
    def test_model(self):
        # The model_test_runner contract from tests/test_models.py.
        model = tiny_model()
        vocab_size, num_layers = 256, 4
        self.assertEqual(len(model.layers), num_layers)
        self.assertEqual(model.model_type, "ernie4_5_moe_vl")

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

        inputs = mx.array([[0, 1], [2, 3]])
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (2, 2, vocab_size))

        copy.deepcopy(model)

    def test_single_expert_group_model(self):
        # Text-only checkpoint shape (moe_num_experts as a plain int).
        model = tiny_model(model_type="ernie4_5_moe", moe_num_experts=4)
        out = model(mx.array([[0, 1]]))
        self.assertEqual(out.shape, (1, 2, 256))
        self.assertFalse(
            any("switch_mlp_1" in k for k, _ in tree_flatten(model.parameters()))
        )

    def test_mrope_with_equal_axes_matches_text_rope(self):
        model = tiny_model()
        tokens = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
        baseline = model(tokens)

        # Sequential positions on all three axes must reproduce 1D RoPE exactly.
        L = tokens.shape[1]
        positions = mx.broadcast_to(mx.arange(L).reshape(1, 1, L), (3, 1, L))
        model.model.set_mrope_state(positions, mx.zeros((1, 1), dtype=mx.int32))
        embeddings = model.model.embed_tokens(tokens)
        with_state = model(tokens, input_embeddings=embeddings)
        model.model.reset_mrope_state()

        self.assertTrue(mx.allclose(baseline, with_state, rtol=1e-4, atol=1e-4))

    def test_reset_restores_text_only_behavior(self):
        model = tiny_model()
        tokens, state, token_types = image_prompt()
        baseline = model(tokens)

        model.model.set_mrope_state(state.position_ids, state.rope_deltas)
        model.model.set_visual_state(mm_token_type_ids=token_types)
        with_state = model(tokens, input_embeddings=model.model.embed_tokens(tokens))
        # Grid positions + mm-expert routing must actually change the forward.
        self.assertFalse(mx.allclose(baseline, with_state, rtol=1e-3, atol=1e-3))

        model.model.reset_mrope_state()
        model.model.reset_visual_state()
        self.assertTrue(mx.array_equal(baseline, model(tokens)))

    def test_all_text_token_types_are_inert(self):
        model = tiny_model()
        tokens = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
        baseline = model(tokens)
        model.model.set_visual_state(mm_token_type_ids=mx.zeros_like(tokens))
        with_state = model(tokens)
        model.model.reset_visual_state()
        self.assertTrue(mx.array_equal(baseline, with_state))

    def test_chunked_prefill_matches_single_prefill(self):
        model = tiny_model()
        tokens, state, token_types = image_prompt()
        embeddings = model.model.embed_tokens(tokens)
        next_token = mx.array([[7]])

        model.model.set_mrope_state(state.position_ids, state.rope_deltas)
        model.model.set_visual_state(mm_token_type_ids=token_types)

        cache = make_prompt_cache(model)
        model(tokens, cache=cache, input_embeddings=embeddings)
        single = model(next_token, cache=cache)

        # A chunk boundary INSIDE the image run: positions and token types must
        # be sliced by cache offset, then decode falls back to offset + delta.
        cache = make_prompt_cache(model)
        model(tokens[:, :5], cache=cache, input_embeddings=embeddings[:, :5])
        model(tokens[:, 5:], cache=cache, input_embeddings=embeddings[:, 5:])
        chunked = model(next_token, cache=cache)

        model.model.reset_mrope_state()
        model.model.reset_visual_state()

        self.assertTrue(mx.allclose(single, chunked, rtol=1e-4, atol=1e-4))

    def test_vl_checkpoint_sanitize_roundtrip(self):
        source = tiny_model()
        num_text_experts = 4

        # Reconstruct a raw-HF-style ERNIE VL checkpoint from the live weights:
        # vision tower + resampler + mtp extras, language weights under
        # language_model.*, per-expert (unstacked) MoE tensors with the mm group
        # appended after the text group, Paddle-transposed gates, gate.weight_1,
        # and both aux-free biases stacked under moe_statics.
        checkpoint = {
            "vision_tower.patch_embed.proj.weight": mx.zeros((4, 4)),
            "resampler_model.mlp.weight": mx.zeros((4, 4)),
            "language_model.model.mtp_block.linear.weight": mx.zeros((4, 4)),
        }
        params = dict(tree_flatten(source.parameters()))
        biases = {}
        for key, value in params.items():
            if ".mlp.switch_mlp." in key or ".mlp.switch_mlp_1." in key:
                offset = num_text_experts if ".switch_mlp_1." in key else 0
                base, _, tail = key.partition(".mlp.switch_mlp")
                proj = tail.split(".")[1]
                for e in range(value.shape[0]):
                    checkpoint[
                        f"language_model.{base}.mlp.experts.{offset + e}.{proj}.weight"
                    ] = value[e]
            elif key.endswith(".mlp.gate.weight"):
                checkpoint[f"language_model.{key}"] = value.T
            elif key.endswith(".mlp.gate_1.weight"):
                vl_key = key.replace(".gate_1.weight", ".gate.weight_1")
                checkpoint[f"language_model.{vl_key}"] = value.T
            elif key.endswith(".mlp.e_score_correction_bias") or key.endswith(
                ".mlp.e_score_correction_bias_1"
            ):
                base = key.replace("_1", "").replace(".e_score_correction_bias", "")
                biases.setdefault(base, [None, None])[key.endswith("_1")] = value
            else:
                checkpoint[f"language_model.{key}"] = value
        for base, (text_bias, mm_bias) in biases.items():
            checkpoint[
                f"language_model.{base}.moe_statics.e_score_correction_bias"
            ] = mx.stack([text_bias, mm_bias])

        restored = tiny_model()
        restored.load_weights(list(source.sanitize(checkpoint).items()), strict=True)

        tokens = mx.array([[3, 1, 4, 1, 5, 9]])
        self.assertTrue(mx.array_equal(source(tokens), restored(tokens)))


if __name__ == "__main__":
    unittest.main()
