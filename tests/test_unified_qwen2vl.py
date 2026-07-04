# Copyright © 2025 Apple Inc.

import copy
import unittest

import mlx.core as mx
from mlx.utils import tree_map

from mlx_lm.models import qwen2_vl
from mlx_lm.models.cache import make_prompt_cache


def tiny_model(tie_word_embeddings=False):
    args = qwen2_vl.ModelArgs.from_dict(
        {
            "model_type": "qwen2_vl",
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "intermediate_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256,
            "rope_theta": 10000,
            "rope_scaling": {"type": "mrope", "mrope_section": [4, 6, 6]},
            "tie_word_embeddings": tie_word_embeddings,
        }
    )
    return qwen2_vl.Model(args), args


def grid_position_ids(text_before, grid_thw, text_after):
    """Qwen2-VL mrope positions for [text, one image run, text] — (3, 1, L) + delta."""
    t, h, w = grid_thw
    positions = [[], [], []]
    for axis in range(3):
        positions[axis].extend(range(text_before))
    for t_idx in range(t):
        for h_idx in range(h):
            for w_idx in range(w):
                positions[0].append(text_before + t_idx)
                positions[1].append(text_before + h_idx)
                positions[2].append(text_before + w_idx)
    tail_start = text_before + max(t, h, w)
    for axis in range(3):
        positions[axis].extend(range(tail_start, tail_start + text_after))
    length = text_before + t * h * w + text_after
    position_ids = mx.array(positions).reshape(3, 1, length)
    rope_deltas = mx.array([[tail_start + text_after - length]])
    return position_ids, rope_deltas


class TestUnifiedQwen2VL(unittest.TestCase):
    def test_model(self):
        vocab_size, num_layers = 256, 2
        model, _ = tiny_model()
        self.assertEqual(len(model.layers), num_layers)
        self.assertEqual(model.model_type, "qwen2_vl")

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

    def test_mrope_smoke(self):
        model, _ = tiny_model()
        inputs = mx.array([[3, 7, 9, 9, 9, 9, 11, 5]])
        baseline = model(inputs)

        # 2 text tokens, a 1x2x2 image run, 2 text tokens.
        position_ids, rope_deltas = grid_position_ids(2, (1, 2, 2), 2)
        self.assertEqual(position_ids.shape, (3, 1, 8))

        embeddings = model.model.embed_tokens(inputs)
        model.model.set_mrope_state(position_ids, rope_deltas)
        cache = make_prompt_cache(model)
        outputs = model(inputs, cache=cache, input_embeddings=embeddings)
        self.assertEqual(outputs.shape, (1, 8, 256))
        vision_prefill = outputs

        # Decode step exercises the sequential (cache_offset + delta) position path.
        outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(outputs.shape, (1, 1, 256))

        # Grid positions must actually change the result relative to 1D rope.
        self.assertFalse(mx.allclose(vision_prefill, baseline, atol=1e-5))

        # Reset restores byte-identical text-only behavior.
        model.model.reset_mrope_state()
        self.assertTrue(mx.array_equal(model(inputs), baseline))

    def test_mrope_degenerates_to_text_rope(self):
        model, _ = tiny_model()
        inputs = mx.array([[3, 7, 9, 4, 11, 5]])
        L = inputs.shape[1]

        text_cache = make_prompt_cache(model)
        text_out = model(inputs, cache=text_cache)
        text_step = model(mx.array([[2]]), cache=text_cache)

        # Equal positions on all three axes must reproduce plain 1D rope.
        position_ids = mx.broadcast_to(mx.arange(L).reshape(1, 1, L), (3, 1, L))
        model.model.set_mrope_state(position_ids, mx.zeros((1, 1), dtype=mx.int32))
        mm_cache = make_prompt_cache(model)
        mm_out = model(
            inputs, cache=mm_cache, input_embeddings=model.model.embed_tokens(inputs)
        )
        mm_step = model(mx.array([[2]]), cache=mm_cache)
        model.model.reset_mrope_state()

        self.assertTrue(mx.allclose(mm_out, text_out, rtol=1e-4, atol=1e-4))
        self.assertTrue(mx.allclose(mm_step, text_step, rtol=1e-4, atol=1e-4))

    def test_mrope_chunked_prefill(self):
        model, _ = tiny_model()
        inputs = mx.array([[3, 7, 9, 9, 9, 9, 11, 5]])
        position_ids, rope_deltas = grid_position_ids(2, (1, 2, 2), 2)
        embeddings = model.model.embed_tokens(inputs)

        model.model.set_mrope_state(position_ids, rope_deltas)
        cache = make_prompt_cache(model)
        single = model(inputs, cache=cache, input_embeddings=embeddings)

        chunked_cache = make_prompt_cache(model)
        model(inputs[:, :5], cache=chunked_cache, input_embeddings=embeddings[:, :5])
        chunked = model(
            inputs[:, 5:], cache=chunked_cache, input_embeddings=embeddings[:, 5:]
        )
        model.model.reset_mrope_state()

        self.assertTrue(
            mx.allclose(single[:, -1], chunked[:, -1], rtol=1e-4, atol=1e-4)
        )

    def test_sanitize(self):
        model, _ = tiny_model()
        x = mx.zeros((1,))
        weights = {
            "visual.patch_embed.proj.weight": x,
            "vision_tower.blocks.0.attn.qkv.weight": x,
            "model.visual.blocks.0.attn.qkv.weight": x,
            "model.layers.0.self_attn.q_proj.weight": x,
            "model.layers.0.self_attn.rotary_emb.inv_freq": x,
            "model.language_model.layers.1.mlp.up_proj.weight": x,
            "language_model.model.embed_tokens.weight": x,
            "lm_head.weight": x,
        }
        sanitized = model.sanitize(weights)
        self.assertEqual(
            sorted(sanitized),
            [
                "language_model.lm_head.weight",
                "language_model.model.embed_tokens.weight",
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "language_model.model.layers.1.mlp.up_proj.weight",
            ],
        )

        tied_model, _ = tiny_model(tie_word_embeddings=True)
        self.assertNotIn(
            "language_model.lm_head.weight", tied_model.sanitize(dict(weights))
        )


if __name__ == "__main__":
    unittest.main()
