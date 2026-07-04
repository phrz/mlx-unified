# Copyright © 2026 Apple Inc.

import copy
import unittest

import mlx.core as mx
from mlx.utils import tree_map

from mlx_lm.models import hunyuan_vl
from mlx_lm.models.cache import make_prompt_cache


def tiny_model(tie_word_embeddings=False):
    args = hunyuan_vl.ModelArgs.from_dict(
        {
            "model_type": "hunyuan_vl",
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "intermediate_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "rms_norm_eps": 1e-5,
            "vocab_size": 256,
            "rope_theta": 10000,
            "use_qk_norm": True,
            "rope_scaling": {
                "type": "xdrope",
                "alpha": 50.0,
                "xdrope_section": [4, 4, 4, 4],
            },
            "tie_word_embeddings": tie_word_embeddings,
        }
    )
    return hunyuan_vl.Model(args), args


def xdrope_position_ids(text_before, grid_hw, text_after):
    """Hunyuan-VL xdrope positions for [text, <begin>, grid-with-row-newlines,
    <end>, text] — (4, 1, L) in the processor's axis order (p, w, h, t).

    The p axis stays the raw sequential index everywhere; only the
    patch_h x (patch_w + 1) grid tokens (patch rows + a newline slot per row)
    get w/h grid positions and t = 0. Begin/end tokens keep sequential
    positions on all axes, and the rope delta is always 0.
    """
    patch_h, patch_w = grid_hw
    run = patch_h * (patch_w + 1)
    length = text_before + 1 + run + 1 + text_after
    p = list(range(length))
    w = list(range(length))
    h = list(range(length))
    t = list(range(length))
    start = text_before + 1
    for row in range(patch_h):
        for col in range(patch_w + 1):
            idx = start + row * (patch_w + 1) + col
            w[idx] = col
            h[idx] = row
            t[idx] = 0
    return mx.array([p, w, h, t]).reshape(4, 1, length)


class TestUnifiedHunyuan(unittest.TestCase):
    def test_model(self):
        vocab_size, num_layers = 256, 2
        model, _ = tiny_model()
        self.assertEqual(len(model.layers), num_layers)
        self.assertEqual(model.model_type, "hunyuan_vl")

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

    def test_xdrope_smoke(self):
        model, _ = tiny_model()
        # 2 text tokens, <begin>, a 2x2 grid (+1 newline slot per row = 6 image
        # tokens), <end>, 2 text tokens.
        inputs = mx.array([[3, 7, 8, 9, 9, 9, 9, 9, 9, 10, 11, 5]])
        baseline = model(inputs)

        position_ids = xdrope_position_ids(2, (2, 2), 2)
        self.assertEqual(position_ids.shape, (4, 1, 12))

        embeddings = model.model.embed_tokens(inputs)
        model.model.set_mrope_state(position_ids, mx.zeros((1, 1), dtype=mx.int32))
        cache = make_prompt_cache(model)
        outputs = model(inputs, cache=cache, input_embeddings=embeddings)
        self.assertEqual(outputs.shape, (1, 12, 256))
        vision_prefill = outputs

        # Decode step exercises the sequential (cache_offset + delta) position path.
        outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(outputs.shape, (1, 1, 256))

        # Grid positions must actually change the result relative to 1D rope.
        self.assertFalse(mx.allclose(vision_prefill, baseline, atol=1e-5))

        # Reset restores byte-identical text-only behavior.
        model.model.reset_mrope_state()
        self.assertTrue(mx.array_equal(model(inputs), baseline))

    def test_xdrope_degenerates_to_text_rope(self):
        model, _ = tiny_model()
        inputs = mx.array([[3, 7, 9, 4, 11, 5]])
        L = inputs.shape[1]

        text_cache = make_prompt_cache(model)
        text_out = model(inputs, cache=text_cache)
        text_step = model(mx.array([[2]]), cache=text_cache)

        # Equal positions on all four axes must reproduce plain NTK-alpha 1D rope.
        position_ids = mx.broadcast_to(mx.arange(L).reshape(1, 1, L), (4, 1, L))
        model.model.set_mrope_state(position_ids, mx.zeros((1, 1), dtype=mx.int32))
        mm_cache = make_prompt_cache(model)
        mm_out = model(
            inputs, cache=mm_cache, input_embeddings=model.model.embed_tokens(inputs)
        )
        mm_step = model(mx.array([[2]]), cache=mm_cache)
        model.model.reset_mrope_state()

        self.assertTrue(mx.allclose(mm_out, text_out, rtol=1e-4, atol=1e-4))
        self.assertTrue(mx.allclose(mm_step, text_step, rtol=1e-4, atol=1e-4))

    def test_xdrope_chunked_prefill(self):
        model, _ = tiny_model()
        inputs = mx.array([[3, 7, 8, 9, 9, 9, 9, 9, 9, 10, 11, 5]])
        position_ids = xdrope_position_ids(2, (2, 2), 2)
        rope_deltas = mx.zeros((1, 1), dtype=mx.int32)
        embeddings = model.model.embed_tokens(inputs)

        model.model.set_mrope_state(position_ids, rope_deltas)
        cache = make_prompt_cache(model)
        single = model(inputs, cache=cache, input_embeddings=embeddings)

        # A chunk boundary inside the image span must not change the result.
        chunked_cache = make_prompt_cache(model)
        model(inputs[:, :6], cache=chunked_cache, input_embeddings=embeddings[:, :6])
        chunked = model(
            inputs[:, 6:], cache=chunked_cache, input_embeddings=embeddings[:, 6:]
        )
        model.model.reset_mrope_state()

        self.assertTrue(
            mx.allclose(single[:, -1], chunked[:, -1], rtol=1e-4, atol=1e-4)
        )

    def test_sanitize(self):
        model, _ = tiny_model()
        x = mx.zeros((1,))
        weights = {
            "vit.patch_embed.proj.weight": x,
            "vision_tower.blocks.0.attn.qkv.weight": x,
            "model.vit.blocks.0.attn.qkv.weight": x,
            "model.layers.0.self_attn.q_proj.weight": x,
            "model.layers.0.self_attn.rotary_emb.inv_freq": x,
            "language_model.model.layers.1.mlp.up_proj.weight": x,
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
