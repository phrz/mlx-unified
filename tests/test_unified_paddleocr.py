# Copyright © 2026 Apple Inc.

import copy
import unittest

import mlx.core as mx

from mlx_lm.models import paddleocr_vl
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.multimodal import build_qwen_image_mrope_state

IMAGE_TOKEN_ID = 250


def tiny_model():
    args = paddleocr_vl.ModelArgs(
        model_type="paddleocr_vl",
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=256,
        rope_theta=500000.0,
        rope_scaling={"type": "default", "mrope_section": [4, 6, 6]},
    )
    return paddleocr_vl.Model(args), args


class TestUnifiedPaddleOCR(unittest.TestCase):
    def test_model(self):
        model, args = tiny_model()
        self.assertEqual(len(model.layers), args.num_hidden_layers)
        self.assertEqual(model.model_type, "paddleocr_vl")

        inputs = mx.array([[0, 1]])
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (1, 2, args.vocab_size))

        cache = make_prompt_cache(model)
        outputs = model(inputs, cache=cache)
        self.assertEqual(outputs.shape, (1, 2, args.vocab_size))

        outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(outputs.shape, (1, 1, args.vocab_size))

        outputs = model(mx.array([[0, 1], [2, 3]]))
        self.assertEqual(outputs.shape, (2, 2, args.vocab_size))

        copy.deepcopy(model)

    def test_mrope_sequential_matches_plain_rope(self):
        # With sequential positions on all three axes and delta 0, sectioned
        # mrope must reproduce plain 1D RoPE (both prefill and cached decode).
        model, args = tiny_model()
        inputs = mx.array([[3, 1, 4, 1, 5, 9]])
        L = inputs.shape[1]

        plain_cache = make_prompt_cache(model)
        plain_prefill = model(inputs, cache=plain_cache)
        plain_decode = model(mx.array([[7]]), cache=plain_cache)

        position_ids = mx.broadcast_to(mx.arange(L).reshape(1, 1, L), (3, 1, L))
        model.model.set_mrope_state(position_ids, mx.zeros((1, 1), dtype=mx.int32))
        mrope_cache = make_prompt_cache(model)
        mrope_prefill = model(inputs, cache=mrope_cache)
        mrope_decode = model(mx.array([[7]]), cache=mrope_cache)
        model.model.reset_mrope_state()

        self.assertTrue(mx.allclose(plain_prefill, mrope_prefill, atol=1e-4))
        self.assertTrue(mx.allclose(plain_decode, mrope_decode, atol=1e-4))

    def test_vision_injection_and_reset(self):
        # The full server-shaped flow: build qwen-style grid positions for an
        # image run, inject merged embeddings via input_embeddings with mrope
        # side state set, decode a step, then reset and confirm text-only
        # behavior is byte-identical to a never-touched forward.
        model, args = tiny_model()
        text_inputs = mx.array([[10, 11, 12, 13]])
        baseline = model(text_inputs)

        # 2 text tokens + a 2x4x4 patch grid merged 2x2 -> run of 4 + 2 text.
        tokens = [10, 11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12, 13]
        input_ids = mx.array([tokens])
        state = build_qwen_image_mrope_state(
            input_ids=input_ids,
            image_grid_thw=mx.array([[1, 4, 4]]),
            image_token_id=IMAGE_TOKEN_ID,
            spatial_merge_size=2,
        )
        self.assertEqual(state.position_ids.shape, (3, 1, len(tokens)))

        embeddings = model.model.embed_tokens(input_ids)
        # Stand-in for vision features: overwrite the image run's embeddings.
        image_mask = (input_ids == IMAGE_TOKEN_ID)[..., None]
        embeddings = mx.where(
            image_mask, mx.random.normal(embeddings.shape) * 0.02, embeddings
        )

        model.model.set_mrope_state(state.position_ids, state.rope_deltas)
        cache = make_prompt_cache(model)
        out = model(input_ids, cache=cache, input_embeddings=embeddings)
        self.assertEqual(out.shape, (1, len(tokens), args.vocab_size))
        out = model(mx.argmax(out[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(out.shape, (1, 1, args.vocab_size))
        model.model.reset_mrope_state()

        after_reset = model(text_inputs)
        self.assertTrue(mx.array_equal(baseline, after_reset))

    def test_chunked_prefill_positions(self):
        # Splitting a vision prompt across prefill chunks must produce the
        # same logits as a single-shot prefill (stored positions are sliced by
        # cache offset).
        model, args = tiny_model()
        tokens = [10, 11, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12, 13]
        input_ids = mx.array([tokens])
        state = build_qwen_image_mrope_state(
            input_ids=input_ids,
            image_grid_thw=mx.array([[1, 4, 4]]),
            image_token_id=IMAGE_TOKEN_ID,
            spatial_merge_size=2,
        )
        embeddings = model.model.embed_tokens(input_ids)

        model.model.set_mrope_state(state.position_ids, state.rope_deltas)
        cache = make_prompt_cache(model)
        single = model(input_ids, cache=cache, input_embeddings=embeddings)

        cache = make_prompt_cache(model)
        model(input_ids[:, :5], cache=cache, input_embeddings=embeddings[:, :5])
        chunked = model(
            input_ids[:, 5:], cache=cache, input_embeddings=embeddings[:, 5:]
        )
        model.model.reset_mrope_state()

        self.assertTrue(mx.allclose(single[:, 5:], chunked, atol=1e-4))

    def test_sanitize(self):
        model, _ = tiny_model()
        weights = {
            "language_model.model.embed_tokens.weight": mx.zeros((1,)),
            "language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros((1,)),
            "language_model.lm_head.weight": mx.zeros((1,)),
            "visual.embeddings.patch_embedding.weight": mx.zeros((1,)),
            "visual.projector.0.weight": mx.zeros((1,)),
        }
        sanitized = model.sanitize(weights)
        self.assertEqual(
            set(sanitized),
            {
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "lm_head.weight",
            },
        )

        # Raw (pre-mlx-vlm) checkpoint layout.
        raw = {
            "model.norm.weight": mx.zeros((1,)),
            "lm_head.weight": mx.zeros((1,)),
            "visual.vision_model.head.probe": mx.zeros((1,)),
            "mlp_AR.0.weight": mx.zeros((1,)),
            "visual.packing_position_embedding.weight": mx.zeros((1,)),
        }
        self.assertEqual(
            set(model.sanitize(raw)), {"model.norm.weight", "lm_head.weight"}
        )


if __name__ == "__main__":
    unittest.main()
