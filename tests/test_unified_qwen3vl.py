# Copyright © 2026 Apple Inc.
#
# mlx-unified: qwen3_vl / qwen3_vl_moe text-side multimodal support —
# interleaved 3D mrope (set_mrope_state) and deepstack mid-layer visual
# injection (set_visual_state). Text-only behavior must be byte-identical
# once the side state is reset.

import copy
import unittest

import mlx.core as mx

from mlx_lm.models import qwen3_vl, qwen3_vl_moe
from mlx_lm.models.cache import make_prompt_cache

HIDDEN = 128
LAYERS = 4
VOCAB = 512


def text_config(**extra):
    config = {
        "model_type": "qwen3_vl_text",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB,
        "head_dim": 32,
        "rope_theta": 1e6,
        "max_position_embeddings": 1024,
        "tie_word_embeddings": False,
        "rope_scaling": {"type": "default", "mrope_section": [8, 4, 4]},
    }
    config.update(extra)
    return config


def make_model():
    args = qwen3_vl.ModelArgs.from_dict(
        {"model_type": "qwen3_vl", "text_config": text_config()}
    )
    return qwen3_vl.Model(args)


# Interleaved-mrope positions for text(3) + one 1x4x4 image grid (=4 merged
# tokens at 3..6) + text(3): each image token gets its (t, h, w) grid position
# offset by the preceding text length; text resumes at offset + max(t, h, w).
IMAGE_PROMPT_LEN = 10
IMAGE_POSITION_IDS = mx.array(
    [
        [[0, 1, 2, 3, 3, 3, 3, 5, 6, 7]],
        [[0, 1, 2, 3, 3, 4, 4, 5, 6, 7]],
        [[0, 1, 2, 3, 4, 3, 4, 5, 6, 7]],
    ]
)
IMAGE_ROPE_DELTAS = mx.array([[8 - IMAGE_PROMPT_LEN]])
VISUAL_POS_MASKS = mx.array([[False, False, False, True, True, True, True, False, False, False]])


class TestUnifiedQwen3VL(unittest.TestCase):
    def model_test_runner(self, model, model_type, vocab_size, num_layers):
        # tests/test_models.py's runner idiom, minus the dtype-cast loop.
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

        inputs = mx.array([[0, 1], [2, 3]])
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (2, 2, vocab_size))

        copy.deepcopy(model)

    def test_qwen3_vl(self):
        self.model_test_runner(make_model(), "qwen3_vl", VOCAB, LAYERS)

    def test_qwen3_vl_moe(self):
        args = qwen3_vl_moe.ModelArgs.from_dict(
            {
                "model_type": "qwen3_vl_moe",
                "text_config": text_config(
                    model_type="qwen3_vl_moe_text",
                    num_experts=4,
                    num_experts_per_tok=2,
                    decoder_sparse_step=1,
                    mlp_only_layers=[],
                    moe_intermediate_size=128,
                    norm_topk_prob=True,
                ),
            }
        )
        model = qwen3_vl_moe.Model(args)
        self.assertTrue(hasattr(model.layers[0].mlp, "switch_mlp"))
        self.model_test_runner(model, "qwen3_vl_moe", VOCAB, LAYERS)

    def test_mrope_state_and_reset(self):
        model = make_model()
        tokens = mx.array([[i % VOCAB for i in range(IMAGE_PROMPT_LEN)]])
        baseline = model(tokens)

        model.model.set_mrope_state(IMAGE_POSITION_IDS, IMAGE_ROPE_DELTAS)
        cache = make_prompt_cache(model)
        embeddings = model.model.embed_tokens(tokens)
        out = model(tokens, cache=cache, input_embeddings=embeddings)
        self.assertEqual(out.shape, (1, IMAGE_PROMPT_LEN, VOCAB))
        # Decode: stored positions are exhausted, so sequential offset+delta ones.
        step = model(mx.argmax(out[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(step.shape, (1, 1, VOCAB))

        model.model.reset_mrope_state()
        self.assertTrue(mx.array_equal(model(tokens), baseline))

    def test_deepstack_injection(self):
        model = make_model()
        tokens = mx.array([[i % VOCAB for i in range(IMAGE_PROMPT_LEN)]])
        text_baseline = model(tokens)
        embeddings = model.model.embed_tokens(tokens)
        baseline = model(tokens, cache=make_prompt_cache(model), input_embeddings=embeddings)

        # Zero embeds must be an exact no-op (nothing added anywhere).
        model.model.set_visual_state(
            visual_pos_masks=VISUAL_POS_MASKS,
            deepstack_visual_embeds=[mx.zeros((4, HIDDEN)) for _ in range(2)],
        )
        out_zero = model(tokens, cache=make_prompt_cache(model), input_embeddings=embeddings)
        self.assertTrue(mx.array_equal(out_zero, baseline))

        # Nonzero embeds change outputs at/after the image span, but the causal
        # positions before it (0..2) cannot see the injection.
        deepstack = [mx.random.normal((4, HIDDEN)) for _ in range(2)]
        model.model.set_visual_state(
            visual_pos_masks=VISUAL_POS_MASKS, deepstack_visual_embeds=deepstack
        )
        out = model(tokens, cache=make_prompt_cache(model), input_embeddings=embeddings)
        self.assertTrue(mx.array_equal(out[:, :3], baseline[:, :3]))
        self.assertFalse(mx.allclose(out[:, 3:], baseline[:, 3:]))

        # Reset restores byte-identical text-only behavior.
        model.model.reset_visual_state()
        self.assertTrue(mx.array_equal(model(tokens), text_baseline))

    def test_deepstack_decode_skips_injection(self):
        model = make_model()
        tokens = mx.array([[i % VOCAB for i in range(IMAGE_PROMPT_LEN)]])
        embeddings = model.model.embed_tokens(tokens)
        deepstack = [mx.random.normal((4, HIDDEN)) for _ in range(2)]

        model.model.set_visual_state(
            visual_pos_masks=VISUAL_POS_MASKS, deepstack_visual_embeds=deepstack
        )
        cache = make_prompt_cache(model)
        out = model(tokens, cache=cache, input_embeddings=embeddings)
        next_token = mx.argmax(out[0, -1:, :], keepdims=True)
        cache_reset = copy.deepcopy(cache)

        # Decode with state still set vs after reset: identical — the stored
        # mask is exhausted, so decode forwards never re-apply deepstack.
        step_with_state = model(next_token, cache=cache)
        model.model.reset_visual_state()
        step_reset = model(next_token, cache=cache_reset)
        self.assertTrue(mx.array_equal(step_with_state, step_reset))

    def test_deepstack_chunked_prefill(self):
        model = make_model()
        tokens = mx.array([[i % VOCAB for i in range(IMAGE_PROMPT_LEN)]])
        embeddings = model.model.embed_tokens(tokens)
        deepstack = [mx.random.normal((4, HIDDEN)) for _ in range(2)]

        model.model.set_visual_state(
            visual_pos_masks=VISUAL_POS_MASKS, deepstack_visual_embeds=deepstack
        )
        cache = make_prompt_cache(model)
        single = model(tokens, cache=cache, input_embeddings=embeddings)

        # Chunk boundary at 6 splits the image span (3..6) across two forwards.
        model.model.set_visual_state(
            visual_pos_masks=VISUAL_POS_MASKS, deepstack_visual_embeds=deepstack
        )
        cache = make_prompt_cache(model)
        model(tokens[:, :6], cache=cache, input_embeddings=embeddings[:, :6])
        chunked = model(tokens[:, 6:], cache=cache, input_embeddings=embeddings[:, 6:])
        self.assertTrue(
            mx.allclose(single[:, -1], chunked[:, -1], rtol=1e-4, atol=1e-5)
        )
        model.model.reset_visual_state()


if __name__ == "__main__":
    unittest.main()
