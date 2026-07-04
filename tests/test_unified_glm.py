# Copyright © 2026 Apple Inc.
#
# mlx-unified: GLM-family 3D multimodal RoPE (glm4v_text / glm_ocr_text /
# glm4v_moe→glm4_moe) — model forward, mrope side-state, and checkpoint
# sanitize coverage. Follows tests/test_models.py's model_test_runner idiom.

import copy
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from mlx_lm.models.cache import make_prompt_cache

GLM4V_CONFIG = {
    "model_type": "glm4v",
    "text_config": {
        "model_type": "glm4v_text",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 1000,
        "attention_bias": True,
        "max_position_embeddings": 1000,
        "partial_rotary_factor": 0.5,
        "rope_theta": 10000.0,
        # head_dim 32 → rotary 16 → 8 frequency pairs = sum(mrope_section)
        "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
    },
}

# glm_ocr_text nests every rope field under rope_parameters and uses FULL rotary
# dims; loaded through the same glm4v_text module (MODEL_REMAPPING).
GLM_OCR_CONFIG = {
    "model_type": "glm_ocr_text",
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "rms_norm_eps": 1e-5,
    "vocab_size": 1000,
    "attention_bias": False,
    "max_position_embeddings": 1000,
    "tie_word_embeddings": False,
    "rope_parameters": {
        "rope_type": "default",
        "mrope_section": [4, 6, 6],
        "partial_rotary_factor": 1.0,
        "rope_theta": 10000,
    },
}

GLM4V_MOE_CONFIG = {
    "model_type": "glm4v_moe",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "glm4v_moe_text",
        "vocab_size": 1000,
        "hidden_size": 128,
        "intermediate_size": 128,
        "max_position_embeddings": 1000,
        "moe_intermediate_size": 128,
        "norm_topk_prob": True,
        "num_attention_heads": 4,
        "n_group": 2,
        "head_dim": 32,
        "topk_group": 1,
        "n_shared_experts": 1,
        "n_routed_experts": 4,
        "routed_scaling_factor": 1.0,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000,
        "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
        "use_qk_norm": False,
        "attention_bias": True,
        "partial_rotary_factor": 0.5,
    },
}


def make_glm4v(config=GLM4V_CONFIG):
    from mlx_lm.models import glm4v_text

    return glm4v_text.Model(glm4v_text.ModelArgs.from_dict(config))


def make_glm4_moe(config=GLM4V_MOE_CONFIG):
    from mlx_lm.models import glm4_moe

    return glm4_moe.Model(glm4_moe.ModelArgs.from_dict(config))


def sequential_mrope_state(seq_length):
    """Text-only-equivalent 3D positions: all axes 0..L-1, zero delta."""
    position_ids = mx.broadcast_to(
        mx.arange(seq_length).reshape(1, 1, seq_length), (3, 1, seq_length)
    )
    return position_ids, mx.zeros((1, 1), dtype=mx.int32)


def image_mrope_state():
    """GLM-style positions for [2 text][1x2x2 image run][2 text] (L = 8).

    Text spans sequential on all axes; the image run gets its (t, h, w) grid
    offset by the cursor; the cursor then advances by max(t, h, w) = 2, so the
    delta is 6 - 8 = -2.
    """
    position_ids = mx.array(
        [
            [[0, 1, 2, 2, 2, 2, 4, 5]],
            [[0, 1, 2, 2, 3, 3, 4, 5]],
            [[0, 1, 2, 3, 2, 3, 4, 5]],
        ]
    )
    return position_ids, mx.array([[-2]], dtype=mx.int32)


class TestUnifiedGlm(unittest.TestCase):

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

    def mrope_test_runner(self, model, vocab_size):
        """set state → forward with input_embeddings → reset → text-only unchanged."""
        inner = model.model
        tokens = mx.array([[3, 7, 11, 11, 11, 11, 13, 17]])
        baseline = model(tokens)
        mx.eval(baseline)

        # Degenerate positions (equal t/h/w axes, zero delta) must reproduce the
        # 1D-RoPE path — this pins the pairing + frequency layout of the mrope apply.
        inner.set_mrope_state(*sequential_mrope_state(tokens.shape[1]))
        degenerate = model(tokens)
        self.assertTrue(mx.allclose(baseline, degenerate, rtol=1e-4, atol=1e-4))
        inner.reset_mrope_state()

        # Image-shaped positions through chunked prefill + two decode steps.
        inner.set_mrope_state(*image_mrope_state())
        embeddings = inner.embed_tokens(tokens)
        cache = make_prompt_cache(model)
        out = model(tokens[:, :5], cache=cache, input_embeddings=embeddings[:, :5])
        self.assertEqual(out.shape, (1, 5, vocab_size))
        out = model(tokens[:, 5:], cache=cache, input_embeddings=embeddings[:, 5:])
        self.assertEqual(out.shape, (1, 3, vocab_size))
        for _ in range(2):
            out = model(mx.argmax(out[0, -1:, :], keepdims=True), cache=cache)
            self.assertEqual(out.shape, (1, 1, vocab_size))
        mx.eval(out)

        # Reset must restore byte-identical text-only behavior.
        inner.reset_mrope_state()
        self.assertTrue(mx.array_equal(baseline, model(tokens)))

    def test_glm4v_text(self):
        model = make_glm4v()
        text = GLM4V_CONFIG["text_config"]
        self.model_test_runner(
            model, "glm4v", text["vocab_size"], text["num_hidden_layers"]
        )

    def test_glm4v_text_mrope(self):
        model = make_glm4v()
        self.mrope_test_runner(model, GLM4V_CONFIG["text_config"]["vocab_size"])

    def test_glm_ocr_text(self):
        # Flat (text_config-less) config exercises ModelArgs.from_dict's fallback.
        model = make_glm4v(GLM_OCR_CONFIG)
        self.model_test_runner(
            model,
            "glm_ocr_text",
            GLM_OCR_CONFIG["vocab_size"],
            GLM_OCR_CONFIG["num_hidden_layers"],
        )

    def test_glm_ocr_text_mrope(self):
        model = make_glm4v(GLM_OCR_CONFIG)
        self.mrope_test_runner(model, GLM_OCR_CONFIG["vocab_size"])

    def test_glm4_moe(self):
        model = make_glm4_moe()
        text = GLM4V_MOE_CONFIG["text_config"]
        self.model_test_runner(
            model, "glm4v_moe_text", text["vocab_size"], text["num_hidden_layers"]
        )

    def test_glm4_moe_mrope(self):
        model = make_glm4_moe()
        self.mrope_test_runner(model, GLM4V_MOE_CONFIG["text_config"]["vocab_size"])

    def test_glm4_moe_rejects_mrope_without_section(self):
        config = copy.deepcopy(GLM4V_MOE_CONFIG)
        config["text_config"]["rope_scaling"] = None
        model = make_glm4_moe(config)
        with self.assertRaises(ValueError):
            model.model.set_mrope_state(*sequential_mrope_state(4))

    def test_glm4v_text_sanitize(self):
        model = make_glm4v()
        expected = dict(tree_flatten(model.parameters())).keys()

        # HF-original multimodal layout: model.language_model.* / model.visual.* /
        # top-level lm_head, plus GLM-OCR's multi-token-prediction block.
        checkpoint = {}
        for k in expected:
            k = k.replace("language_model.model.", "model.language_model.")
            k = k.replace("language_model.lm_head.", "lm_head.")
            checkpoint[k] = mx.zeros((1,))
        checkpoint["model.visual.patch_embed.proj.weight"] = mx.zeros((1,))
        checkpoint["model.language_model.layers.2.self_attn.q_proj.weight"] = mx.zeros(
            (1,)
        )
        self.assertEqual(set(model.sanitize(checkpoint)), set(expected))

        # mlx-vlm-converted layout: language_model.* / vision_tower.*.
        checkpoint = {k: mx.zeros((1,)) for k in expected}
        checkpoint["vision_tower.patch_embed.proj.weight"] = mx.zeros((1,))
        self.assertEqual(set(model.sanitize(checkpoint)), set(expected))

    def test_glm4_moe_sanitize(self):
        model = make_glm4_moe()
        expected = dict(tree_flatten(model.parameters())).keys()

        # mlx-vlm-converted GLM-4.5V layout: everything under language_model.*
        # (experts already stacked as switch_mlp), plus a vision tower.
        checkpoint = {f"language_model.{k}": mx.zeros((1,)) for k in expected}
        checkpoint["vision_tower.patch_embed.proj.weight"] = mx.zeros((1,))
        self.assertEqual(set(model.sanitize(checkpoint)), set(expected))


if __name__ == "__main__":
    unittest.main()
