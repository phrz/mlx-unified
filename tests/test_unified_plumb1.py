# Copyright © 2026 Apple Inc.
#
# input_embeddings plumbing for the plain-injection architectures:
# granite (granite_vision) and llama4 / llama4_text (Scout/Maverick).

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import granite, llama4, llama4_text
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import does_model_support_input_embeddings

GRANITE_CONFIG = {
    "model_type": "granite",
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "rms_norm_eps": 1e-5,
    "vocab_size": 1000,
    "logits_scaling": 8.0,
    "attention_multiplier": 0.015625,
    "embedding_multiplier": 12.0,
    "residual_multiplier": 0.22,
    "max_position_embeddings": 1000,
    "num_key_value_heads": 2,
    "attention_bias": False,
    "mlp_bias": False,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}

LLAMA4_CONFIG = {
    "model_type": "llama4",
    "text_config": {
        "attention_bias": False,
        "attention_chunk_size": 8,
        "head_dim": 32,
        "hidden_size": 128,
        "interleave_moe_layer_step": 2,
        "intermediate_size": 128,
        "intermediate_size_mlp": 128,
        "max_position_embeddings": 1000,
        "model_type": "llama4",
        "num_attention_heads": 4,
        "num_experts_per_tok": 1,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "num_local_experts": 2,
        "rms_norm_eps": 1e-4,
        "rope_scaling": None,
        "rope_theta": 1000,
        "use_qk_norm": True,
        "vocab_size": 1000,
    },
}

LLAMA4_TEXT_CONFIG = {
    "model_type": "llama4_text",
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "rms_norm_eps": 1e-5,
    "vocab_size": 1000,
    "num_key_value_heads": 2,
    "intermediate_size_mlp": 128,
    "rope_theta": 1000.0,
    "head_dim": 8,
    "tie_word_embeddings": False,
    "no_rope_layers": [0, 0, 1, 1],
    "use_qk_norm": True,
}


class TestUnifiedPlumb1(unittest.TestCase):
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

    def injection_test_runner(self, model, embed_tokens, vocab_size):
        """Injecting embed_tokens's own output must reproduce the token path exactly
        (proving any embedding scaling applies to injected embeddings too), and a
        multimodal call must leave subsequent text-only calls untouched."""
        self.assertTrue(does_model_support_input_embeddings(model))

        inputs = mx.array([[0, 1, 2, 3]])
        baseline = model(inputs)

        injected = model(inputs, input_embeddings=embed_tokens(inputs))
        self.assertTrue(mx.array_equal(baseline, injected))

        # "vision" embeddings (arbitrary values) through a cached prefill + decode
        cache = make_prompt_cache(model)
        vision = mx.random.uniform(shape=embed_tokens(inputs).shape) * 0.1
        out = model(inputs, cache=cache, input_embeddings=vision)
        self.assertEqual(out.shape, (1, 4, vocab_size))
        out = model(mx.argmax(out[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(out.shape, (1, 1, vocab_size))

        # no residue: plain injection holds no side state
        self.assertTrue(mx.array_equal(baseline, model(inputs)))

    def test_granite(self):
        args = granite.ModelArgs.from_dict(GRANITE_CONFIG)
        model = granite.Model(args)
        self.model_test_runner(model, "granite", 1000, 4)
        self.injection_test_runner(model, model.model.embed_tokens, 1000)

    def test_granite_vision_config_and_sanitize(self):
        # granite_vision nests the granite config under text_config and prefixes
        # language-model weights with language_model.* — both must load.
        config = {
            "model_type": "granite_vision",
            "text_config": {k: v for k, v in GRANITE_CONFIG.items()},
            "vision_config": {"hidden_size": 64},
            "image_token_index": 999,
        }
        args = granite.ModelArgs.from_dict(config)
        self.assertEqual(args.model_type, "granite_vision")
        self.assertEqual(args.embedding_multiplier, 12.0)
        model = granite.Model(args)

        weights = {
            f"language_model.{k}": v for k, v in tree_flatten(model.parameters())
        }
        weights["language_model.lm_head.weight"] = mx.zeros((1000, 128))  # tied: drop
        weights["vision_tower.blocks.0.attn.qkv.weight"] = mx.zeros((8, 8))
        weights["multi_modal_projector.linear_1.weight"] = mx.zeros((8, 8))
        weights["image_newline"] = mx.zeros((128,))
        model.load_weights(list(model.sanitize(weights).items()), strict=True)

    def test_llama4(self):
        args = llama4.ModelArgs.from_dict(LLAMA4_CONFIG)
        model = llama4.Model(args)
        self.model_test_runner(model, "llama4", 1000, 4)
        self.injection_test_runner(
            model, model.language_model.model.embed_tokens, 1000
        )

    def test_llama4_text(self):
        args = llama4_text.ModelArgs.from_dict(LLAMA4_TEXT_CONFIG)
        model = llama4_text.Model(args)
        self.model_test_runner(model, "llama4_text", 1000, 4)
        self.injection_test_runner(model, model.model.embed_tokens, 1000)


if __name__ == "__main__":
    unittest.main()
