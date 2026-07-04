# Copyright © 2026 Apple Inc.
#
# input_embeddings plumbing for nemotron_h (nemotron_h_nano_omni's text model) and
# verification that gemma3n's native injection re-derives per-layer inputs from the
# raw token ids that generation always passes alongside the embeddings.

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import gemma3n, nemotron_h
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import does_model_support_input_embeddings

NEMOTRON_TEXT_CONFIG = {
    "model_type": "nemotron_h",
    "vocab_size": 1000,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 4,
    "max_position_embeddings": 1000,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "attention_bias": False,
    "mamba_num_heads": 4,
    "mamba_head_dim": 32,
    "mamba_proj_bias": False,
    "ssm_state_size": 16,
    "conv_kernel": 4,
    "n_groups": 1,
    "mlp_bias": False,
    "layer_norm_epsilon": 1e-5,
    "use_bias": False,
    "use_conv_bias": True,
    "hybrid_override_pattern": ["M", "*", "-", "M"],
}

GEMMA3N_CONFIG = {
    "model_type": "gemma3n",
    "text_config": {
        "model_type": "gemma3n_text",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        # mlx-vlm conversions emit a per-layer list here; MLP handles both forms
        "intermediate_size": [256, 256, 256, 256],
        "num_attention_heads": 4,
        "head_dim": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1000,
        "num_key_value_heads": 2,
        "num_kv_shared_layers": 2,
        "vocab_size_per_layer_input": 500,
        "sliding_window": 8,
        "max_position_embeddings": 1000,
        "rope_local_base_freq": 10000.0,
        "rope_theta": 1000000.0,
        "final_logit_softcapping": 30.0,
        "layer_types": [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        "activation_sparsity_pattern": [0.95, 0.0, 0.0, 0.0],
        "hidden_size_per_layer_input": 16,
        "altup_num_inputs": 2,
        "altup_coef_clip": 120.0,
        "altup_correct_scale": True,
        "altup_active_idx": 0,
        "laurel_rank": 8,
    },
}


class TestUnifiedPlumb3(unittest.TestCase):
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

    def injection_test_runner(self, model, embed, vocab_size):
        """Injecting the token path's own embeddings must reproduce it exactly, and a
        multimodal call must leave subsequent text-only calls untouched."""
        self.assertTrue(does_model_support_input_embeddings(model))

        inputs = mx.array([[0, 1, 2, 3]])
        baseline = model(inputs)

        injected = model(inputs, input_embeddings=embed(inputs))
        self.assertTrue(mx.array_equal(baseline, injected))

        # "vision" embeddings (arbitrary values) through a cached prefill + decode
        cache = make_prompt_cache(model)
        vision = mx.random.uniform(shape=embed(inputs).shape) * 0.1
        out = model(inputs, cache=cache, input_embeddings=vision)
        self.assertEqual(out.shape, (1, 4, vocab_size))
        out = model(mx.argmax(out[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(out.shape, (1, 1, vocab_size))

        # no residue: plain injection holds no side state
        self.assertTrue(mx.array_equal(baseline, model(inputs)))

    def test_nemotron_h(self):
        args = nemotron_h.ModelArgs.from_dict(NEMOTRON_TEXT_CONFIG)
        model = nemotron_h.Model(args)
        self.model_test_runner(model, "nemotron_h", 1000, 4)
        self.injection_test_runner(model, model.backbone.embeddings, 1000)

    def test_nemotron_h_nano_omni_config(self):
        # omni checkpoints nest the language config; both key spellings must load
        for key in ("text_config", "llm_config"):
            args = nemotron_h.ModelArgs.from_dict(
                {
                    "model_type": "nemotron_h_nano_omni",
                    key: dict(NEMOTRON_TEXT_CONFIG),
                    "vision_config": {"model_type": "radio"},
                    "img_context_token_id": 999,
                }
            )
            self.assertEqual(args.model_type, "nemotron_h_nano_omni")
            self.assertEqual(args.hidden_size, 128)
            self.assertEqual(args.hybrid_override_pattern, ["M", "*", "-", "M"])

    def test_nemotron_h_nano_omni_sanitize(self):
        args = nemotron_h.ModelArgs.from_dict(NEMOTRON_TEXT_CONFIG)
        model = nemotron_h.Model(args)
        weights = {
            f"language_model.{k}": v for k, v in tree_flatten(model.parameters())
        }
        weights["language_model.mtp.layers.0.weight"] = mx.zeros((8, 8))
        weights["vision_model.radio_model.blocks.0.attn.qkv.weight"] = mx.zeros((8, 8))
        weights["mlp1.layers.1.weight"] = mx.zeros((8, 8))
        weights["sound_encoder.encoder.layers.0.weight"] = mx.zeros((8, 8))
        weights["sound_projection.linear.weight"] = mx.zeros((8, 8))
        model.load_weights(list(model.sanitize(weights).items()), strict=True)

    def test_gemma3n(self):
        args = gemma3n.ModelArgs.from_dict(GEMMA3N_CONFIG)
        model = gemma3n.Model(args)
        lm = model.model.language_model
        self.model_test_runner(model, "gemma3n", 1000, 4)

        # gemma3n's token path scales embeddings by sqrt(hidden); injected
        # embeddings are consumed as-is, so pre-scale to reproduce it exactly
        hidden = GEMMA3N_CONFIG["text_config"]["hidden_size"]
        self.injection_test_runner(
            model, lambda ids: lm.embed_tokens(ids) * (hidden**0.5), 1000
        )

    def test_gemma3n_per_layer_rederivation(self):
        """The per-layer inputs gemma3n re-derives from raw token ids equal what
        mlx-vlm's get_input_embeddings returns (its formula: zero every id >=
        vocab_size_per_layer_input, then embed) — so the bridge can drop the
        returned per_layer_inputs and pass only ids + embeddings."""
        args = gemma3n.ModelArgs.from_dict(GEMMA3N_CONFIG)
        model = gemma3n.Model(args)
        lm = model.model.language_model
        vpli = GEMMA3N_CONFIG["text_config"]["vocab_size_per_layer_input"]

        # 900 stands in for an image soft token (id >= vocab_size_per_layer_input)
        ids = mx.array([[0, 1, 900, 5]])
        mlx_vlm_formula = lm.get_per_layer_inputs(
            mx.where(ids < vpli, ids, mx.zeros_like(ids))
        )
        self.assertTrue(mx.array_equal(lm.get_per_layer_inputs(ids), mlx_vlm_formula))

        # a mixed prompt (scaled text embeddings + a raw "vision feature" at the
        # image position) prefills and decodes; text-only behavior is untouched
        hidden = GEMMA3N_CONFIG["text_config"]["hidden_size"]
        baseline = model(ids)
        embeds = lm.embed_tokens(ids) * (hidden**0.5)
        vision = mx.random.uniform(shape=(1, 1, hidden))
        embeds = mx.concatenate([embeds[:, :2], vision, embeds[:, 3:]], axis=1)
        cache = make_prompt_cache(model)
        out = model(ids, cache=cache, input_embeddings=embeds)
        self.assertEqual(out.shape, (1, 4, 1000))
        out = model(mx.argmax(out[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(out.shape, (1, 1, 1000))
        self.assertTrue(mx.array_equal(baseline, model(ids)))

    def test_gemma3n_sanitize(self):
        args = gemma3n.ModelArgs.from_dict(GEMMA3N_CONFIG)
        model = gemma3n.Model(args)
        params = tree_flatten(model.parameters())

        # HF layout: model.language_model.* with sibling towers under model.*
        weights = {k: v for k, v in params}
        weights["model.vision_tower.blocks.0.conv.weight"] = mx.zeros((8, 8))
        weights["model.audio_tower.layers.0.weight"] = mx.zeros((8, 8))
        weights["model.embed_vision.embedding.weight"] = mx.zeros((8, 8))
        weights["model.embed_audio.embedding.weight"] = mx.zeros((8, 8))
        model.load_weights(list(model.sanitize(weights).items()), strict=True)

        # mlx-vlm layout: language_model.model.* with towers at the top level
        weights = {
            "language_model.model." + k.removeprefix("model.language_model."): v
            for k, v in params
        }
        weights["vision_tower.blocks.0.conv.weight"] = mx.zeros((8, 8))
        weights["audio_tower.layers.0.weight"] = mx.zeros((8, 8))
        weights["embed_vision.embedding.weight"] = mx.zeros((8, 8))
        weights["embed_audio.embedding.weight"] = mx.zeros((8, 8))
        model.load_weights(list(model.sanitize(weights).items()), strict=True)


if __name__ == "__main__":
    unittest.main()
