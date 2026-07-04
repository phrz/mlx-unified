# Copyright © 2026 Apple Inc.
#
# input_embeddings plumbing for the plain-injection architectures:
# step3p5 (step3p7 / Step-3.7-Flash) and youtu_llm (youtu_vl / Youtu-VL).

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import step3p5, youtu_llm
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import does_model_support_input_embeddings

STEP3P5_CONFIG = {
    "model_type": "step3p5",
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "vocab_size": 1000,
    "num_attention_heads": 4,
    "num_attention_groups": 2,
    "head_dim": 32,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 1000,
    "sliding_window": 8,
    "layer_types": [
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
    ],
    "use_head_wise_attn_gate": True,
    "moe_num_experts": 4,
    "moe_top_k": 2,
    "moe_intermediate_size": 64,
    "share_expert_dim": 64,
    "moe_layers_enum": "2,3",
    "moe_router_scaling_factor": 1.0,
}

YOUTU_CONFIG = {
    "model_type": "youtu_llm",
    "vocab_size": 1000,
    "hidden_size": 128,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "kv_lora_rank": 32,
    "q_lora_rank": 48,
    "qk_rope_head_dim": 16,
    "v_head_dim": 32,
    "qk_nope_head_dim": 32,
    "max_position_embeddings": 1000,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}


class TestUnifiedPlumb2(unittest.TestCase):
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
        """Injecting embed_tokens's own output must reproduce the token path exactly,
        and a multimodal call must leave subsequent text-only calls untouched."""
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

    def test_step3p5(self):
        args = step3p5.ModelArgs.from_dict(STEP3P5_CONFIG)
        model = step3p5.Model(args)
        self.model_test_runner(model, "step3p5", 1000, 4)
        self.injection_test_runner(model, model.model.embed_tokens, 1000)

    def test_step3p7_config_hoist_and_quantization_remap(self):
        # step3p7 (VLM) configs nest the text config under text_config and key
        # per-path quantization overrides on the language_model.* tree.
        override = {"group_size": 64, "bits": 8}
        config = {
            "model_type": "step3p7",
            "text_config": {**STEP3P5_CONFIG},
            "vision_config": {"width": 8},
            "image_token_id": 999,
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "language_model.model.layers.2.mlp.gate.gate": override,
            },
        }
        args = step3p5.ModelArgs.from_dict(config)
        self.assertEqual(args.model_type, "step3p7")
        self.assertEqual(args.hidden_size, 128)
        self.assertEqual(args.moe_layers_enum, "2,3")
        self.assertEqual(
            config["quantization"].get("model.layers.2.mlp.gate.gate"), override
        )
        self.assertNotIn(
            "language_model.model.layers.2.mlp.gate.gate", config["quantization"]
        )

    def test_step3p7_sanitize(self):
        # An mlx-vlm conversion nests text weights under language_model.* next to
        # the vision tower — sanitize must yield a loadable text-only weight set.
        args = step3p5.ModelArgs.from_dict(STEP3P5_CONFIG)
        model = step3p5.Model(args)
        weights = {
            f"language_model.{k}": v for k, v in tree_flatten(model.parameters())
        }
        weights["vision_model.conv1.weight"] = mx.zeros((8, 8))
        weights["vision_model.transformer.resblocks.0.attn.weight"] = mx.zeros((8, 8))
        weights["vit_large_projector.weight"] = mx.zeros((8, 8))
        model.load_weights(list(model.sanitize(weights).items()), strict=True)

    def test_youtu_llm(self):
        args = youtu_llm.ModelArgs.from_dict(YOUTU_CONFIG)
        model = youtu_llm.Model(args)
        self.model_test_runner(model, "youtu_llm", 1000, 4)
        self.injection_test_runner(model, model.model.embed_tokens, 1000)

    def test_youtu_vl_sanitize(self):
        # mlx-vlm conversions store kv_b_proj split into per-head embed_q /
        # unembed_out under language_model.*, next to a vision tower; sanitize
        # must rebuild the exact joint projection (byte-identical forward).
        args = youtu_llm.ModelArgs.from_dict({**YOUTU_CONFIG, "model_type": "youtu_vl"})
        model = youtu_llm.Model(args)
        inputs = mx.array([[0, 1, 2, 3]])
        baseline = model(inputs)

        num_heads = args.num_attention_heads
        qk_nope = args.qk_nope_head_dim
        weights = dict(tree_flatten(model.parameters()))
        for layer_idx in range(args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.self_attn"
            w = weights.pop(f"{prefix}.kv_b_proj.weight")
            w = w.reshape(num_heads, qk_nope + args.v_head_dim, -1)
            weights[f"{prefix}.embed_q.weight"] = mx.contiguous(
                w[:, :qk_nope, :].swapaxes(-1, -2)
            )
            weights[f"{prefix}.unembed_out.weight"] = mx.contiguous(w[:, qk_nope:, :])
        weights = {f"language_model.{k}": v for k, v in weights.items()}
        weights["language_model.lm_head.weight"] = mx.zeros((1000, 128))  # tied: drop
        weights["vision_tower.merger.ln_q.weight"] = mx.zeros((8,))
        weights["siglip2.encoder.layers.0.mlp.fc1.weight"] = mx.zeros((8, 8))
        weights["merger.mlp.0.weight"] = mx.zeros((8, 8))

        model.load_weights(list(model.sanitize(weights).items()), strict=True)
        self.assertTrue(mx.array_equal(baseline, model(inputs)))


if __name__ == "__main__":
    unittest.main()
