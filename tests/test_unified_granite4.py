# Copyright © 2026 Apple Inc.
#
# mlx-unified: granite4_vision text-side support in models/granitemoehybrid.py —
# input_embeddings plumbing through the hybrid Mamba2/attention loop and deepstack
# mid-layer visual injection (set_visual_state/reset_visual_state).

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import granitemoehybrid
from mlx_lm.models.cache import make_prompt_cache


def tiny_args(**overrides):
    params = {
        "model_type": "granitemoehybrid",
        "vocab_size": 1000,
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "max_position_embeddings": 1000,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "attention_bias": False,
        "embedding_multiplier": 1.0,
        "attention_multiplier": 1.0,
        "logits_scaling": 1.0,
        "residual_multiplier": 1.0,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "shared_intermediate_size": 128,
        "mamba_n_heads": 8,
        "mamba_d_head": 16,
        "mamba_proj_bias": False,
        "mamba_d_state": 128,
        "mamba_d_conv": 4,
        "mamba_n_groups": 1,
        "mamba_conv_bias": False,
        "layer_types": ["mamba", "attention", "mamba", "attention"],
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000.0,
    }
    params.update(overrides)
    return granitemoehybrid.ModelArgs.from_dict(params)


class TestUnifiedGranite4(unittest.TestCase):
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
        model = granitemoehybrid.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_nested_text_config(self):
        # granite4_vision checkpoints nest the text body under text_config
        flat = tiny_args()
        nested = granitemoehybrid.ModelArgs.from_dict(
            {
                "model_type": "granite4_vision",
                "vision_config": {"hidden_size": 64},
                "deepstack_layer_map": [[-1, 0]],
                "text_config": {
                    k: v for k, v in vars(flat).items() if k != "model_type"
                },
            }
        )
        self.assertEqual(nested.model_type, "granite4_vision")
        self.assertEqual(nested.hidden_size, flat.hidden_size)
        self.assertEqual(nested.layer_types, flat.layer_types)

    def test_input_embeddings(self):
        # Only the embedding source changes: feeding embed_tokens output through
        # input_embeddings must reproduce the token forward exactly.
        model = granitemoehybrid.Model(tiny_args())
        inputs = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
        expected = model(inputs)
        embeds = model.model.embed_tokens(inputs)
        outputs = model(inputs, input_embeddings=embeds)
        self.assertTrue(mx.array_equal(outputs, expected))

    def test_deepstack_injection_and_reset(self):
        model = granitemoehybrid.Model(tiny_args())
        inner = model.model
        inputs = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
        seq_len = inputs.shape[1]
        hidden = 128

        baseline = model(inputs)

        # image run at positions 2..5; one feature set per target layer, targeting
        # a mamba layer (0) and an attention layer (1)
        visual_mask = mx.array([[False, False, True, True, True, True, False, False]])
        deepstack = mx.where(
            visual_mask[..., None],
            mx.random.normal((2, seq_len, hidden)),
            mx.zeros((2, seq_len, hidden)),
        )
        embeds = mx.where(
            visual_mask[..., None],
            mx.zeros((1, seq_len, hidden)),
            inner.embed_tokens(inputs),
        )

        inner.set_visual_state(
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack,
            deepstack_target_layers=[0, 1],
        )
        injected = model(inputs, input_embeddings=embeds)
        self.assertEqual(injected.shape, baseline.shape)
        self.assertFalse(mx.array_equal(injected, baseline))

        # state cleared must restore byte-identical text-only behavior
        inner.reset_visual_state()
        self.assertTrue(mx.array_equal(model(inputs), baseline))

    def test_deepstack_chunked_prefill_and_decode(self):
        # The cursor must keep chunked prefill aligned with the full-prompt state,
        # and decode steps (token ids, no input_embeddings) must be untouched.
        model = granitemoehybrid.Model(tiny_args())
        inner = model.model
        inputs = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
        seq_len = inputs.shape[1]
        hidden = 128

        visual_mask = mx.array([[False, True, True, True, True, False, False, False]])
        deepstack = mx.where(
            visual_mask[..., None],
            mx.random.normal((2, seq_len, hidden)),
            mx.zeros((2, seq_len, hidden)),
        )
        embeds = mx.where(
            visual_mask[..., None],
            mx.zeros((1, seq_len, hidden)),
            inner.embed_tokens(inputs),
        )

        inner.set_visual_state(
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack,
            deepstack_target_layers=[0, 3],
        )
        cache = make_prompt_cache(model)
        single = model(inputs, cache=cache, input_embeddings=embeds)
        decode_single = model(mx.array([[7]]), cache=cache)

        inner.set_visual_state(
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack,
            deepstack_target_layers=[0, 3],
        )
        cache = make_prompt_cache(model)
        model(inputs[:, :5], cache=cache, input_embeddings=embeds[:, :5])
        chunked = model(inputs[:, 5:], cache=cache, input_embeddings=embeds[:, 5:])
        decode_chunked = model(mx.array([[7]]), cache=cache)
        inner.reset_visual_state()

        self.assertTrue(mx.allclose(single[:, -1], chunked[:, -1], atol=1e-4))
        self.assertTrue(mx.allclose(decode_single, decode_chunked, atol=1e-4))

    def test_sanitize_mlx_vlm_layout(self):
        # A converted granite4_vision checkpoint stores the text body under
        # language_model.* alongside a vision tower and projectors; sanitize must
        # strip the vision side, remap, and drop the materialized tied head.
        model = granitemoehybrid.Model(tiny_args())
        flat = dict(tree_flatten(model.parameters()))
        weights = {f"language_model.{k}": v for k, v in flat.items()}
        weights["language_model.lm_head.weight"] = flat["model.embed_tokens.weight"]
        weights["vision_tower.blocks.0.weight"] = mx.zeros((4, 4))
        weights["layerwise_projectors.0.out_linear.weight"] = mx.zeros((4, 4))
        weights["spatial_projectors.0.out_linear.weight"] = mx.zeros((4, 4))
        weights["image_newline"] = mx.zeros((4,))

        sanitized = model.sanitize(weights)
        self.assertEqual(set(sanitized.keys()), set(flat.keys()))
        model.load_weights(list(sanitized.items()))

    def test_sanitize_raw_hf_layout(self):
        model = granitemoehybrid.Model(tiny_args())
        flat = dict(tree_flatten(model.parameters()))
        weights = {
            "model.language_model." + k[len("model.") :]: v for k, v in flat.items()
        }
        weights["lm_head.weight"] = flat["model.embed_tokens.weight"]
        weights["vision_tower.blocks.0.weight"] = mx.zeros((4, 4))
        weights["model.image_newline"] = mx.zeros((4,))

        sanitized = model.sanitize(weights)
        self.assertEqual(set(sanitized.keys()), set(flat.keys()))


if __name__ == "__main__":
    unittest.main()
