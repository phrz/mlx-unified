# Copyright © 2026 Apple Inc.
#
# mlx-unified: MiniMax M3 decoder (mlx_lm/models/minimax_m3.py) — MoE routing,
# block-sparse attention (indexer + custom cache), input_embeddings injection,
# and sanitize() of mlx-vlm / HF checkpoint layouts.

import copy
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from mlx_lm.models import minimax_m3
from mlx_lm.models.cache import KVCache, make_prompt_cache


def tiny_args(**overrides):
    base = dict(
        model_type="minimax_m3",
        hidden_size=128,
        intermediate_size=64,
        dense_intermediate_size=128,
        shared_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_hidden_layers=4,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        max_position_embeddings=2048,
        vocab_size=1000,
        num_local_experts=8,
        num_experts_per_tok=2,
        # Tiny blocks so short prompts exceed block_size * topk and force the
        # sparse selection path on layer 3 (layers 0-2 stay dense-attention).
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_index_dim": 16,
            "sparse_num_index_heads": 2,
            "sparse_topk_blocks": 2,
            "sparse_block_size": 4,
            "sparse_init_block": 1,
            "sparse_local_block": 1,
            "sparse_attention_freq": [0, 0, 0, 1],
        },
    )
    base.update(overrides)
    return minimax_m3.ModelArgs(**base)


class TestUnifiedMiniMax(unittest.TestCase):
    def test_model(self):
        args = tiny_args()
        model = minimax_m3.Model(args)

        self.assertEqual(len(model.layers), args.num_hidden_layers)
        self.assertEqual(model.model_type, args.model_type)
        # shared_intermediate_size == intermediate_size packs the shared expert
        self.assertIsInstance(
            model.layers[3].block_sparse_moe.switch_mlp,
            minimax_m3.MiniMaxPackedSwitchGLU,
        )

        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs = model(inputs)
            self.assertEqual(outputs.shape, (1, 2, args.vocab_size))
            self.assertEqual(outputs.dtype, t)

            cache = make_prompt_cache(model)
            self.assertIsInstance(cache[0], KVCache)
            self.assertIsInstance(cache[3], minimax_m3.MiniMaxM3KVCache)
            outputs = model(inputs, cache=cache)
            self.assertEqual(outputs.shape, (1, 2, args.vocab_size))
            self.assertEqual(outputs.dtype, t)

            outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
            self.assertEqual(outputs.shape, (1, 1, args.vocab_size))
            self.assertEqual(outputs.dtype, t)

        inputs = mx.array([[0, 1], [2, 3]])
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (2, 2, args.vocab_size))

        copy.deepcopy(model)

    def test_sparse_attention_decode(self):
        """Prefill past the sparse threshold, decode, and verify the trimmable
        index cache reproduces the same logits after a trim + replay."""
        args = tiny_args()
        model = minimax_m3.Model(args)
        tokens = mx.random.randint(0, args.vocab_size, (1, 40))

        cache = make_prompt_cache(model)
        prefill = model(tokens, cache=cache)
        self.assertEqual(prefill.shape, (1, 40, args.vocab_size))
        self.assertTrue(mx.isfinite(prefill).all().item())
        self.assertEqual(cache[3].offset, 40)
        self.assertEqual(cache[3].index_offset, 40)

        step = mx.argmax(prefill[:, -1:, :], axis=-1)
        decode = model(step, cache=cache)
        self.assertEqual(decode.shape, (1, 1, args.vocab_size))
        self.assertEqual(cache[3].index_offset, 41)

        # Trim the decode step plus a tail of the prompt, replay the tail, and
        # the logits must match the original prefill at the same position.
        self.assertTrue(all(c.is_trimmable() for c in cache))
        for c in cache:
            self.assertEqual(c.trim(11), 11)
        self.assertEqual(cache[3].index_offset, 30)
        replay = model(tokens[:, 30:], cache=cache)
        self.assertTrue(
            mx.allclose(replay, prefill[:, 30:], atol=1e-5, rtol=1e-5).item()
        )

    def test_sparse_decode_gather_matches_block_mask(self):
        """Past its crossover (key_length >= 64 * selected_length) decode takes
        the gather fast path; it must match the block-mask fallback exactly."""
        args = tiny_args()
        model = minimax_m3.Model(args)
        # selected_length = topk * block = 8, so 600 keys clear the 512 crossover
        tokens = mx.random.randint(0, args.vocab_size, (1, 600))
        step = mx.array([[7]])

        cache_a = make_prompt_cache(model)
        model(tokens, cache=cache_a)
        gather = model(step, cache=cache_a)

        cache_b = make_prompt_cache(model)
        model(tokens, cache=cache_b)
        attn = model.layers[3].self_attn
        original = attn._sparse_decode_attention
        attn._sparse_decode_attention = lambda *a, **k: None
        try:
            fallback = model(step, cache=cache_b)
        finally:
            attn._sparse_decode_attention = original

        self.assertTrue(mx.allclose(gather, fallback, atol=1e-5, rtol=1e-5).item())

    def test_input_embeddings_injection(self):
        """Injecting the model's own token embeddings must match token input,
        and a later token-only forward must be unchanged (no residual state)."""
        args = tiny_args()
        model = minimax_m3.Model(args)
        tokens = mx.random.randint(0, args.vocab_size, (1, 12))

        baseline = model(tokens)
        injected = model(tokens, input_embeddings=model.model.embed_tokens(tokens))
        self.assertTrue(mx.allclose(baseline, injected).item())

        after = model(tokens)
        self.assertTrue(mx.array_equal(baseline, after).item())

    def test_sanitize_mlx_vlm_layout(self):
        """An mlx-vlm minimax_m3_vl conversion (language_model.* + vision keys)
        must load into this text-only model."""
        args = tiny_args()
        model = minimax_m3.Model(args)
        params = dict(tree_flatten(model.parameters()))

        checkpoint = {f"language_model.{k}": v for k, v in params.items()}
        checkpoint["vision_tower.blocks.0.attn.qkv.weight"] = mx.zeros((4, 4))
        checkpoint["multi_modal_projector.linear_1.weight"] = mx.zeros((4, 4))
        checkpoint["patch_merge_mlp.0.weight"] = mx.zeros((4, 4))

        sanitized = model.sanitize(checkpoint)
        self.assertEqual(set(sanitized.keys()), set(params.keys()))
        model.load_weights(list(sanitized.items()), strict=True)

    def test_sanitize_hf_expert_stacking(self):
        """Per-expert HF weights (w1/w2/w3 + shared expert) must stack into the
        packed switch_mlp tensors this model uses."""
        args = tiny_args()
        model = minimax_m3.Model(args)
        params = dict(tree_flatten(model.parameters()))

        checkpoint = dict(params)
        for layer_idx in range(args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.block_sparse_moe"
            gate_up = checkpoint.pop(f"{prefix}.switch_mlp.gate_up_proj.weight", None)
            if gate_up is None:
                continue
            down = checkpoint.pop(f"{prefix}.switch_mlp.down_proj.weight")
            for e in range(args.num_local_experts):
                gate, up = mx.split(gate_up[e], 2, axis=0)
                checkpoint[f"{prefix}.experts.{e}.w1.weight"] = gate
                checkpoint[f"{prefix}.experts.{e}.w3.weight"] = up
                checkpoint[f"{prefix}.experts.{e}.w2.weight"] = down[e]
            shared_gate, shared_up = mx.split(gate_up[-1], 2, axis=0)
            checkpoint[f"{prefix}.shared_experts.gate_proj.weight"] = shared_gate
            checkpoint[f"{prefix}.shared_experts.up_proj.weight"] = shared_up
            checkpoint[f"{prefix}.shared_experts.down_proj.weight"] = down[-1]

        sanitized = model.sanitize(checkpoint)
        self.assertEqual(set(sanitized.keys()), set(params.keys()))
        for k, v in params.items():
            self.assertTrue(mx.array_equal(sanitized[k], v).item(), k)

    def test_vl_config_flattening(self):
        """ModelArgs.from_dict must adopt a nested text_config (minimax_m3_vl)."""
        args = minimax_m3.ModelArgs.from_dict(
            {
                "model_type": "minimax_m3_vl",
                "vision_config": {"hidden_size": 1280},
                "text_config": {
                    "model_type": "minimax_m3",
                    "hidden_size": 128,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 32,
                    "vocab_size": 1000,
                },
            }
        )
        self.assertEqual(args.model_type, "minimax_m3")
        self.assertEqual(args.hidden_size, 128)
        self.assertEqual(args.rotary_dim, 16)
        self.assertEqual(args.moe_layer_freq, [0, 0, 0, 1])
        self.assertTrue(args.has_sparse_index(3))
        self.assertFalse(args.has_sparse_index(0))


if __name__ == "__main__":
    unittest.main()
