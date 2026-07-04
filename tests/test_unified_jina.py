# Copyright © 2026 Apple Inc.

import copy
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from mlx_lm.models import jina_vlm
from mlx_lm.models.cache import make_prompt_cache


def tiny_args(**overrides):
    text_config = {
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "block_config": {
            "attn_config": {"n_heads": 4, "n_kv_heads": 2, "head_dim": 32},
            "ffn_config": {"size": 256},
            "lnorm_config": {"eps": 1e-6},
        },
        "vocab_size": 256,
        "additional_vocab_size": 8,
        "rope_theta": 1000000.0,
        "max_sequence_length": 2048,
    }
    text_config.update(overrides)
    return jina_vlm.ModelArgs(model_type="jvlm", text_config=text_config)


class TestUnifiedJina(unittest.TestCase):
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

    def test_jina_vlm(self):
        args = tiny_args()
        model = jina_vlm.Model(args)
        self.model_test_runner(model, "jvlm", 256, 2)

    def test_text_args_from_nested_config(self):
        # The real checkpoint nests per-block settings under block_config.
        args = jina_vlm.TextArgs.from_dict(
            {
                "hidden_size": 2048,
                "n_layers": 28,
                "block_config": {
                    "attn_config": {
                        "n_heads": 16,
                        "n_kv_heads": 8,
                        "head_dim": 128,
                        "q_lnorm": True,
                    },
                    "ffn_config": {"size": 6144},
                    "lnorm_config": {"eps": 1e-6},
                },
                "vocab_size": 151936,
                "additional_vocab_size": 128,
                "rope_theta": 1000000,
                "max_sequence_length": 40960,
                "model_type": "jvlm",
            }
        )
        self.assertEqual(args.num_hidden_layers, 28)
        self.assertEqual(args.num_key_value_heads, 8)
        self.assertEqual(args.head_dim, 128)
        self.assertEqual(args.intermediate_size, 6144)
        self.assertEqual(args.additional_vocab_size, 128)
        self.assertTrue(args.use_qk_norm)

    def test_extended_embedding(self):
        emb = jina_vlm.ExtendedEmbedding(16, 4, 8)
        emb.embedding = mx.random.normal((16, 8))
        emb.new_embedding = mx.random.normal((4, 8))
        # ids straddling the base-vocab boundary pick rows from the right table.
        ids = mx.array([[0, 15, 16, 19]])
        out = emb(ids)
        self.assertTrue(mx.allclose(out[0, 0], emb.embedding[0]))
        self.assertTrue(mx.allclose(out[0, 1], emb.embedding[15]))
        self.assertTrue(mx.allclose(out[0, 2], emb.new_embedding[0]))
        self.assertTrue(mx.allclose(out[0, 3], emb.new_embedding[3]))

    def test_gate_up_split_order(self):
        # jvlm's fused projection is [up, gate] — value half FIRST, gate SECOND.
        args = jina_vlm.TextArgs(hidden_size=8, intermediate_size=16)
        mlp = jina_vlm.MLP(args)
        w_up = mx.random.normal((16, 8))
        w_gate = mx.random.normal((16, 8))
        mlp.gate_up.weight = mx.concatenate([w_up, w_gate], axis=0)
        mlp.down.weight = mx.random.normal((8, 16))

        x = mx.random.normal((1, 3, 8))
        expected = (mx.sigmoid(x @ w_gate.T) * (x @ w_gate.T) * (x @ w_up.T)) @ (
            mlp.down.weight.T
        )
        self.assertTrue(mx.allclose(mlp(x), expected, atol=1e-5))

    def test_sanitize_loads_checkpoint_names(self):
        model = jina_vlm.Model(tiny_args())
        params = dict(tree_flatten(model.parameters()))

        # Reconstruct the checkpoint's key layout: lm_head at the top level (as in
        # raw jvlm conversions) plus vision-tower tensors that must be stripped.
        checkpoint = {
            k.removeprefix("language_model.") if k.startswith("language_model.lm_head.") else k: v
            for k, v in params.items()
        }
        checkpoint["vision_model.patch_embed.proj.weight"] = mx.zeros((4, 4))
        checkpoint["vl_connector.pad_embed"] = mx.zeros((2, 4))

        sanitized = model.sanitize(checkpoint)
        self.assertEqual(set(sanitized), set(params))
        model.load_weights(list(sanitized.items()), strict=True)

    def test_plain_injection_smoke(self):
        model = jina_vlm.Model(tiny_args())
        ids = mx.array([[3, 200, 258, 7]])  # includes an image-token row (>= 256)

        baseline = model(ids)

        # Injecting the model's own embeddings must reproduce the token forward.
        embeds = model.model.embedding(ids)
        injected = model(ids, input_embeddings=embeds)
        self.assertTrue(mx.allclose(baseline, injected))

        # A vision prompt = the same embeddings with image rows replaced; the forward
        # must run cleanly through prefill + a cached decode step.
        vision_embeds = embeds + mx.random.normal(embeds.shape) * (
            (ids >= 256)[..., None]
        )
        cache = make_prompt_cache(model)
        out = model(ids, cache=cache, input_embeddings=vision_embeds)
        self.assertEqual(out.shape, (1, 4, 256))
        step = model(mx.argmax(out[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(step.shape, (1, 1, 256))

        # Plain injection keeps no side state — the text-only forward is unchanged.
        after = model(ids)
        self.assertTrue(mx.array_equal(baseline, after))


if __name__ == "__main__":
    unittest.main()
