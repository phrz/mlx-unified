# Copyright © 2026 Apple Inc.

import copy
import unittest

import mlx.core as mx
from mlx.utils import tree_map

from mlx_lm.models.cache import make_prompt_cache


def tiny_molmo_args():
    from mlx_lm.models import molmo

    return molmo.ModelArgs(
        model_type="molmo",
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_norm_eps=1e-6,
        rope_theta=1e6,
        vocab_size=1000,
        embedding_size=1000,
        additional_vocab_size=8,
    )


def tiny_molmo2_args():
    from mlx_lm.models import molmo2

    return molmo2.ModelArgs(
        model_type="molmo2",
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=1000,
        additional_vocab_size=8,
        rope_theta=5e6,
    )


def randomize(model):
    mx.random.seed(0)
    model.update(
        tree_map(lambda p: 0.02 * mx.random.normal(p.shape), model.parameters())
    )
    return model


class TestUnifiedMolmo(unittest.TestCase):
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

        inputs = mx.array([[0, 1], [2, 3]])
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (2, 2, vocab_size))

        copy.deepcopy(model)

    def injection_test_runner(self, model):
        """Plain-injection contract: forwarding wte(tokens) as input_embeddings
        matches the token forward, injected (merged) embeddings prefill + decode
        works, and the text-only forward is unchanged afterwards."""
        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

        # Include an id from the extension table to exercise the split vocab.
        inputs = mx.array([[1, 2, model.args.vocab_size + 1, 3]])
        baseline = model(inputs)

        embeds = model.model.wte(inputs)
        via_embeds = model(inputs, input_embeddings=embeds)
        self.assertTrue(mx.allclose(baseline, via_embeds))

        # "Vision" merge: perturb one position (molmo merges additively into the
        # text embeddings), prefill from embeddings, then a token decode step.
        merged = embeds + 0.1 * (mx.arange(inputs.shape[1]) == 2)[None, :, None]
        cache = make_prompt_cache(model)
        out = model(inputs, cache=cache, input_embeddings=merged)
        self.assertEqual(out.shape[:2], (1, inputs.shape[1]))
        step = model(mx.argmax(out[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(step.shape[:2], (1, 1))
        self.assertFalse(mx.allclose(baseline, out))

        # No side state: a fresh text-only forward is byte-identical to before.
        self.assertTrue(mx.array_equal(baseline, model(inputs)))

    def test_molmo(self):
        from mlx_lm.models import molmo

        args = tiny_molmo_args()
        model = randomize(molmo.Model(args))
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )
        self.injection_test_runner(randomize(molmo.Model(args)))

    def test_molmo_weight_tying(self):
        from mlx_lm.models import molmo

        args = tiny_molmo_args()
        args.weight_tying = True
        model = randomize(molmo.Model(args))
        out = model(mx.array([[0, 1]]))
        self.assertEqual(out.shape, (1, 2, args.embedding_size))

    def test_molmo2(self):
        from mlx_lm.models import molmo2

        args = tiny_molmo2_args()
        model = randomize(molmo2.Model(args))
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )
        self.injection_test_runner(randomize(molmo2.Model(args)))

    def test_molmo2_nested_text_config(self):
        from mlx_lm.models import molmo2

        args = molmo2.ModelArgs.from_dict(
            {
                "model_type": "molmo2",
                "text_config": {
                    "model_type": "molmo2_text",
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 32,
                    "vocab_size": 500,
                    "additional_vocab_size": 4,
                    "attn_implementation": "sdpa",  # non-field keys are ignored
                },
            }
        )
        self.assertEqual(args.model_type, "molmo2")
        self.assertEqual(args.hidden_size, 64)
        self.assertEqual(args.vocab_size, 500)
        out = randomize(molmo2.Model(args))(mx.array([[0, 1]]))
        self.assertEqual(out.shape, (1, 2, 500))

    def test_molmo_sanitize(self):
        from mlx_lm.models import molmo

        model = molmo.Model(tiny_molmo_args())
        # mlx-vlm-converted checkpoint namespace.
        vlm = {
            "language_model.model.wte.embedding": 0,
            "language_model.model.wte.new_embedding": 0,
            "language_model.model.blocks.0.att_proj.weight": 0,
            "language_model.model.blocks.0.att_proj.bias": 0,
            "language_model.model.ff_out.weight": 0,
            "language_model.model.ln_f.weight": 0,
            "vision_tower.image_vit.class_embedding": 0,
        }
        self.assertEqual(
            sorted(model.sanitize(vlm)),
            [
                "model.blocks.0.att_proj.bias",
                "model.blocks.0.att_proj.weight",
                "model.ff_out.weight",
                "model.ln_f.weight",
                "model.wte.embedding",
                "model.wte.new_embedding",
            ],
        )
        # Raw HF checkpoint namespace.
        hf = {
            "model.transformer.wte.embedding": 0,
            "model.transformer.blocks.0.attn_out.weight": 0,
            "model.transformer.ff_out.weight": 0,
            "model.vision_backbone.image_vit.class_embedding": 0,
        }
        self.assertEqual(
            sorted(model.sanitize(hf)),
            [
                "model.blocks.0.attn_out.weight",
                "model.ff_out.weight",
                "model.wte.embedding",
            ],
        )

    def test_molmo2_sanitize(self):
        from mlx_lm.models import molmo2

        model = molmo2.Model(tiny_molmo2_args())
        vlm = {
            "language_model.model.wte.embedding": 0,
            "language_model.model.blocks.0.self_attn.att_proj.weight": 0,
            "language_model.model.blocks.0.self_attn.q_norm.weight": 0,
            "language_model.model.ln_f.weight": 0,
            "language_model.lm_head.weight": 0,
            "vision_tower.image_vit.positional_embedding": 0,
        }
        self.assertEqual(
            sorted(model.sanitize(vlm)),
            [
                "lm_head.weight",
                "model.blocks.0.self_attn.att_proj.weight",
                "model.blocks.0.self_attn.q_norm.weight",
                "model.ln_f.weight",
                "model.wte.embedding",
            ],
        )
        hf = {
            "model.transformer.wte.new_embedding": 0,
            "model.transformer.blocks.0.mlp.ff_proj.weight": 0,
            "lm_head.weight": 0,
            "model.vision_backbone.image_pooling_2d.wq.weight": 0,
        }
        self.assertEqual(
            sorted(model.sanitize(hf)),
            [
                "lm_head.weight",
                "model.blocks.0.mlp.ff_proj.weight",
                "model.wte.new_embedding",
            ],
        )


if __name__ == "__main__":
    unittest.main()
