# Copyright © 2026 Apple Inc.

import copy
import unittest

import mlx.core as mx
from mlx.utils import tree_map

from mlx_lm.models.cache import make_prompt_cache


def tiny_args(**overrides):
    from mlx_lm.models import molmo_point

    base = dict(
        model_type="molmo_point",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=100,
        additional_vocab_size=8,
        rope_theta=1e6,
        vit_hidden_size=6,
        vit_layers=[-3, -9],  # vit_feature_dim = 12
        patch_embed_dim=16,
        patch_token_id=102,
        subpatch_token_id=103,
        location_token_id=104,
    )
    base.update(overrides)
    return molmo_point.ModelArgs(**base)


def randomize(model, seed=0):
    mx.random.seed(seed)
    model.update(
        tree_map(lambda p: 0.02 * mx.random.normal(p.shape), model.parameters())
    )
    return model


def tiny_visual_scenario(model):
    """B=1, L=10 prompt: image tokens at positions 2..5 (2,3,4 indexable, 5 not),
    P=4 pooled patch rows (one per image token), S=3 subpatches per row."""
    mx.random.seed(1)
    L = 10
    tokens = mx.array([[1, 2, 96, 96, 96, 97, 3, 4, 5, 6]])
    is_image = mx.array(
        [[False, False, True, True, True, True, False, False, False, False]]
    )
    is_indexable = mx.array(
        [[False, False, True, True, True, False, False, False, False, False]]
    )
    token_pooling = mx.array([[[0, 1, 2], [3, 4, -1], [5, -1, -1], [6, 7, 8]]])
    vit_features = mx.random.normal((1, 4, 3, 12))
    image_features = mx.random.normal((4, 64))
    model.model.set_visual_state(
        token_pooling=token_pooling,
        vit_features=vit_features,
        image_features=image_features,
        image_token_offsets=mx.array([0]),
        is_image_token=is_image,
        is_indexable_image_token=is_indexable,
    )
    embeds = model.lm.model.wte(tokens)
    return dict(
        tokens=tokens,
        embeds=embeds,
        vit_features=vit_features,
        image_features=image_features,
        L=L,
    )


class TestUnifiedMolmoPoint(unittest.TestCase):
    # Extended-vocab layout for the tiny scenario: total vocab 108, patches
    # 108..111, no-more-points 112, subpatches 113..115, locations 116..124.
    TOTAL = 108
    V_EXT = 108 + 4 + 1 + 3 + 9  # 125

    def test_model_text_only(self):
        from mlx_lm.models import molmo_point

        args = tiny_args()
        model = randomize(molmo_point.Model(args))
        self.assertEqual(len(model.layers), args.num_hidden_layers)
        self.assertEqual(model.model_type, args.model_type)

        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs = model(inputs)
            self.assertEqual(outputs.shape, (1, 2, self.TOTAL))
            self.assertEqual(outputs.dtype, t)

            cache = make_prompt_cache(model)
            outputs = model(inputs, cache=cache)
            self.assertEqual(outputs.shape, (1, 2, self.TOTAL))

            outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
            self.assertEqual(outputs.shape, (1, 1, self.TOTAL))
            self.assertEqual(outputs.dtype, t)

        inputs = mx.array([[0, 1], [2, 3]])
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (2, 2, self.TOTAL))

        copy.deepcopy(model)

    def test_text_only_matches_molmo2(self):
        """Same weights -> the molmo_point text path is molmo2 exactly (identical
        hidden states; logits agree over the shared base+extension vocab)."""
        from mlx_lm.models import molmo2, molmo_point

        args = tiny_args()
        m2 = randomize(
            molmo2.Model(
                molmo2.ModelArgs(
                    model_type="molmo2",
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=16,
                    vocab_size=100,
                    additional_vocab_size=8,
                    rope_theta=1e6,
                )
            )
        )
        mp = randomize(molmo_point.Model(args), seed=3)
        mp.lm.model.update(m2.model.parameters())
        mp.lm.lm_head.output_embeddings = m2.lm_head.weight

        # Include an extension-table id to exercise the split vocab.
        inputs = mx.array([[1, 2, 101, 3]])
        self.assertTrue(
            mx.array_equal(mp.lm.model(inputs), m2.model(inputs))
        )
        self.assertTrue(
            mx.allclose(mp(inputs)[..., :100], m2(inputs), atol=1e-6)
        )

        # Plain-injection contract: wte(tokens) as input_embeddings matches the
        # token forward.
        embeds = mp.lm.model.wte(inputs)
        self.assertTrue(mx.array_equal(mp(inputs), mp(inputs, input_embeddings=embeds)))

        # Cached decode agrees too.
        c2, cp = make_prompt_cache(m2), make_prompt_cache(mp)
        m2_out = m2(inputs, cache=c2)
        mp_out = mp(inputs, cache=cp)
        step = mx.argmax(m2_out[0, -1:, :], keepdims=True)
        self.assertTrue(
            mx.allclose(
                mp(step, cache=cp)[..., :100], m2(step, cache=c2), atol=1e-6
            )
        )

    def test_extended_id_without_state_raises(self):
        from mlx_lm.models import molmo_point

        model = randomize(molmo_point.Model(tiny_args()))
        with self.assertRaises(ValueError):
            model(mx.array([[1, self.TOTAL]]))

    def test_visual_state_prefill_and_decode(self):
        from mlx_lm.models import molmo_point

        model = randomize(molmo_point.Model(tiny_args()))
        baseline = model(mx.array([[1, 2, 3]]))

        sc = tiny_visual_scenario(model)
        state = model.model._point_state
        cache = make_prompt_cache(model)

        # Prefill: extended width, dummy point logits, patch keys captured.
        out = model(sc["tokens"], cache=cache, input_embeddings=sc["embeds"])
        self.assertEqual(out.shape, (1, sc["L"], self.V_EXT))
        self.assertTrue(bool(mx.all(out[..., self.TOTAL :] <= -1e4).item()))
        self.assertEqual(state["patch_k"].shape, (1, 5, 16))
        self.assertEqual(
            state["patch_k_mask"].tolist(), [[True, True, True, False, True]]
        )

        # Text-token decode: width matches prefill; exactly one live patch slot
        # (the argmax trick); subpatch/location regions are banned.
        out = model(mx.array([[7]]), cache=cache)
        self.assertEqual(out.shape, (1, 1, self.V_EXT))
        ext = out[0, -1, self.TOTAL :]
        self.assertEqual(int((ext > -1e4).sum().item()), 1)
        self.assertTrue(bool(mx.all(out[0, -1, 113:125] <= -1e4).item()))
        # The special ids are suppressed in the base region.
        for tid in (102, 103, 104):
            self.assertLessEqual(out[0, -1, tid].item(), -1e4)

        # Patch id decode: embedding = wte(patch_token_id) + connector feature.
        pid_ext = 109  # patch 1
        x, _, _, _ = model._embed_generated(mx.array([[pid_ext]]), state)
        expected = model.lm.model.wte(mx.array([[102]])) + sc["image_features"][1]
        self.assertTrue(mx.allclose(x, expected, atol=1e-6))
        out = model(mx.array([[pid_ext]]), cache=cache)
        # token_pooling row 1 = [3, 4, -1] -> subpatches 0,1 live, 2 masked;
        # everything outside the subpatch region is banned.
        live = (out[0, -1] > -1e4).tolist()
        self.assertEqual([i for i, v in enumerate(live) if v], [113, 114])
        self.assertEqual(state["last_patch_id"].tolist(), [[1]])

        # Subpatch id decode: embedding is REPLACED by build_vit_embedding of the
        # chosen ViT sub-patch of the last selected patch.
        sp_ext = 113  # subpatch 0
        x, _, _, _ = model._embed_generated(mx.array([[sp_ext]]), state)
        expected = model.build_vit_embedding(state["vit_sparse"][1, 0:1])
        self.assertTrue(mx.allclose(x[0, 0], expected[0], atol=1e-6))
        out = model(mx.array([[sp_ext]]), cache=cache)
        # After a subpatch only the 3x3 location region is legal.
        live = (out[0, -1] > -1e4).tolist()
        self.assertTrue(set(i for i, v in enumerate(live) if v) <= set(range(116, 125)))
        self.assertTrue(any(live[116:125]))

        # Location id decode: plain wte(location_token_id) embedding; afterwards
        # subpatch/location stay banned until the next patch.
        loc_ext = 118
        x, _, _, _ = model._embed_generated(mx.array([[loc_ext]]), state)
        self.assertTrue(
            mx.allclose(x, model.lm.model.wte(mx.array([[104]])), atol=1e-6)
        )
        out = model(mx.array([[loc_ext]]), cache=cache)
        self.assertTrue(bool(mx.all(out[0, -1, 113:125] <= -1e4).item()))

        # no-more-points: embeds as the patch token, then bans ALL point tokens.
        out = model(mx.array([[112]]), cache=cache)
        self.assertTrue(bool(mx.all(out[0, -1, self.TOTAL :] <= -1e4).item()))

        # Reset: a fresh text-only forward is byte-identical to before.
        model.model.reset_visual_state()
        self.assertIsNone(model.model._point_state)
        self.assertTrue(mx.array_equal(baseline, model(mx.array([[1, 2, 3]]))))

    def test_chunked_prefill_matches_single(self):
        """patch_k accumulation by absolute cache offset: splitting the prefill —
        even through the middle of the image span, as generate_step's L-1/1 split
        or a small prefill_step_size would — reproduces the single-shot keys."""
        from mlx_lm.models import molmo_point

        model = randomize(molmo_point.Model(tiny_args()))

        sc = tiny_visual_scenario(model)
        cache = make_prompt_cache(model)
        model(sc["tokens"], cache=cache, input_embeddings=sc["embeds"])
        single = model.model._point_state["patch_k"]
        model.model.reset_visual_state()

        for split in (4, 7, 9):
            sc = tiny_visual_scenario(model)
            state = model.model._point_state
            cache = make_prompt_cache(model)
            model(
                sc["tokens"][:, :split],
                cache=cache,
                input_embeddings=sc["embeds"][:, :split],
            )
            if split < 6:  # image span (positions 2..5) not fully seen yet
                self.assertIsNone(state["patch_k"])
            model(
                sc["tokens"][:, split:],
                cache=cache,
                input_embeddings=sc["embeds"][:, split:],
            )
            self.assertTrue(mx.allclose(single, state["patch_k"], atol=1e-5))
            model.model.reset_visual_state()

    def test_decode_before_prefill_raises(self):
        from mlx_lm.models import molmo_point

        model = randomize(molmo_point.Model(tiny_args()))
        tiny_visual_scenario(model)
        with self.assertRaises(ValueError):
            model(mx.array([[7]]), cache=make_prompt_cache(model))
        model.model.reset_visual_state()

    def test_nested_config_from_dict(self):
        from mlx_lm.models import molmo_point

        args = molmo_point.ModelArgs.from_dict(
            {
                "model_type": "molmo_point",
                "text_config": {
                    "model_type": "molmo2_text",
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "vocab_size": 100,
                    "additional_vocab_size": 8,
                    "rope_theta": 1e6,
                    "attn_implementation": "sdpa",  # non-field keys are ignored
                },
                "vit_config": {"hidden_size": 6, "num_hidden_layers": 27},
                "adapter_config": {"vit_layers": [-3, -9], "text_hidden_size": 64},
                "patch_embed_dim": 16,
                "patch_token_id": 102,
                "subpatch_token_id": 103,
                "location_token_id": 104,
                "norm_logits": True,
                "no_more_points_class": True,
                "patch_location": "3x3",
            }
        )
        self.assertEqual(args.hidden_size, 64)
        self.assertEqual(args.vocab_size, 100)
        self.assertEqual(args.vit_feature_dim, 12)
        self.assertEqual(args.total_vocab_size, 108)
        out = randomize(molmo_point.Model(args))(mx.array([[0, 1]]))
        self.assertEqual(out.shape, (1, 2, 108))

    def test_sanitize(self):
        from mlx_lm.models import molmo_point

        model = molmo_point.Model(tiny_args())
        # mlx-vlm conversion namespace (verified: mlx-community/MolmoPoint-8B-4bit).
        vlm = {
            "lm.model.wte.embedding": 0,
            "lm.model.wte.new_embedding": 0,
            "lm.model.blocks.0.self_attn.att_proj.weight": 0,
            "lm.model.ln_f.weight": 0,
            "lm.lm_head.output_embeddings": 0,
            "lm.lm_head.new_output_embeddings": 0,
            "point_predictor.patch_q.weight": 0,
            "point_predictor.x_norm.weight": 0,
            "point_predictor.add_no_point_class_embed.vector": 0,
            "build_vit_embedding.weight": 0,
            "build_vit_embedding.bias": 0,
            "vision_model.patch_embedding.weight": 0,
            "connector.image_projector.w1.weight": 0,
        }
        self.assertEqual(
            sorted(model.sanitize(vlm)),
            [
                "build_vit_embedding.bias",
                "build_vit_embedding.weight",
                "lm.lm_head.new_output_embeddings",
                "lm.lm_head.output_embeddings",
                "lm.model.blocks.0.self_attn.att_proj.weight",
                "lm.model.ln_f.weight",
                "lm.model.wte.embedding",
                "lm.model.wte.new_embedding",
                "point_predictor.add_no_point_class_embed.vector",
                "point_predictor.patch_q.weight",
                "point_predictor.x_norm.weight",
            ],
        )
        # Raw HF namespace (verified: allenai/MolmoPoint-8B).
        hf = {
            "model.transformer.wte.embedding": 0,
            "model.transformer.blocks.0.attn_norm.weight": 0,
            "model.transformer.ln_f.weight": 0,
            "lm_head.output_embeddings": 0,
            "lm_head.new_output_embeddings": 0,
            "model.point_predictor.subpatch_k.weight": 0,
            "model.point_predictor.subpatch_loc_k.bias": 0,
            "model.build_vit_embedding.weight": 0,
            "model.vit.patch_embedding.weight": 0,
            "model.connector.image_pooling_2d.wq.weight": 0,
        }
        self.assertEqual(
            sorted(model.sanitize(hf)),
            [
                "build_vit_embedding.weight",
                "lm.lm_head.new_output_embeddings",
                "lm.lm_head.output_embeddings",
                "lm.model.blocks.0.attn_norm.weight",
                "lm.model.ln_f.weight",
                "lm.model.wte.embedding",
                "point_predictor.subpatch_k.weight",
                "point_predictor.subpatch_loc_k.bias",
            ],
        )


if __name__ == "__main__":
    unittest.main()
