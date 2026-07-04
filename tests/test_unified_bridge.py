# Copyright © 2026 Apple Inc.
"""Unit tests for mlx_lm/multimodal.py's central vision wiring — the TEXT_SIDE
capability registry, VisionPrompt slicing, and MlxVlmBridge.prepare's capability
dispatch — all exercised with mock feats/inputs, no model checkpoints."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from mlx_lm.multimodal import (
    BYPASS_CACHE_SIDES,
    GUARDED_FEATURES,
    MROPE_SIDES,
    QWEN_MROPE_TYPES,
    TEXT_SIDE,
    MlxVlmBridge,
    VisionPrompt,
    load_vision_encoder,
)

IMG = 9  # image placeholder id used throughout the dispatch fixtures


def make_feats(L=7, hidden=4, **kwargs):
    """A mock InputEmbeddingsFeatures: the three always-read fields plus extras."""
    base = dict(
        inputs_embeds=mx.ones((1, L, hidden)),
        position_ids=None,
        rope_deltas=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def make_inputs(L=7, **kwargs):
    """A mock prepare_inputs() output: tokens [1, 2, IMG*4, 3] by default — one
    image run of 4 (grid (1,4,4) at spatial_merge_size 2)."""
    ids = [1, 2, IMG, IMG, IMG, IMG, 3][:L]
    base = dict(input_ids=mx.array([ids]))
    base.update(kwargs)
    return base


def run_prepare(config, feats, inputs=None, vlm=None):
    """Drive MlxVlmBridge.prepare with a fake vlm/processor and mocked
    prepare_inputs — the checkpoint is never touched."""
    bridge = MlxVlmBridge(Path("/nonexistent"), config)
    if vlm is None:
        vlm = SimpleNamespace()
    vlm.get_input_embeddings = lambda *a, **k: feats
    bridge._vlm = vlm  # _ensure_loaded no-ops once set
    bridge._processor = object()
    inputs = inputs if inputs is not None else make_inputs()
    with mock.patch("mlx_vlm.utils.prepare_inputs", return_value=inputs):
        return bridge.prepare("prompt", [])


class TestTextSideRegistry(unittest.TestCase):
    def test_capability_entries(self):
        expected = {
            # mrope family (positions built by the bridge / processor-supplied)
            "qwen3_5": "mrope",
            "qwen3_5_moe": "mrope",
            "qwen2_vl": "mrope",
            "qwen2_5_vl": "mrope",
            "glm4v": "mrope",
            "glm4v_moe": "mrope",
            "glm_ocr": "mrope",
            "paddleocr_vl": "mrope",
            "hunyuan_vl": "mrope",
            # mrope + mm-expert token routing
            "ernie4_5_moe_vl": "ernie-visual",
            # gemma families
            "gemma4": "gemma-visual",
            "gemma4_unified": "gemma-visual",
            "gemma3n": "gemma3n-visual",
            # whole-prompt bidirectional prefix masks
            "paligemma": "attn-mask-4d",
            "gemma3": "attn-mask-4d",
            "moondream2": "attn-mask-4d",
            "moondream3": "attn-mask-4d",
            # deepstack
            "qwen3_vl": "mrope+deepstack",
            "qwen3_vl_moe": "mrope+deepstack",
            "qwen3_omni_moe": "mrope+deepstack",
            "granite4_vision": "granite-deepstack",
            # one-offs
            "mllama": "cross-attention",
            "zaya1_vl": "visual-lora",
            "falcon_ocr": "falcon-visual",
        }
        for arch, capability in expected.items():
            self.assertEqual(TEXT_SIDE.get(arch), capability, arch)

    def test_plain_archs_have_no_entry(self):
        # plain is the default — these must NOT appear in TEXT_SIDE.
        for arch in (
            "granite_vision",
            "llama4",
            "step3p7",
            "youtu_vl",
            "nemotron_h_nano_omni",
            "deepseekocr",
            "deepseekocr_2",
            "unlimited_ocr",
            "molmo",
            "molmo2",
            "minimax_m3_vl",
            "jvlm",
            "diffusion_gemma",
        ):
            self.assertNotIn(arch, TEXT_SIDE, arch)

    def test_qwen_mrope_types(self):
        self.assertEqual(
            set(QWEN_MROPE_TYPES),
            {
                "qwen3_5",
                "qwen3_5_moe",
                "qwen2_vl",
                "qwen2_5_vl",
                "qwen3_vl",
                "qwen3_vl_moe",
                "qwen3_omni_moe",
                "ernie4_5_moe_vl",
                "paddleocr_vl",
                "glm4v",
                "glm4v_moe",
                "glm_ocr",
            },
        )
        # every builder arch's capability must accept positions
        for arch in QWEN_MROPE_TYPES:
            self.assertIn(TEXT_SIDE[arch], MROPE_SIDES, arch)
        # hunyuan gets processor positions through the same mrope channel
        self.assertIn(TEXT_SIDE["hunyuan_vl"], MROPE_SIDES)

    def test_cache_policy(self):
        # the pre-existing paths keep the image-fingerprinted prompt cache…
        for side in ("mrope", "gemma-visual", "gemma3n-visual", "ernie-visual"):
            self.assertNotIn(side, BYPASS_CACHE_SIDES, side)
        # …every new capability class bypasses it for v1
        self.assertEqual(
            set(BYPASS_CACHE_SIDES),
            {
                "attn-mask-4d",
                "mrope+deepstack",
                "granite-deepstack",
                "cross-attention",
                "visual-lora",
                "falcon-visual",
            },
        )
        # every guarded feature's consumers are real capabilities
        capabilities = set(TEXT_SIDE.values())
        for field, (_, sides) in GUARDED_FEATURES.items():
            for side in sides:
                self.assertIn(side, capabilities, f"{field}: {side}")


class TestVisionPromptSlicing(unittest.TestCase):
    def full_prompt(self, deepstack):
        L, hidden = 6, 4
        # visual positions 2..3 (2 of them before nothing, both inside the drop)
        mask = mx.array([[False, True, True, False, False, False]])
        return VisionPrompt(
            tokens=list(range(L)),
            embeddings=mx.arange(L * hidden).reshape(L, hidden),
            position_ids=mx.arange(3 * L).reshape(3, 1, L),
            rope_deltas=mx.array([[-2]]),
            mm_token_type_ids=mx.arange(L).reshape(1, L),
            per_layer_token_ids=list(range(L)),
            attention_mask_4d=mx.ones((1, 1, L, L)),
            visual_pos_masks=mask,
            deepstack_visual_embeds=deepstack,
            deepstack_target_layers=[9, 6, 3, 0],
            cross_attention_states=mx.ones((1, 5, hidden)),
            cross_attention_mask=mx.arange(L)[None, None, :, None] * mx.ones((1, 1, L, 5)),
            full_text_row_masked_out_mask=mx.arange(L).reshape(1, 1, L, 1),
            visual_position_ids=mx.arange(L)[None],
            visual_rope_deltas=mx.array([[0]]),
            pos_hw=mx.ones((1, L, 2)),
        )

    def test_sliced_zero_is_self(self):
        vp = self.full_prompt([mx.ones((2, 4))])
        self.assertIs(vp.sliced(0), vp)

    def test_per_position_fields_slice(self):
        vp = self.full_prompt([mx.arange(2 * 4).reshape(2, 4)]).sliced(3)
        self.assertEqual(vp.tokens, [3, 4, 5])
        self.assertEqual(vp.embeddings.shape, (3, 4))
        self.assertTrue(mx.array_equal(vp.position_ids, mx.arange(18).reshape(3, 1, 6)[:, :, 3:]))
        self.assertTrue(mx.array_equal(vp.mm_token_type_ids, mx.array([[3, 4, 5]])))
        self.assertEqual(vp.per_layer_token_ids, [3, 4, 5])
        self.assertTrue(mx.array_equal(vp.visual_pos_masks, mx.array([[False, False, False]])))
        # text-axis cross-attention masks slice; the vision-axis states stay whole
        self.assertEqual(vp.cross_attention_mask.shape, (1, 1, 3, 5))
        self.assertTrue(mx.array_equal(vp.full_text_row_masked_out_mask, mx.array([3, 4, 5]).reshape(1, 1, 3, 1)))
        self.assertEqual(vp.cross_attention_states.shape, (1, 5, 4))

    def test_absolute_indexed_fields_stay_whole(self):
        vp = self.full_prompt([mx.ones((2, 4))]).sliced(2)
        self.assertEqual(vp.attention_mask_4d.shape, (1, 1, 6, 6))
        self.assertEqual(vp.visual_position_ids.shape, (1, 6))
        self.assertEqual(vp.visual_rope_deltas.shape, (1, 1))
        self.assertEqual(vp.pos_hw.shape, (1, 6, 2))
        self.assertEqual(vp.deepstack_target_layers, [9, 6, 3, 0])

    def test_deepstack_list_drops_consumed_rows(self):
        # qwen3_vl-style: flat per-layer tables lose the rows of visual positions
        # that fell inside the slice (one of the two masked positions is < 3).
        table = mx.arange(2 * 4).reshape(2, 4)
        vp = self.full_prompt([table, table + 100]).sliced(3)
        self.assertEqual(len(vp.deepstack_visual_embeds), 2)
        for got, want in zip(vp.deepstack_visual_embeds, (table, table + 100)):
            self.assertTrue(mx.array_equal(got, want[2:]))  # both rows consumed

    def test_deepstack_array_slices_sequence_axis(self):
        # granite4_vision-style: (n_sets, L, hidden) full-sequence adds
        full = mx.arange(2 * 6 * 4).reshape(2, 6, 4)
        vp = self.full_prompt(full).sliced(3)
        self.assertTrue(mx.array_equal(vp.deepstack_visual_embeds, full[:, 3:]))


class TestBridgeDispatch(unittest.TestCase):
    def test_plain_arch(self):
        vp = run_prepare({"model_type": "youtu_vl", "image_token_id": IMG}, make_feats())
        self.assertEqual(vp.embeddings.shape, (7, 4))
        self.assertEqual(vp.tokens, [1, 2, IMG, IMG, IMG, IMG, 3])
        for field in (
            "position_ids",
            "rope_deltas",
            "mm_token_type_ids",
            "attention_mask_4d",
            "visual_pos_masks",
            "deepstack_visual_embeds",
            "cross_attention_states",
            "visual_position_ids",
            "pos_hw",
        ):
            self.assertIsNone(getattr(vp, field), field)
        self.assertFalse(vp.single_prefill)
        self.assertFalse(vp.bypass_cache)

    def test_plain_arch_tolerates_informational_visual_pos_masks(self):
        # minimax_m3_vl returns visual_pos_masks that generation never consumes —
        # ignored, not rejected (only GUARDED_FEATURES fields fail loud).
        feats = make_feats(visual_pos_masks=mx.ones((1, 7), dtype=mx.bool_))
        vp = run_prepare({"model_type": "minimax_m3_vl", "image_token_id": IMG}, feats)
        self.assertIsNone(vp.visual_pos_masks)
        self.assertFalse(vp.bypass_cache)

    def test_unregistered_arch_rejections(self):
        # fail-loud: an arch with no TEXT_SIDE entry must not silently drop side
        # state that changes forward semantics.
        cases = {
            "attention_mask_4d": mx.ones((1, 1, 7, 7)),
            "deepstack_visual_embeds": [mx.ones((4, 4))],
            "cross_attention_states": mx.ones((1, 5, 4)),
            "per_layer_inputs": mx.ones((1, 7, 2, 2)),
        }
        for field, value in cases.items():
            with self.assertRaises(ValueError, msg=field):
                run_prepare({"model_type": "mystery_vlm"}, make_feats(**{field: value}))
        # positions from an arch without mrope support also fail loud
        with self.assertRaises(ValueError):
            run_prepare(
                {"model_type": "mystery_vlm"},
                make_feats(position_ids=mx.zeros((3, 1, 7)), rope_deltas=mx.zeros((1, 1))),
            )

    def test_registered_arch_rejects_other_capabilities_fields(self):
        # moondream2 owns attention_mask_4d but must still reject e.g. deepstack.
        with self.assertRaises(ValueError):
            run_prepare(
                {"model_type": "moondream2"},
                make_feats(deepstack_visual_embeds=[mx.ones((4, 4))]),
            )

    def test_attn_mask_4d_archs(self):
        mask = mx.ones((1, 1, 7, 7))
        for arch in ("moondream2", "moondream3", "paligemma", "gemma3"):
            vp = run_prepare({"model_type": arch}, make_feats(attention_mask_4d=mask))
            self.assertTrue(mx.array_equal(vp.attention_mask_4d, mask), arch)
            self.assertTrue(vp.single_prefill, arch)
            self.assertTrue(vp.bypass_cache, arch)
            self.assertIsNone(vp.position_ids, arch)

    def test_qwen_mrope_build(self):
        config = {
            "model_type": "qwen2_vl",
            "image_token_id": IMG,
            "vision_config": {"spatial_merge_size": 2},
        }
        inputs = make_inputs(image_grid_thw=mx.array([[1, 4, 4]]))
        vp = run_prepare(config, make_feats(), inputs)
        # tokens [1, 2, IMG×4, 3]: text 0..1, grid (t,h,w) offset 2, trailing at 4
        self.assertTrue(
            mx.array_equal(
                vp.position_ids,
                mx.array(
                    [
                        [0, 1, 2, 2, 2, 2, 4],
                        [0, 1, 2, 2, 3, 3, 4],
                        [0, 1, 2, 3, 2, 3, 4],
                    ]
                ).reshape(3, 1, 7),
            )
        )
        self.assertTrue(mx.array_equal(vp.rope_deltas, mx.array([[-2]])))
        self.assertFalse(vp.bypass_cache)

    def test_ernie_visual(self):
        # flat config: image token id via im_patch_id; mrope + token-type routing
        config = {
            "model_type": "ernie4_5_moe_vl",
            "im_patch_id": IMG,
            "vision_config": {"spatial_merge_size": 2},
        }
        inputs = make_inputs(image_grid_thw=mx.array([[1, 4, 4]]))
        vp = run_prepare(config, make_feats(), inputs)
        self.assertIsNotNone(vp.position_ids)
        self.assertTrue(
            mx.array_equal(vp.mm_token_type_ids, mx.array([[0, 0, 1, 1, 1, 1, 0]]))
        )
        self.assertIsNone(vp.per_layer_token_ids)
        self.assertFalse(vp.single_prefill)
        self.assertFalse(vp.bypass_cache)

    def test_hunyuan_processor_positions(self):
        config = {"model_type": "hunyuan_vl", "image_token_id": IMG}
        proc_positions = mx.arange(4 * 7).reshape(1, 4, 7)
        inputs = make_inputs(position_ids=proc_positions)
        vp = run_prepare(config, make_feats(), inputs)
        self.assertTrue(mx.array_equal(vp.position_ids, proc_positions.transpose(1, 0, 2)))
        self.assertTrue(mx.array_equal(vp.rope_deltas, mx.zeros((1, 1), dtype=vp.position_ids.dtype)))
        self.assertFalse(vp.bypass_cache)

    def test_gemma3n_scales_text_positions_and_discards_per_layer(self):
        config = {
            "model_type": "gemma3n",
            "text_config": {"hidden_size": 4, "vocab_size_per_layer_input": 100},
            "vision_config": {"vocab_offset": 100},
        }
        inputs = {"input_ids": mx.array([[1, 100, 2]])}
        feats = make_feats(L=3, per_layer_inputs=mx.ones((1, 3, 2, 2)))
        vp = run_prepare(config, feats, inputs)
        # ids < vocab_offset gain ×sqrt(hidden)=2; multimodal row 1 stays as-is
        self.assertTrue(
            mx.array_equal(vp.embeddings, mx.array([[2.0] * 4, [1.0] * 4, [2.0] * 4]))
        )
        self.assertIsNone(vp.per_layer_token_ids)
        self.assertFalse(vp.single_prefill)
        self.assertFalse(vp.bypass_cache)

    def test_gemma3n_vocab_offset_fallback(self):
        # no vision_config.vocab_offset → text_config.vocab_size_per_layer_input
        config = {
            "model_type": "gemma3n",
            "text_config": {"hidden_size": 4, "vocab_size_per_layer_input": 100},
            "vision_config": {},
        }
        vp = run_prepare(config, make_feats(L=3), {"input_ids": mx.array([[1, 100, 2]])})
        self.assertTrue(mx.array_equal(vp.embeddings[:, 0], mx.array([2.0, 1.0, 2.0])))

    def test_qwen3_vl_deepstack(self):
        config = {
            "model_type": "qwen3_vl",
            "image_token_id": IMG,
            "vision_config": {"spatial_merge_size": 2},
        }
        inputs = make_inputs(image_grid_thw=mx.array([[1, 4, 4]]))
        mask = mx.array([[0, 0, 1, 1, 1, 1, 0]]).astype(mx.bool_)
        tables = [mx.ones((4, 4)), mx.zeros((4, 4))]
        feats = make_feats(visual_pos_masks=mask, deepstack_visual_embeds=tables)
        vp = run_prepare(config, feats, inputs)
        self.assertIsNotNone(vp.position_ids)  # mrope built alongside deepstack
        self.assertTrue(mx.array_equal(vp.visual_pos_masks, mask))
        self.assertEqual(len(vp.deepstack_visual_embeds), 2)
        self.assertFalse(vp.single_prefill)
        self.assertTrue(vp.bypass_cache)

    def test_qwen3_omni_thinker_config_and_missing_deepstack(self):
        # everything nests under thinker_config; mlx-vlm's omni port returns no
        # deepstack tables yet — mrope + injection still work without them.
        config = {
            "model_type": "qwen3_omni_moe",
            "thinker_config": {
                "image_token_id": IMG,
                "vision_config": {"spatial_merge_size": 2},
            },
        }
        inputs = make_inputs(image_grid_thw=mx.array([[1, 4, 4]]))
        feats = make_feats(visual_pos_masks=mx.ones((1, 7), dtype=mx.bool_))
        vp = run_prepare(config, feats, inputs)
        self.assertEqual(vp.position_ids.shape, (3, 1, 7))
        self.assertIsNone(vp.deepstack_visual_embeds)
        self.assertIsNotNone(vp.visual_pos_masks)
        self.assertTrue(vp.bypass_cache)

    def test_granite_deepstack_target_layers_from_vlm(self):
        config = {"model_type": "granite4_vision", "image_token_index": 5}
        mask = mx.array([[False, True, True, False, False, False, False]])
        feats = make_feats(
            visual_pos_masks=mask, deepstack_visual_embeds=mx.ones((2, 7, 4))
        )
        vlm = SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(_deepstack_target_layers=[9, 6, 3, 0])
            )
        )
        vp = run_prepare(config, feats, vlm=vlm)
        self.assertEqual(vp.deepstack_target_layers, [9, 6, 3, 0])
        self.assertEqual(vp.deepstack_visual_embeds.shape, (2, 7, 4))
        self.assertIsNone(vp.position_ids)  # no mrope for granite
        self.assertFalse(vp.single_prefill)
        self.assertTrue(vp.bypass_cache)

    def test_granite_deepstack_target_layers_from_config(self):
        config = {
            "model_type": "granite4_vision",
            "image_token_index": 5,
            "deepstack_layer_map": [[0, 9], [1, 6], [2, 3], [3, 0]],
            "use_spatial_sampling": True,
            "spatial_target_layers": [12, 15, 18, 21],
        }
        feats = make_feats(
            visual_pos_masks=mx.ones((1, 7), dtype=mx.bool_),
            deepstack_visual_embeds=mx.ones((8, 7, 4)),
        )
        vp = run_prepare(config, feats)
        # deepstack sets first, then spatial — order must match the feature sets
        self.assertEqual(vp.deepstack_target_layers, [9, 6, 3, 0, 12, 15, 18, 21])

    def test_mllama_cross_attention(self):
        config = {"model_type": "mllama", "image_token_index": IMG}
        states = mx.ones((1, 5, 4))
        xmask = mx.zeros((1, 1, 7, 5))
        row_mask = mx.ones((1, 1, 7, 1))
        feats = make_feats(
            cross_attention_states=states,
            cross_attention_mask=xmask,
            full_text_row_masked_out_mask=row_mask,
        )
        vp = run_prepare(config, feats)
        self.assertTrue(mx.array_equal(vp.cross_attention_states, states))
        self.assertTrue(mx.array_equal(vp.cross_attention_mask, xmask))
        self.assertTrue(mx.array_equal(vp.full_text_row_masked_out_mask, row_mask))
        self.assertFalse(vp.single_prefill)
        self.assertTrue(vp.bypass_cache)

    def test_zaya_visual_lora(self):
        mask = mx.array([[0, 0, 1, 1, 1, 1, 0]]).astype(mx.bool_)
        vp = run_prepare(
            {"model_type": "zaya1_vl", "image_token_id": IMG},
            make_feats(visual_pos_masks=mask),
        )
        self.assertTrue(mx.array_equal(vp.visual_pos_masks, mask))
        self.assertIsNone(vp.deepstack_visual_embeds)
        self.assertFalse(vp.single_prefill)
        self.assertTrue(vp.bypass_cache)

    def test_falcon_visual(self):
        # collapsed 1D positions must land in the visual_* fields, NOT the mrope
        # ones (the server's set_mrope_state branch keys on vp.position_ids).
        config = {"model_type": "falcon_ocr", "img_id": 227}
        positions = mx.array([[0, 1, 1, 1, 1, 1, 2]])
        deltas = mx.array([[-4]])
        hw = mx.ones((1, 7, 2))
        mask = mx.ones((1, 1, 7, 7), dtype=mx.bool_)
        feats = make_feats(
            position_ids=positions,
            rope_deltas=deltas,
            pos_hw=hw,
            attention_mask_4d=mask,
        )
        vp = run_prepare(config, feats)
        self.assertIsNone(vp.position_ids)
        self.assertIsNone(vp.rope_deltas)
        self.assertTrue(mx.array_equal(vp.visual_position_ids, positions))
        self.assertTrue(mx.array_equal(vp.visual_rope_deltas, deltas))
        self.assertTrue(mx.array_equal(vp.pos_hw, hw))
        self.assertTrue(mx.array_equal(vp.attention_mask_4d, mask))
        self.assertTrue(vp.single_prefill)
        self.assertTrue(vp.bypass_cache)


class TestConfigDiscovery(unittest.TestCase):
    def bridge(self, config):
        return MlxVlmBridge(Path("/nonexistent"), config)

    def test_image_token_id_fallbacks(self):
        self.assertEqual(self.bridge({"image_token_id": 1, "img_id": 2}).image_token_id, 1)
        self.assertEqual(self.bridge({"image_token_index": 3}).image_token_id, 3)
        # ernie4_5_moe_vl (flat config)
        self.assertEqual(self.bridge({"im_patch_id": 100295}).image_token_id, 100295)
        # falcon_ocr (flat TII config)
        self.assertEqual(self.bridge({"img_id": 227}).image_token_id, 227)
        # nemotron_h_nano_omni
        self.assertEqual(self.bridge({"img_context_token_id": 42}).image_token_id, 42)
        # qwen3_omni_moe: thinker_config carries both id and vision_config
        b = self.bridge(
            {
                "thinker_config": {
                    "image_token_id": 151655,
                    "vision_config": {"spatial_merge_size": 2},
                }
            }
        )
        self.assertEqual(b.image_token_id, 151655)
        self.assertEqual(b.vision_config, {"spatial_merge_size": 2})

    def encoder_for(self, config):
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d) / "config.json", "w") as f:
                json.dump(config, f)
            return load_vision_encoder(d)

    def test_load_vision_encoder_special_cases(self):
        # ordinary VLM
        self.assertIsNotNone(self.encoder_for({"model_type": "qwen2_vl", "vision_config": {}}))
        # falcon_ocr: flat config, no vision_config — still a vision checkpoint
        enc = self.encoder_for({"model_type": "falcon_ocr", "img_id": 227})
        self.assertIsNotNone(enc)
        self.assertEqual(enc.image_token_id, 227)
        # qwen3_omni_moe: vision_config only under thinker_config
        self.assertIsNotNone(
            self.encoder_for(
                {"model_type": "qwen3_omni_moe", "thinker_config": {"vision_config": {}}}
            )
        )
        # molmo v1 flat configs have no vision_config — vision genuinely unusable
        self.assertIsNone(self.encoder_for({"model_type": "molmo", "hidden_size": 3584}))
        # text-only checkpoints
        self.assertIsNone(self.encoder_for({"model_type": "llama"}))

    def test_load_vision_encoder_missing_config(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(load_vision_encoder(Path(d) / "nope"))


class TestGemma3ModelProperty(unittest.TestCase):
    def test_inner_model_access(self):
        # server.py reaches side-state hooks through model.model — the gemma3
        # multimodal wrapper must expose the same inner-model property as gemma4.
        from mlx_lm.models import gemma3

        model = gemma3.Model(
            gemma3.ModelArgs(
                model_type="gemma3",
                text_config={
                    "model_type": "gemma3_text",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "intermediate_size": 128,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 32,
                },
                vocab_size=100,
            )
        )
        self.assertIs(model.model, model.language_model.model)
        self.assertTrue(hasattr(model.model, "set_visual_state"))
        self.assertTrue(hasattr(model.model, "reset_visual_state"))


if __name__ == "__main__":
    unittest.main()
