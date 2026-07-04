# Copyright © 2026 Apple Inc.
"""Unit tests for two bridge repairs in mlx_lm/multimodal.py:

1. molmo v1 flat-config vision — mlx-vlm-converted Molmo-0924 checkpoints have a
   FLAT config.json (model_type "molmo", no vision_config) yet carry full
   vision_tower.* weights; load_vision_encoder must exempt them from the
   vision_config gate, and prepare() must pass the processor's
   image_input_idx/image_masks through to get_input_embeddings (whose molmo
   implementation does an ADDITIVE merge at image_input_idx positions — the
   input_ids already contain the expanded image-token block, so plain injection
   stays row-aligned).

2. qwen3_omni_moe deepstack — mlx-vlm's Thinker.get_input_embeddings computes
   the tower's multiscale features but returns deepstack_visual_embeds=None; the
   bridge runs the tower ONCE itself, feeds the embeds back through
   cached_image_features (no double tower run), and wires the multiscale tuple
   into the deepstack fields. A future mlx-vlm that returns the tables itself
   must take precedence.

All exercised with mock feats/inputs — no model checkpoints.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from mlx_lm.multimodal import MlxVlmBridge, load_vision_encoder

IMG = 9  # image placeholder id used throughout the fixtures


def make_feats(L=7, hidden=4, **kwargs):
    base = dict(
        inputs_embeds=mx.ones((1, L, hidden)),
        position_ids=None,
        rope_deltas=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def run_prepare(config, feats, inputs, vlm=None):
    """Drive MlxVlmBridge.prepare with a fake vlm/processor and mocked
    prepare_inputs — the checkpoint is never touched."""
    bridge = MlxVlmBridge(Path("/nonexistent"), config)
    if vlm is None:
        vlm = SimpleNamespace(get_input_embeddings=lambda *a, **k: feats)
    bridge._vlm = vlm  # _ensure_loaded no-ops once set
    bridge._processor = object()
    with mock.patch("mlx_vlm.utils.prepare_inputs", return_value=inputs):
        return bridge.prepare("prompt", [])


class FakeOmniVlm:
    """A fake qwen3_omni_moe mlx-vlm object: a thinker whose vision tower returns
    an (embeds, multiscale) tuple and whose get_input_embeddings honors
    cached_image_features and (like upstream today) returns deepstack None."""

    def __init__(self, n_visual=4, hidden=4, L=7, fixed_deepstack=None):
        self.tower_calls = 0
        self.tower_pixel_dtypes = []
        self.embed_kwargs = None
        self._n_visual = n_visual
        self._hidden = hidden
        self._L = L
        self._fixed_deepstack = fixed_deepstack  # simulate a fixed upstream
        self.tower_embeds = mx.full((n_visual, hidden), 2.0, dtype=mx.float16)
        self.multiscale = tuple(
            mx.full((n_visual, hidden), float(i + 1), dtype=mx.float16)
            for i in range(3)
        )

        outer = self

        def tower(pixel_values, image_grid_thw):
            outer.tower_calls += 1
            outer.tower_pixel_dtypes.append(pixel_values.dtype)
            return outer.tower_embeds, outer.multiscale

        tower.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.float16))
        )
        self.thinker = SimpleNamespace(vision_tower=tower)

    def get_input_embeddings(self, input_ids, pixel_values, **kwargs):
        self.embed_kwargs = kwargs
        cached = kwargs.get("cached_image_features")
        if pixel_values is not None and cached is None:
            # the reference thinker would run its tower here — the bridge's
            # cached_image_features hand-off must prevent that.
            self.tower_calls += 1
        mask = mx.array([[t == IMG for t in [1, 2, IMG, IMG, IMG, IMG, 3]]])
        return make_feats(
            L=self._L,
            hidden=self._hidden,
            visual_pos_masks=mask,
            deepstack_visual_embeds=self._fixed_deepstack,
        )


OMNI_CONFIG = {
    "model_type": "qwen3_omni_moe",
    "thinker_config": {
        "image_token_id": IMG,
        "vision_config": {"spatial_merge_size": 2},
    },
}


def omni_inputs(**kwargs):
    base = dict(
        input_ids=mx.array([[1, 2, IMG, IMG, IMG, IMG, 3]]),
        pixel_values=mx.ones((4, 8), dtype=mx.float32),
        image_grid_thw=mx.array([[1, 4, 4]]),
    )
    base.update(kwargs)
    return base


class TestQwen3OmniDeepstack(unittest.TestCase):
    def test_multiscale_wired_with_single_tower_run(self):
        vlm = FakeOmniVlm()
        vp = run_prepare(OMNI_CONFIG, None, omni_inputs(), vlm=vlm)
        # exactly one tower pass, with the reference's dtype cast applied
        self.assertEqual(vlm.tower_calls, 1)
        self.assertEqual(vlm.tower_pixel_dtypes, [mx.float16])
        # the tower embeds were handed back so the thinker skipped its own run
        self.assertIs(vlm.embed_kwargs["cached_image_features"], vlm.tower_embeds)
        # deepstack fields populated from the multiscale tuple, rows intact,
        # cast to the merged-embedding dtype
        self.assertIsNotNone(vp.visual_pos_masks)
        self.assertEqual(int(vp.visual_pos_masks.sum()), 4)
        self.assertEqual(len(vp.deepstack_visual_embeds), 3)
        for i, table in enumerate(vp.deepstack_visual_embeds):
            self.assertEqual(table.shape, (4, 4))
            self.assertEqual(table.dtype, vp.embeddings.dtype)
            self.assertTrue(
                mx.array_equal(table, mx.full((4, 4), float(i + 1)))
            )
        # mrope still built by the bridge; deepstack rides the bypass-cache path
        self.assertEqual(vp.position_ids.shape, (3, 1, 7))
        self.assertTrue(vp.bypass_cache)

    def test_future_upstream_fix_takes_precedence(self):
        fixed = [mx.full((4, 4), 42.0) for _ in range(3)]
        vlm = FakeOmniVlm(fixed_deepstack=fixed)
        vp = run_prepare(OMNI_CONFIG, None, omni_inputs(), vlm=vlm)
        # feats.deepstack_visual_embeds non-None short-circuits the bridge's
        # own multiscale (which deliberately differs from `fixed`).
        self.assertIs(vp.deepstack_visual_embeds, fixed)
        for table in vp.deepstack_visual_embeds:
            self.assertTrue(mx.array_equal(table, mx.full((4, 4), 42.0)))

    def test_video_stays_on_degraded_path(self):
        # pixel_values_videos present: no cached-features hook upstream, so the
        # bridge must NOT pre-run the image tower (that would break the thinker's
        # image/video multiscale join) — deepstack stays absent.
        vlm = FakeOmniVlm()
        inputs = omni_inputs(pixel_values_videos=mx.ones((4, 8)))
        vp = run_prepare(OMNI_CONFIG, None, inputs, vlm=vlm)
        self.assertNotIn("cached_image_features", vlm.embed_kwargs)
        self.assertIsNone(vp.deepstack_visual_embeds)
        self.assertIsNotNone(vp.visual_pos_masks)  # injection mask still flows

    def test_no_pixel_values_never_touches_the_tower(self):
        # audio-only / degenerate processor output: no pixel_values, nothing to
        # pre-run (the server only calls prepare() when a request has images,
        # so image_grid_thw stays present for the mrope builder).
        vlm = FakeOmniVlm()
        inputs = omni_inputs()
        del inputs["pixel_values"]
        run_prepare(OMNI_CONFIG, None, inputs, vlm=vlm)
        self.assertEqual(vlm.tower_calls, 0)
        self.assertNotIn("cached_image_features", vlm.embed_kwargs)


class TestMolmoFlatConfig(unittest.TestCase):
    # the real mlx-community/Molmo-7B-D-0924-4bit config is FLAT: no vision_config
    MOLMO_CONFIG = {"model_type": "molmo", "hidden_size": 3584}

    def encoder_for(self, config):
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d) / "config.json", "w") as f:
                json.dump(config, f)
            return load_vision_encoder(d)

    def test_flat_config_gets_an_encoder(self):
        enc = self.encoder_for(self.MOLMO_CONFIG)
        self.assertIsNotNone(enc)
        # no image_token_id anywhere in the flat config — fine, the molmo
        # processor expands placeholders itself
        self.assertIsNone(enc.image_token_id)

    def test_other_flat_configs_still_text_only(self):
        self.assertIsNone(self.encoder_for({"model_type": "llama"}))

    def test_prepare_passes_processor_extras_through(self):
        # molmo's processor PREPENDS the expanded image-token block to the text
        # tokens and emits image_input_idx/image_masks; get_input_embeddings
        # needs both for its additive merge at image_input_idx positions.
        ids = [IMG, IMG, IMG, 1, 2, 3]  # image block first, molmo-style
        idx = mx.array([[[0, 1, 2]]])
        masks = mx.ones((1, 3, 4))
        inputs = dict(
            input_ids=mx.array([ids]),
            pixel_values=mx.ones((1, 3, 4)),
            image_input_idx=idx,
            image_masks=masks,
        )
        seen = {}

        def get_input_embeddings(input_ids, pixel_values, **kwargs):
            seen.update(kwargs)
            return make_feats(L=6)

        vlm = SimpleNamespace(get_input_embeddings=get_input_embeddings)
        vp = run_prepare(self.MOLMO_CONFIG, None, inputs, vlm=vlm)
        self.assertIs(seen["image_input_idx"], idx)
        self.assertIs(seen["image_masks"], masks)
        # plain injection: tokens carry the expanded block, embeddings stay
        # row-aligned with them, and no side state is produced
        self.assertEqual(vp.tokens, ids)
        self.assertEqual(vp.embeddings.shape, (6, 4))
        for field in (
            "position_ids",
            "rope_deltas",
            "visual_pos_masks",
            "deepstack_visual_embeds",
            "attention_mask_4d",
        ):
            self.assertIsNone(getattr(vp, field), field)
        self.assertFalse(vp.single_prefill)
        self.assertFalse(vp.bypass_cache)

    def guard_bridge(self, weight_keys):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        with open(Path(d.name) / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {k: "model.safetensors" for k in weight_keys}}, f)
        return MlxVlmBridge(Path(d.name), self.MOLMO_CONFIG)

    def test_vision_weight_guard_accepts_real_conversion(self):
        # the real checkpoint's index has language_model.* AND vision_tower.* keys
        bridge = self.guard_bridge(
            ["language_model.model.wte.embedding", "vision_tower.image_vit.x"]
        )
        with mock.patch("mlx_vlm.utils.load_model", return_value=object()), mock.patch(
            "mlx_vlm.utils.load_processor", return_value=object()
        ):
            bridge._ensure_loaded()
        self.assertIsNotNone(bridge._vlm)

    def test_vision_weight_guard_refuses_stripped_conversion(self):
        # an mlx-lm-made conversion keeps only language weights — fail loud
        # rather than serving randomly-initialized vision projections
        bridge = self.guard_bridge(["language_model.model.wte.embedding"])
        with self.assertRaises(ValueError):
            bridge._ensure_loaded()


if __name__ == "__main__":
    unittest.main()
