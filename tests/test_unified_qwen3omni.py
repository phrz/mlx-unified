# Copyright © 2026 Apple Inc.

import copy
import unittest

import mlx.core as mx

from mlx_lm.models import qwen3_omni_moe
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.multimodal import build_qwen_image_mrope_state

IMAGE_TOKEN_ID = 250


def tiny_model():
    args = qwen3_omni_moe.ModelArgs(
        model_type="qwen3_omni_moe",
        hidden_size=128,
        num_hidden_layers=4,
        intermediate_size=256,
        num_attention_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        rms_norm_eps=1e-5,
        vocab_size=256,
        num_key_value_heads=2,
        rope_theta=1000000.0,
        mlp_only_layers=[0],
        rope_scaling={"rope_type": "default", "mrope_section": [8, 4, 4]},
    )
    return qwen3_omni_moe.Model(args), args


def vision_prompt_state(model):
    """A server-shaped vision prompt: 2 text tokens, a 4-token image run
    (1x4x4 grid merged 2x2), 2 text tokens — with merged embeddings, qwen
    mrope positions, and deepstack state for the first 3 layers."""
    tokens = [10, 11] + [IMAGE_TOKEN_ID] * 4 + [12, 13]
    input_ids = mx.array([tokens])
    state = build_qwen_image_mrope_state(
        input_ids=input_ids,
        image_grid_thw=mx.array([[1, 4, 4]]),
        image_token_id=IMAGE_TOKEN_ID,
        spatial_merge_size=2,
    )
    embeddings = model.model.embed_tokens(input_ids)
    image_mask = input_ids == IMAGE_TOKEN_ID
    embeddings = mx.where(
        image_mask[..., None], mx.random.normal(embeddings.shape) * 0.02, embeddings
    )
    deepstack = [mx.random.normal((4, embeddings.shape[-1])) * 0.02 for _ in range(3)]
    return input_ids, embeddings, state, image_mask, deepstack


class TestUnifiedQwen3Omni(unittest.TestCase):
    def test_model(self):
        model, args = tiny_model()
        self.assertEqual(len(model.layers), args.num_hidden_layers)
        self.assertEqual(model.model_type, "qwen3_omni_moe")

        inputs = mx.array([[0, 1]])
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (1, 2, args.vocab_size))

        cache = make_prompt_cache(model)
        outputs = model(inputs, cache=cache)
        self.assertEqual(outputs.shape, (1, 2, args.vocab_size))

        outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(outputs.shape, (1, 1, args.vocab_size))

        outputs = model(mx.array([[0, 1], [2, 3]]))
        self.assertEqual(outputs.shape, (2, 2, args.vocab_size))

        copy.deepcopy(model)

    def test_from_dict_nested_thinker_config(self):
        args = qwen3_omni_moe.ModelArgs.from_dict(
            {
                "model_type": "qwen3_omni_moe",
                "enable_audio_output": True,
                "thinker_config": {
                    "text_config": {
                        "hidden_size": 128,
                        "num_hidden_layers": 4,
                        "intermediate_size": 256,
                        "num_attention_heads": 4,
                        "num_experts": 8,
                        "num_experts_per_tok": 2,
                        "moe_intermediate_size": 64,
                        "rms_norm_eps": 1e-5,
                        "vocab_size": 256,
                        "num_key_value_heads": 2,
                        "rope_theta": 1000000.0,
                        "head_dim": 32,
                        "model_type": "qwen3_omni_moe_text_encoder",
                        "rope_scaling": {"mrope_section": [8, 4, 4]},
                    },
                    "vision_config": {},
                    "audio_config": {},
                },
            }
        )
        self.assertEqual(args.model_type, "qwen3_omni_moe")
        self.assertEqual(args.head_dim, 32)
        self.assertEqual(args.mrope_section, [8, 4, 4])

    def test_mrope_sequential_matches_plain_rope(self):
        # With sequential positions on all three axes and delta 0, interleaved
        # mrope must reproduce plain 1D RoPE (both prefill and cached decode).
        model, args = tiny_model()
        inputs = mx.array([[3, 1, 4, 1, 5, 9]])
        L = inputs.shape[1]

        plain_cache = make_prompt_cache(model)
        plain_prefill = model(inputs, cache=plain_cache)
        plain_decode = model(mx.array([[7]]), cache=plain_cache)

        position_ids = mx.broadcast_to(mx.arange(L).reshape(1, 1, L), (3, 1, L))
        model.model.set_mrope_state(position_ids, mx.zeros((1, 1), dtype=mx.int32))
        mrope_cache = make_prompt_cache(model)
        mrope_prefill = model(inputs, cache=mrope_cache)
        mrope_decode = model(mx.array([[7]]), cache=mrope_cache)
        model.model.reset_mrope_state()

        self.assertTrue(mx.allclose(plain_prefill, mrope_prefill, atol=1e-4))
        self.assertTrue(mx.allclose(plain_decode, mrope_decode, atol=1e-4))

    def test_vision_injection_and_reset(self):
        # The full server-shaped flow: mrope + deepstack side state set, merged
        # embeddings injected via input_embeddings, a decode step, then reset —
        # text-only behavior must be byte-identical to a never-touched forward.
        model, args = tiny_model()
        text_inputs = mx.array([[10, 11, 12, 13]])
        baseline = model(text_inputs)

        input_ids, embeddings, state, _, deepstack = vision_prompt_state(model)
        L = input_ids.shape[1]
        self.assertEqual(state.position_ids.shape, (3, 1, L))

        model.model.set_mrope_state(state.position_ids, state.rope_deltas)
        model.model.set_visual_state(
            visual_pos_masks=input_ids == IMAGE_TOKEN_ID,
            deepstack_visual_embeds=deepstack,
        )
        cache = make_prompt_cache(model)
        out = model(input_ids, cache=cache, input_embeddings=embeddings)
        self.assertEqual(out.shape, (1, L, args.vocab_size))
        out = model(mx.argmax(out[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(out.shape, (1, 1, args.vocab_size))
        model.model.reset_mrope_state()
        model.model.reset_visual_state()

        after_reset = model(text_inputs)
        self.assertTrue(mx.array_equal(baseline, after_reset))

    def test_deepstack_changes_visual_positions(self):
        # Deepstack injection must alter the logits (it feeds mid-layer visual
        # features), and clearing ONLY the visual state must return the model
        # to the plain injected-embeddings result.
        model, _ = tiny_model()
        input_ids, embeddings, state, _, deepstack = vision_prompt_state(model)

        model.model.set_mrope_state(state.position_ids, state.rope_deltas)
        without_ds = model(input_ids, input_embeddings=embeddings)

        model.model.set_visual_state(
            visual_pos_masks=input_ids == IMAGE_TOKEN_ID,
            deepstack_visual_embeds=deepstack,
        )
        with_ds = model(input_ids, input_embeddings=embeddings)
        self.assertFalse(mx.allclose(without_ds, with_ds, atol=1e-6))

        model.model.reset_visual_state()
        cleared = model(input_ids, input_embeddings=embeddings)
        model.model.reset_mrope_state()
        self.assertTrue(mx.array_equal(without_ds, cleared))

    def test_chunked_prefill_matches_single_shot(self):
        # Splitting a vision prompt across prefill chunks — through the middle
        # of the image run — must produce the same logits as a single-shot
        # prefill (positions AND deepstack rows are sliced by cache offset).
        model, _ = tiny_model()
        input_ids, embeddings, state, _, deepstack = vision_prompt_state(model)

        model.model.set_mrope_state(state.position_ids, state.rope_deltas)
        model.model.set_visual_state(
            visual_pos_masks=input_ids == IMAGE_TOKEN_ID,
            deepstack_visual_embeds=deepstack,
        )
        cache = make_prompt_cache(model)
        single = model(input_ids, cache=cache, input_embeddings=embeddings)

        cache = make_prompt_cache(model)
        model(input_ids[:, :3], cache=cache, input_embeddings=embeddings[:, :3])
        chunked = model(
            input_ids[:, 3:], cache=cache, input_embeddings=embeddings[:, 3:]
        )
        model.model.reset_mrope_state()
        model.model.reset_visual_state()

        self.assertTrue(mx.allclose(single[:, 3:], chunked, atol=1e-4))

    def test_sanitize(self):
        model, args = tiny_model()

        # Raw HF checkpoint layout: thinker.model.* text keys, separate experts,
        # vision/audio towers, and the full TTS stack.
        raw = {
            "thinker.model.embed_tokens.weight": mx.zeros((1,)),
            "thinker.model.layers.0.self_attn.q_proj.weight": mx.zeros((1,)),
            "thinker.model.norm.weight": mx.zeros((1,)),
            "thinker.lm_head.weight": mx.zeros((1,)),
            "thinker.visual.patch_embed.proj.weight": mx.zeros((1,)),
            "thinker.audio_tower.conv1.weight": mx.zeros((1,)),
            "talker.model.layers.0.mlp.gate.weight": mx.zeros((1,)),
            "code2wav.decoder.weight": mx.zeros((1,)),
        }
        for n in ["up_proj", "down_proj", "gate_proj"]:
            for e in range(args.num_experts):
                raw[f"thinker.model.layers.1.mlp.experts.{e}.{n}.weight"] = mx.zeros(
                    (2, 2)
                )
        sanitized = model.sanitize(raw)
        self.assertEqual(
            set(sanitized),
            {
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
                "model.layers.1.mlp.switch_mlp.up_proj.weight",
                "model.layers.1.mlp.switch_mlp.down_proj.weight",
                "model.layers.1.mlp.switch_mlp.gate_proj.weight",
            },
        )
        self.assertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"].shape,
            (args.num_experts, 2, 2),
        )

        # mlx-vlm conversion layout: already-stacked experts under
        # thinker.language_model.* and a renamed vision tower.
        converted = {
            "thinker.language_model.model.embed_tokens.weight": mx.zeros((1,)),
            "thinker.language_model.model.layers.1.mlp.switch_mlp.up_proj.weight": mx.zeros((1,)),
            "thinker.language_model.lm_head.weight": mx.zeros((1,)),
            "thinker.vision_tower.patch_embed.proj.weight": mx.zeros((1,)),
            "thinker.audio_tower.conv1.weight": mx.zeros((1,)),
            "talker.model.embed_tokens.weight": mx.zeros((1,)),
        }
        self.assertEqual(
            set(model.sanitize(converted)),
            {
                "model.embed_tokens.weight",
                "model.layers.1.mlp.switch_mlp.up_proj.weight",
                "lm_head.weight",
            },
        )


if __name__ == "__main__":
    unittest.main()
