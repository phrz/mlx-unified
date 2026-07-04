# Copyright © 2026 Apple Inc.

import copy
import unittest

import mlx.core as mx

from mlx_lm.models import falcon_ocr
from mlx_lm.models.cache import make_prompt_cache

IMG, CLS, END = 17, 18, 19


def tiny_model():
    args = falcon_ocr.ModelArgs(
        model_type="falcon_ocr",
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=32,
        num_key_value_heads=2,
        vocab_size=256,
        intermediate_size=256,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = falcon_ocr.Model(args)
    # The golden frequencies are checkpoint data (zeros at init would make the
    # 2D rotary a no-op and hide plumbing bugs).
    model.model.freqs_cis_golden = mx.random.normal((4, 8, 2)) * 0.1
    return model, args


def falcon_vision_state(tokens, grid_h, grid_w):
    """The state falcon_ocr's get_input_embeddings emits for one image prompt:
    collapsed positions (an image block advances the counter once), golden h/w
    coordinates for image tokens, the decode delta, and the causal-or-same-
    image-block boolean mask. Mirrors mlx-vlm's get_rope_index/compute_pos_hw/
    create_falcon_ocr_mask."""
    pos, in_image, nxt = [], False, 0
    for t in tokens:
        if t == CLS and not in_image:
            in_image = True
        pos.append(nxt)
        if not in_image:
            nxt += 1
        if t == END and in_image:
            in_image = False
            nxt += 1
    position_ids = mx.array(pos, dtype=mx.int32)[None]
    delta = max(pos) + 1 - len(tokens)

    a = (grid_h / grid_w) ** 0.5
    b = (grid_w / grid_h) ** 0.5
    coords = [
        (-a + 2 * a * hi / max(grid_h - 1, 1), -b + 2 * b * wi / max(grid_w - 1, 1))
        for hi in range(grid_h)
        for wi in range(grid_w)
    ]
    hw = [[0.0, 0.0]] * len(tokens)
    it = iter(coords)
    for i, t in enumerate(tokens):
        if t == IMG:
            hw[i] = list(next(it))
    pos_hw = mx.array(hw)[None]

    ids = mx.array(tokens)
    soi = mx.cumsum((ids == CLS).astype(mx.int32))
    eoi = mx.cumsum((ids == END).astype(mx.int32))
    in_img = (soi - eoi) > 0
    blk = soi * in_img.astype(mx.int32)
    S = len(tokens)
    q = mx.arange(S)
    causal = q[:, None] >= q[None, :]
    same = in_img[:, None] & in_img[None, :] & (blk[:, None] == blk[None, :])
    mask = (causal | same).reshape(1, 1, S, S)
    return position_ids, pos_hw, mx.array([[delta]], dtype=mx.int32), mask


class TestUnifiedFalconOCR(unittest.TestCase):
    def test_model(self):
        model, args = tiny_model()
        self.assertEqual(len(model.layers), args.num_hidden_layers)
        self.assertEqual(model.model_type, "falcon_ocr")

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

    def test_trivial_visual_state_matches_plain(self):
        # Sequential positions, zero golden coordinates (identity rotation),
        # delta 0, and a plain causal mask must reproduce the text-only
        # forward exactly (both prefill and cached decode).
        model, args = tiny_model()
        inputs = mx.array([[3, 1, 4, 1, 5, 9]])
        L = inputs.shape[1]

        plain_cache = make_prompt_cache(model)
        plain_prefill = model(inputs, cache=plain_cache)
        plain_decode = model(mx.array([[7]]), cache=plain_cache)

        q = mx.arange(L)
        model.model.set_visual_state(
            position_ids=q[None],
            pos_hw=mx.zeros((1, L, 2)),
            rope_deltas=mx.zeros((1, 1), dtype=mx.int32),
            attention_mask_4d=(q[:, None] >= q[None, :]).reshape(1, 1, L, L),
        )
        vis_cache = make_prompt_cache(model)
        vis_prefill = model(inputs, cache=vis_cache)
        vis_decode = model(mx.array([[7]]), cache=vis_cache)
        model.model.reset_visual_state()

        self.assertTrue(mx.allclose(plain_prefill, vis_prefill, atol=1e-5))
        self.assertTrue(mx.allclose(plain_decode, vis_decode, atol=1e-5))

    def test_vision_injection_and_reset(self):
        # The server-shaped flow: install falcon visual state, prefill with
        # injected embeddings, decode a step, then reset and confirm the
        # text-only forward is byte-identical to a never-touched model.
        model, args = tiny_model()
        text_inputs = mx.array([[10, 11, 12, 13]])
        baseline = model(text_inputs)

        tokens = [5, CLS, IMG, IMG, IMG, IMG, END, 7, 8]
        input_ids = mx.array([tokens])
        position_ids, pos_hw, rope_deltas, mask = falcon_vision_state(tokens, 2, 2)
        # Image blocks collapse to a single 1D position; decode resumes there.
        self.assertEqual(position_ids.squeeze(0).tolist(), [0, 1, 1, 1, 1, 1, 1, 2, 3])
        self.assertEqual(rope_deltas.item(), 4 - len(tokens))

        embeddings = model.model.embed_tokens(input_ids)
        image_mask = (input_ids == IMG)[..., None]
        embeddings = mx.where(
            image_mask, mx.random.normal(embeddings.shape) * 0.02, embeddings
        )

        model.model.set_visual_state(
            position_ids=position_ids,
            pos_hw=pos_hw,
            rope_deltas=rope_deltas,
            attention_mask_4d=mask,
        )
        cache = make_prompt_cache(model)
        out = model(input_ids, cache=cache, input_embeddings=embeddings)
        self.assertEqual(out.shape, (1, len(tokens), args.vocab_size))

        # The bidirectional-within-image mask must actually reach attention:
        # image-token logits change when it is replaced by a causal mask.
        q = mx.arange(len(tokens))
        model.model._vis_mask = (q[:, None] >= q[None, :]).reshape(
            1, 1, len(tokens), len(tokens)
        )
        causal_out = model(input_ids, input_embeddings=embeddings)
        model.model._vis_mask = mask
        self.assertFalse(mx.allclose(out[:, 2:6], causal_out[:, 2:6], atol=1e-5))

        out = model(mx.argmax(out[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(out.shape, (1, 1, args.vocab_size))
        model.model.reset_visual_state()

        after_reset = model(text_inputs)
        self.assertTrue(mx.array_equal(baseline, after_reset))

    def test_chunked_prefill_after_image_block(self):
        # A chunk boundary AFTER the image block must reproduce single-shot
        # logits (stored positions/coordinates/mask are sliced by cache
        # offset). Boundaries inside a block are forbidden (single_prefill).
        model, args = tiny_model()
        tokens = [5, CLS, IMG, IMG, IMG, IMG, END, 7, 8]
        input_ids = mx.array([tokens])
        position_ids, pos_hw, rope_deltas, mask = falcon_vision_state(tokens, 2, 2)
        embeddings = model.model.embed_tokens(input_ids)

        model.model.set_visual_state(
            position_ids=position_ids,
            pos_hw=pos_hw,
            rope_deltas=rope_deltas,
            attention_mask_4d=mask,
        )
        cache = make_prompt_cache(model)
        single = model(input_ids, cache=cache, input_embeddings=embeddings)

        cache = make_prompt_cache(model)
        model(input_ids[:, :7], cache=cache, input_embeddings=embeddings[:, :7])
        chunked = model(
            input_ids[:, 7:], cache=cache, input_embeddings=embeddings[:, 7:]
        )
        model.model.reset_visual_state()

        self.assertTrue(mx.allclose(single[:, 7:], chunked, atol=1e-5))

    def test_sanitize(self):
        model, _ = tiny_model()

        # Raw TII checkpoint layout; w13 rows are gate/up-interleaved.
        w13 = mx.arange(8, dtype=mx.float32).reshape(4, 2)
        raw = {
            "tok_embeddings.weight": mx.zeros((1,)),
            "img_projector.weight": mx.zeros((1,)),
            "freqs_cis_golden": mx.zeros((4, 8, 2)),
            "layers.0.attention.wqkv.weight": mx.zeros((1,)),
            "layers.0.attention.sinks": mx.zeros((4,)),
            "layers.0.feed_forward.w13.weight": w13,
            "norm.weight": mx.zeros((1,)),
            "output.weight": mx.zeros((1,)),
        }
        sanitized = model.sanitize(raw)
        self.assertEqual(
            set(sanitized),
            {
                "model.embed_tokens.weight",
                "model.freqs_cis_golden",
                "model.layers.0.self_attn.wqkv.weight",
                "model.layers.0.self_attn.sinks",
                "model.layers.0.mlp.w13.weight",
                "model.norm.weight",
                "lm_head.weight",
            },
        )
        self.assertTrue(
            mx.array_equal(
                sanitized["model.layers.0.mlp.w13.weight"],
                mx.concatenate([w13[0::2], w13[1::2]], axis=0),
            )
        )

        # mlx-vlm conversion: already de-interleaved, language_model-prefixed,
        # with derived rope tables and the patch projector saved alongside.
        converted = {
            "language_model.model.embed_tokens.weight": mx.zeros((1,)),
            "language_model.model.layers.0.mlp.w13.weight": w13,
            "language_model.model.freqs_cis_golden": mx.zeros((4, 8, 2)),
            "language_model.model.cos_1d": mx.zeros((1,)),
            "language_model.model.sin_1d": mx.zeros((1,)),
            "language_model.model.img_projector.weight": mx.zeros((1,)),
            "language_model.model.norm.weight": mx.zeros((1,)),
            "language_model.lm_head.weight": mx.zeros((1,)),
        }
        sanitized = model.sanitize(converted)
        self.assertEqual(
            set(sanitized),
            {
                "model.embed_tokens.weight",
                "model.layers.0.mlp.w13.weight",
                "model.freqs_cis_golden",
                "model.norm.weight",
                "lm_head.weight",
            },
        )
        self.assertTrue(
            mx.array_equal(sanitized["model.layers.0.mlp.w13.weight"], w13)
        )

    def test_config_aliases(self):
        # The TII config.json uses bespoke flat names.
        args = falcon_ocr.ModelArgs.from_dict(
            {
                "model_type": "falcon_ocr",
                "dim": 768,
                "n_layers": 22,
                "n_heads": 16,
                "head_dim": 64,
                "n_kv_heads": 8,
                "vocab_size": 65536,
                "ffn_dim": 2304,
                "norm_eps": 1e-5,
                "max_seq_len": 8192,
                "rope_theta": 10000,
                "img_id": 227,
            }
        )
        self.assertEqual(args.hidden_size, 768)
        self.assertEqual(args.num_hidden_layers, 22)
        self.assertEqual(args.num_key_value_heads, 8)
        self.assertEqual(args.intermediate_size, 2304)
        self.assertEqual(args.max_position_embeddings, 8192)


if __name__ == "__main__":
    unittest.main()
