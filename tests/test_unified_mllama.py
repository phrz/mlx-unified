# Copyright © 2026 Apple Inc.

import copy
import unittest

import mlx.core as mx
from mlx.utils import tree_map

from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

VISION_TOKENS = 5


def tiny_mllama_args():
    from mlx_lm.models import mllama

    return mllama.ModelArgs.from_dict(
        {
            "model_type": "mllama",
            "text_config": {
                "model_type": "mllama_text_model",
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1000,
                "max_position_embeddings": 2048,
                "rope_theta": 10000.0,
                "cross_attention_layers": [1, 3],
            },
            "vision_config": {},  # non-field keys are ignored
        }
    )


def randomize(model):
    mx.random.seed(0)
    model.update(
        tree_map(lambda p: 0.02 * mx.random.normal(p.shape), model.parameters())
    )
    return model


def tiny_model():
    from mlx_lm.models import mllama

    args = tiny_mllama_args()
    return randomize(mllama.Model(args)), args


def vision_prompt_state(model, args):
    """A server-shaped vision prompt: one <|image|> placeholder (living in the 8
    extra embedding rows), embeddings straight from embed_tokens (mllama merges
    nothing — vision arrives only via cross-attention), random projected vision
    states, and the PREPARED masks as mlx-vlm's get_input_embeddings returns
    them: row 0 precedes the image (fully masked out → all-zero mask row and a 0
    full-row gate), row 2 sees only the first 3 vision positions."""
    text = args.text_config
    image_token_id = text.vocab_size  # first of the 8 extra rows
    input_ids = mx.array([[2, image_token_id, 5, 6, 7, 8]])
    L = input_ids.shape[1]
    embeddings = model.model.embed_tokens(input_ids)
    states = 0.02 * mx.random.normal((1, VISION_TOKENS, text.hidden_size))
    neg = -1e9
    rows = [[0.0] * VISION_TOKENS for _ in range(L)]
    rows[2] = [0.0, 0.0, 0.0, neg, neg]
    mask = mx.array(rows).reshape(1, 1, L, VISION_TOKENS)
    full_row = mx.array([0.0] + [1.0] * (L - 1)).reshape(1, 1, L, 1)
    return input_ids, embeddings, states, mask, full_row


class TestUnifiedMllama(unittest.TestCase):
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

    def test_model(self):
        model, args = tiny_model()
        self.model_test_runner(
            model,
            args.model_type,
            args.text_config.vocab_size,
            args.text_config.num_hidden_layers,
        )

    def test_layer_structure_and_cache(self):
        from mlx_lm.models import mllama

        model, args = tiny_model()
        for i, layer in enumerate(model.layers):
            expected = (
                mllama.CrossAttentionBlock
                if i in args.text_config.cross_attention_layers
                else mllama.TransformerBlock
            )
            self.assertIsInstance(layer, expected)
        cache = make_prompt_cache(model)
        for i, c in enumerate(cache):
            if i in args.text_config.cross_attention_layers:
                self.assertIsInstance(c, mllama.VisionKVCache)
            else:
                self.assertNotIsInstance(c, mllama.VisionKVCache)

    def test_injection_matches_tokens(self):
        # With no visual state, forwarding embed_tokens(inputs) as
        # input_embeddings must match the token forward exactly.
        model, _ = tiny_model()
        inputs = mx.array([[1, 2, 3, 4]])
        baseline = model(inputs)
        via_embeds = model(inputs, input_embeddings=model.model.embed_tokens(inputs))
        self.assertTrue(mx.array_equal(baseline, via_embeds))

    def test_cross_attention_flow_and_reset(self):
        # The full server-shaped flow: visual state set, embeddings injected,
        # cross-attn K/V cached at prefill and consumed on the decode step, then
        # reset — text-only behavior must be byte-identical to a never-touched
        # forward.
        model, args = tiny_model()
        vocab = args.text_config.vocab_size
        text_inputs = mx.array([[10, 11, 12, 13]])
        baseline_text = model(text_inputs)

        input_ids, embeddings, states, mask, full_row = vision_prompt_state(
            model, args
        )
        L = input_ids.shape[1]
        step_token = mx.array([[42]])

        # Text-only control over the SAME tokens (cross layers skipped).
        cache_t = make_prompt_cache(model)
        model(input_ids, cache=cache_t, input_embeddings=embeddings)
        decode_t = model(step_token, cache=cache_t)

        model.model.set_visual_state(
            cross_attention_states=states,
            cross_attention_mask=mask,
            full_text_row_masked_out_mask=full_row,
        )
        cache_v = make_prompt_cache(model)
        out = model(input_ids, cache=cache_v, input_embeddings=embeddings)
        self.assertEqual(out.shape, (1, L, vocab))
        # Prefill computed and cached the vision K/V once, per cross layer.
        for i in args.text_config.cross_attention_layers:
            self.assertFalse(cache_v[i].empty())
            self.assertEqual(cache_v[i].keys.shape[2], VISION_TOKENS)
        # Vision must actually change the logits vs the text-only control...
        decode_v = model(step_token, cache=cache_v)
        self.assertEqual(decode_v.shape, (1, 1, vocab))
        # ...including on the DECODE step: vision tokens stay live all generation.
        self.assertFalse(mx.allclose(decode_t, decode_v))

        model.model.reset_visual_state()
        self.assertTrue(mx.array_equal(baseline_text, model(text_inputs)))

    def test_chunked_prefill_matches_single_shot(self):
        # Splitting the vision prompt across prefill chunks must produce the
        # same logits as a single-shot prefill (mask rows are anchored at the
        # cache offset where prefill started; K/V reused from the first chunk).
        model, args = tiny_model()
        input_ids, embeddings, states, mask, full_row = vision_prompt_state(
            model, args
        )
        visual = dict(
            cross_attention_states=states,
            cross_attention_mask=mask,
            full_text_row_masked_out_mask=full_row,
        )

        model.model.set_visual_state(**visual)
        cache = make_prompt_cache(model)
        single = model(input_ids, cache=cache, input_embeddings=embeddings)

        model.model.set_visual_state(**visual)
        cache = make_prompt_cache(model)
        model(input_ids[:, :3], cache=cache, input_embeddings=embeddings[:, :3])
        chunked = model(
            input_ids[:, 3:], cache=cache, input_embeddings=embeddings[:, 3:]
        )
        model.model.reset_visual_state()

        self.assertTrue(mx.allclose(single[:, 3:], chunked, atol=1e-4))

    def test_trim_keeps_vision_kv(self):
        # Trimming text tokens must move the self-attn offsets back while
        # leaving the (non-text-positional) vision K/V untouched.
        model, args = tiny_model()
        input_ids, embeddings, states, mask, full_row = vision_prompt_state(
            model, args
        )
        model.model.set_visual_state(
            cross_attention_states=states,
            cross_attention_mask=mask,
            full_text_row_masked_out_mask=full_row,
        )
        cache = make_prompt_cache(model)
        model(input_ids, cache=cache, input_embeddings=embeddings)
        model.model.reset_visual_state()

        self.assertTrue(can_trim_prompt_cache(cache))
        self.assertEqual(trim_prompt_cache(cache, 2), 2)
        self.assertEqual(cache[0].offset, input_ids.shape[1] - 2)
        for i in args.text_config.cross_attention_layers:
            self.assertEqual(cache[i].keys.shape[2], VISION_TOKENS)

    def test_sanitize(self):
        from mlx_lm.models import mllama

        model, _ = tiny_model()
        # mlx-vlm-converted checkpoint namespace (matches raw HF <= 4.51).
        vlm = {
            "language_model.model.embed_tokens.weight": 0,
            "language_model.model.layers.0.self_attn.q_proj.weight": 0,
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": 0,
            "language_model.model.layers.1.cross_attn.q_proj.weight": 0,
            "language_model.model.layers.1.cross_attn.q_norm.weight": 0,
            "language_model.model.layers.1.cross_attn_attn_gate": 0,
            "language_model.model.layers.1.cross_attn_mlp_gate": 0,
            "language_model.model.norm.weight": 0,
            "language_model.lm_head.weight": 0,
            "vision_tower.patch_embedding.weight": 0,
            "multi_modal_projector.weight": 0,
        }
        self.assertEqual(
            sorted(model.sanitize(vlm)),
            [
                "language_model.lm_head.weight",
                "language_model.model.embed_tokens.weight",
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "language_model.model.layers.1.cross_attn.q_norm.weight",
                "language_model.model.layers.1.cross_attn.q_proj.weight",
                "language_model.model.layers.1.cross_attn_attn_gate",
                "language_model.model.layers.1.cross_attn_mlp_gate",
                "language_model.model.norm.weight",
            ],
        )
        # transformers >= 4.52 namespace: towers under "model.", flat language
        # keys, top-level lm_head.
        hf = {
            "model.language_model.embed_tokens.weight": 0,
            "model.language_model.layers.1.cross_attn.k_norm.weight": 0,
            "model.language_model.layers.1.cross_attn_attn_gate": 0,
            "model.vision_model.patch_embedding.weight": 0,
            "model.multi_modal_projector.weight": 0,
            "lm_head.weight": 0,
        }
        self.assertEqual(
            sorted(model.sanitize(hf)),
            [
                "language_model.lm_head.weight",
                "language_model.model.embed_tokens.weight",
                "language_model.model.layers.1.cross_attn.k_norm.weight",
                "language_model.model.layers.1.cross_attn_attn_gate",
            ],
        )


if __name__ == "__main__":
    unittest.main()
