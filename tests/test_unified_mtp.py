# Copyright © 2026 Apple Inc.
#
# mlx-unified: baked MTP (nextn) self-speculative draft head for
# deepseek_v32 / glm_moe_dsa (GLM-5.2). Covers head construction, the
# sanitize remap of model.layers.{N}.* → mtp.*, and the greedy draft
# recurrence (chained steps against the head's own KV cache).

import unittest

import mlx.core as mx

from mlx_lm.models import deepseek_v32

TINY_CONFIG = dict(
    model_type="deepseek_v32",
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=4,
    num_experts_per_tok=2,
    first_k_dense_replace=1,
    kv_lora_rank=16,
    q_lora_rank=32,
    qk_rope_head_dim=8,
    qk_nope_head_dim=16,
    v_head_dim=16,
    index_head_dim=16,
    index_n_heads=2,
    index_topk=4,
    max_position_embeddings=256,
)


def tiny_model(**overrides):
    args = deepseek_v32.ModelArgs.from_dict({**TINY_CONFIG, **overrides})
    return deepseek_v32.Model(args)


class TestMtpHead(unittest.TestCase):
    def test_head_construction_gated_by_config(self):
        self.assertIsNone(tiny_model().mtp)
        m = tiny_model(num_nextn_predict_layers=1)
        self.assertIsNotNone(m.mtp)
        # Structurally a decoder layer + the MTP couplings.
        self.assertTrue(hasattr(m.mtp, "self_attn"))
        for extra in ("enorm", "hnorm", "eh_proj", "shared_head"):
            self.assertTrue(hasattr(m.mtp, extra), extra)

    def test_sanitize_remaps_mtp_layer(self):
        m = tiny_model(num_nextn_predict_layers=1)
        a = mx.zeros((2, 2))
        weights = {
            "model.layers.0.input_layernorm.weight": a,
            "model.layers.2.eh_proj.weight": a,  # the MTP block (== num_hidden_layers)
            "model.layers.2.self_attn.q_a_proj.weight": a,
            "model.layers.2.shared_head.norm.weight": a,
            "model.layers.3.eh_proj.weight": a,  # a second nextn block: dropped (v1)
        }
        out = m.sanitize(dict(weights))
        self.assertIn("model.layers.0.input_layernorm.weight", out)
        self.assertIn("mtp.eh_proj.weight", out)
        self.assertIn("mtp.self_attn.q_a_proj.weight", out)
        self.assertIn("mtp.shared_head.norm.weight", out)
        self.assertNotIn("model.layers.2.eh_proj.weight", out)
        self.assertFalse(any("layers.3" in k for k in out))

    def test_sanitize_still_strips_when_disabled(self):
        m = tiny_model()  # no MTP head
        out = m.sanitize({"model.layers.2.eh_proj.weight": mx.zeros((2, 2))})
        self.assertEqual(out, {})

    def test_draft_chain(self):
        m = tiny_model(num_nextn_predict_layers=1)
        # Seed: run the main model, take the final-normed hidden of the last
        # position (exactly what model.model() returns) + the token it produced.
        prompt = mx.array([[1, 2, 3]])
        hidden = m.model(prompt)[:, -1:, :]
        tok = mx.argmax(m.lm_head(hidden), axis=-1)

        cache = m.make_mtp_cache()
        logits1, out1 = m.mtp_draft_step(tok, hidden, cache)
        self.assertEqual(logits1.shape, (1, 1, TINY_CONFIG["vocab_size"]))
        self.assertEqual(out1.shape, (1, 1, TINY_CONFIG["hidden_size"]))
        self.assertTrue(bool(mx.all(mx.isfinite(logits1))))

        # Chain: the drafted token + the head's own hidden feed the next step,
        # and the head's KV cache grows by one entry per step.
        draft1 = mx.argmax(logits1, axis=-1)
        logits2, out2 = m.mtp_draft_step(draft1, out1, cache)
        self.assertTrue(bool(mx.all(mx.isfinite(logits2))))
        self.assertEqual(cache[0].offset, 2)

    def test_speculative_generate_loop(self):
        from mlx_lm.mtp_speculative import mtp_speculative_generate

        m = tiny_model(num_nextn_predict_layers=1)
        forwards = []
        out, st = mtp_speculative_generate(
            m,
            [1, 2, 3],
            max_tokens=12,
            num_draft=2,
            after_forward=lambda: forwards.append(1),
        )
        self.assertGreaterEqual(len(out), 12)  # rounds emit 1..K+1, may overshoot by K
        self.assertEqual(st.tokens, len(out))
        # Each round drafts num_draft tokens, except a final round clamped to
        # the remaining budget (K = min(num_draft, max_tokens - len(out))).
        self.assertLessEqual(st.proposed, 2 * st.rounds)
        self.assertGreaterEqual(st.proposed, 2 * st.rounds - 1)
        self.assertLessEqual(st.accepted, st.proposed)
        # prefill + one per verify round
        self.assertEqual(len(forwards), 1 + st.rounds)
        self.assertTrue(all(isinstance(t, int) for t in out))

    def test_glm_moe_dsa_wrapper_accepts_mtp_config(self):
        from mlx_lm.models import glm_moe_dsa

        args = glm_moe_dsa.ModelArgs.from_dict(
            {
                **TINY_CONFIG,
                "model_type": "glm_moe_dsa",
                "rope_parameters": {"rope_theta": 10000.0},
                # glm_moe_dsa's ModelArgs declares these without defaults.
                "routed_scaling_factor": 1.0,
                "topk_method": "noaux_tc",
                "scoring_func": "sigmoid",
                "norm_topk_prob": True,
                "n_group": 1,
                "topk_group": 1,
                "moe_layer_freq": 1,
                "rms_norm_eps": 1e-6,
                "attention_bias": False,
                "num_nextn_predict_layers": 1,
                "index_share_for_mtp_iteration": True,
            }
        )
        self.assertIsNotNone(glm_moe_dsa.Model(args).mtp)


if __name__ == "__main__":
    unittest.main()
