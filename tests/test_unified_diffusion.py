# Copyright © 2026 Apple Inc.
#
# Block-diffusion generation mode: the diffusion_gemma text model (encoder /
# canvas-decoder over shared weights) and the canvas-denoising loop in
# mlx_lm.diffusion_generate.

import unittest
from queue import Queue
from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.diffusion_generate import (
    diffusion_generate,
    is_diffusion_model,
    stream_diffusion_generate,
)
from mlx_lm.models import diffusion_gemma
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.server import Response, ResponseGenerator

CONFIG = {
    "model_type": "diffusion_gemma",
    "canvas_length": 16,
    "generation_config": {"max_denoising_steps": 3},
    "text_config": {
        "model_type": "diffusion_gemma_text",
        "vocab_size": 256,
        "hidden_size": 128,
        "intermediate_size": 96,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 32,
        "global_head_dim": 32,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 512,
        "sliding_window": 8,
        "final_logit_softcapping": 30.0,
        "num_experts": 4,
        "top_k_experts": 2,
    },
}

VOCAB = CONFIG["text_config"]["vocab_size"]
CANVAS = CONFIG["canvas_length"]


class StubDetokenizer:
    """Streaming-detokenizer contract, over a trivial <id> rendering."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.offset = 0
        self.text = ""

    def add_token(self, token):
        self.text += f"<{token}>"

    def finalize(self):
        pass

    @property
    def last_segment(self):
        segment = self.text[self.offset :]
        self.offset = len(self.text)
        return segment


class StubTokenizer:
    eos_token_ids = set()

    @property
    def detokenizer(self):
        return StubDetokenizer()


def make_model():
    args = diffusion_gemma.ModelArgs.from_dict(CONFIG)
    return diffusion_gemma.Model(args)


class TestUnifiedDiffusion(unittest.TestCase):
    def test_model(self):
        # The standard mlx-lm forward contract (encoder pass): tokens → logits,
        # then a cached prefill + decode step.
        model = make_model()
        self.assertEqual(len(model.layers), 4)
        self.assertEqual(model.model_type, "diffusion_gemma")
        # Both attention families must be exercised.
        self.assertEqual(model.text_args.layer_types.count("full_attention"), 1)

        inputs = mx.array([[0, 1]])
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (1, 2, VOCAB))

        cache = make_prompt_cache(model)
        outputs = model(inputs, cache=cache)
        self.assertEqual(outputs.shape, (1, 2, VOCAB))

        outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
        self.assertEqual(outputs.shape, (1, 1, VOCAB))

    def test_diffusion_protocol(self):
        model = make_model()
        self.assertTrue(is_diffusion_model(model))

        cache = make_prompt_cache(model)
        # Prompt longer than the sliding window, so the decoder's windowed
        # context masks and key trimming both trigger.
        prompt = mx.arange(10)[None]
        model.diffusion_extend_cache(prompt, cache=cache)
        offsets = [c.offset for c in cache]
        self.assertEqual(offsets, [10] * 4)

        canvas = mx.random.randint(0, VOCAB, (1, CANVAS))
        masks = model.diffusion_decoder_masks(canvas, cache)
        self.assertIsNone(masks["full_attention"])
        # 10 cached positions + 16 canvas positions, canvas rows only.
        self.assertEqual(masks["sliding_attention"].shape, (1, 1, CANVAS, 26))

        logits = model.diffusion_decoder_logits(canvas, cache=cache, masks=masks)
        self.assertEqual(logits.shape, (1, CANVAS, VOCAB))
        # Canvas passes must not write into the cache.
        self.assertEqual([c.offset for c in cache], offsets)

        context = model.diffusion_prepare_self_conditioning()
        conditioned = model.diffusion_self_conditioning(logits, context)
        self.assertEqual(conditioned.shape, (1, CANVAS, 128))
        logits = model.diffusion_decoder_logits(
            canvas, cache=cache, self_conditioning=conditioned, masks=masks
        )
        self.assertEqual(logits.shape, (1, CANVAS, VOCAB))

    def test_generate_streams_blocks_and_leaves_text_path_untouched(self):
        model = make_model()
        tokenizer = StubTokenizer()

        inputs = mx.array([[1, 2, 3]])
        baseline = model(inputs)

        mx.random.seed(0)
        responses = list(
            stream_diffusion_generate(
                model, tokenizer, [1, 2, 3], max_tokens=20, temperature=0.0
            )
        )
        # One response per accepted token: two 16-token canvases, capped at 20.
        self.assertEqual(len(responses), 20)
        self.assertEqual(responses[-1].generation_tokens, 20)
        self.assertEqual(responses[-1].finish_reason, "length")
        self.assertTrue(responses[-1].block_complete)
        # The first block's final token marks the block boundary for chunking.
        self.assertTrue(responses[15].block_complete)
        self.assertFalse(any(r.block_complete for r in responses[:15]))
        self.assertEqual([r.block_index for r in responses], [1] * 16 + [2] * 4)
        self.assertTrue(all(r.finish_reason is None for r in responses[:-1]))
        self.assertTrue(all(r.logprob <= 0.0 for r in responses))

        text = "".join(r.text for r in responses)
        self.assertEqual(text.count("<"), 20)

        mx.random.seed(0)
        self.assertEqual(
            diffusion_generate(model, tokenizer, [1, 2, 3], max_tokens=20), text
        )

        # Diffusion generation must leave text-only forwards byte-identical.
        self.assertTrue(mx.array_equal(baseline, model(inputs)))

    def test_eos_stops_generation(self):
        model = make_model()
        tokenizer = StubTokenizer()

        # Every token is an eos token → the very first accepted token stops it.
        tokenizer.eos_token_ids = set(range(VOCAB))
        responses = list(
            stream_diffusion_generate(model, tokenizer, [1, 2, 3], max_tokens=20)
        )
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].finish_reason, "stop")
        self.assertEqual(responses[0].generation_tokens, 0)
        self.assertEqual(responses[0].text, "")

    def test_sanitize_loads_mlx_vlm_checkpoint(self):
        # An mlx-vlm-converted checkpoint: fused expert projections, tied
        # encoder weights (only layer scalars stored), a vision tower, and the
        # vestigial keys the reference sanitize drops.
        model = make_model()
        weights = {}
        for k, v in tree_flatten(model.parameters()):
            if ".experts.switch_glu.gate_proj.weight" in k:
                up = dict(tree_flatten(model.parameters()))[
                    k.replace("gate_proj", "up_proj")
                ]
                fused = k.replace(".switch_glu.gate_proj.weight", ".gate_up_proj.weight")
                weights[fused] = mx.concatenate([v, up], axis=-2)
            elif ".experts.switch_glu.up_proj.weight" in k:
                continue
            elif ".experts.switch_glu.down_proj.weight" in k:
                weights[k.replace(".switch_glu.down_proj", ".down_proj")] = v
            else:
                weights[k] = v
        weights["lm_head.weight"] = mx.zeros((VOCAB, 128))
        weights["model.decoder.layers.0.self_attn.rotary_emb.inv_freq"] = mx.zeros((4,))
        weights["model.encoder.vision_tower.blocks.0.attn.qkv.weight"] = mx.zeros((8, 8))
        weights["model.encoder.embed_vision.mm_input_projection.weight"] = mx.zeros((8, 8))
        weights["model.encoder.language_model.layers.0.dropped.weight"] = mx.zeros((8,))

        model.load_weights(list(model.sanitize(weights).items()), strict=True)


class TestServerDiffusionRouting(unittest.TestCase):
    """_serve_diffusion, driven directly (no HTTP): block-sized chunks out,
    per-token accounting, textual stop words."""

    def serve(self, stop_words=(), should_stop=False):
        rg = ResponseGenerator.__new__(ResponseGenerator)  # skip the worker thread
        rg.model_provider = SimpleNamespace(
            draft_model=None,
            cli_args=SimpleNamespace(prefill_step_size=512),
        )
        args = SimpleNamespace(
            stop_words=list(stop_words),
            max_tokens=20,
            sampling=SimpleNamespace(temperature=0.0),
        )
        ctx = SimpleNamespace(_should_stop=should_stop)
        rqueue = Queue()
        rg._serve_diffusion(
            rqueue,
            SimpleNamespace(vision=None),
            args,
            make_model(),
            StubTokenizer(),
            [1, 2, 3],
            ctx,
            lambda processed, total: None,
        )
        responses = []
        while (item := rqueue.get_nowait()) is not None:
            responses.append(item)
        self.assertTrue(rqueue.empty())
        return responses

    def test_blocks_stream_as_chunks(self):
        responses = self.serve()
        self.assertTrue(all(isinstance(r, Response) for r in responses))
        # One Response per accepted token (usage/logprob accounting)...
        self.assertEqual(len(responses), 20)
        # ...but text only on block boundaries: two 16-token canvases capped at 20.
        chunks = [r.text for r in responses if r.text]
        self.assertEqual(len(chunks), 2)
        self.assertEqual("".join(chunks).count("<"), 20)
        self.assertEqual(responses[-1].finish_reason, "length")
        self.assertTrue(all(r.finish_reason is None for r in responses[:-1]))

    def test_stop_word_truncates_block(self):
        # Every stub token renders as "<id>", so "<" always matches: the first
        # block must flush empty and finish with "stop".
        responses = self.serve(stop_words=["<"])
        self.assertEqual(responses[-1].finish_reason, "stop")
        self.assertEqual("".join(r.text for r in responses), "")
        self.assertLessEqual(len(responses), 16)

    def test_client_disconnect_stops_generation(self):
        responses = self.serve(should_stop=True)
        self.assertEqual(len(responses), 1)

    def test_non_diffusion_models_unaffected(self):
        self.assertFalse(is_diffusion_model(SimpleNamespace(model_type="llama")))


if __name__ == "__main__":
    unittest.main()
