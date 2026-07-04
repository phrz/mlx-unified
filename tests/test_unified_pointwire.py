# Copyright © 2026 Apple Inc.
"""Central wiring for molmo_point (mlx-unified): the bridge's "molmo-point"
capability branch in MlxVlmBridge.prepare (six side-state tensors lifted from
mlx-vlm's Model._image_cache), point extraction/rendering helpers (differential
against mlx-vlm's reference point_utils), the fallback streaming detokenizer for
conversions whose tokenizer lacks the <POINT_k> added tokens, and an end-to-end
_serve_single run with a scripted tiny molmo_point model — set_visual_state
wiring, fallback detokenization, the trailing pixel-coordinate chunk, and the
reset in the finally block. No checkpoints are touched."""

import unittest
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from mlx_lm.multimodal import (
    BYPASS_CACHE_SIDES,
    TEXT_SIDE,
    MlxVlmBridge,
    extract_image_points,
    render_image_points,
)
from mlx_lm.server import (
    CompletionRequest,
    GenerationArguments,
    LogitsProcessorArguments,
    ModelDescription,
    PointStreamingDetokenizer,
    Response,
    ResponseGenerator,
    SamplingArguments,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper


def make_image_cache(L=7, hidden=4, P=2, S=3, vit_dim=6):
    """A consistent fake of mlx-vlm molmo_point's Model._image_cache: image
    tokens at positions 2..3, one valid pooled row per image token."""
    return {
        "token_pooling": mx.array([[[0, 1, -1], [2, -1, -1]]]),
        "vit_features": mx.random.normal((1, P, S, vit_dim)),
        "image_features": mx.random.normal((2, 1, hidden)),
        "image_token_offsets": mx.array([0]),
        "is_image_token": mx.array([[False, False, True, True, False, False, False]]),
        "is_indexable_image_token": mx.array(
            [[False, False, True, True, False, False, False]]
        ),
    }


MOLMO_POINT_CONFIG = {
    "model_type": "molmo_point",
    "vision_config": {"hidden_size": 6},
    "text_config": {"vocab_size": 100, "additional_vocab_size": 8},
    "no_more_points_class": True,
    "patch_location": "3x3",
}


def run_prepare(config, inputs, vlm, processor=None):
    """Drive MlxVlmBridge.prepare with a fake vlm/processor and mocked
    prepare_inputs — the checkpoint is never touched."""
    bridge = MlxVlmBridge(Path("/nonexistent"), config)
    bridge._vlm = vlm  # _ensure_loaded no-ops once set
    bridge._processor = processor if processor is not None else object()
    with mock.patch("mlx_vlm.utils.prepare_inputs", return_value=inputs):
        return bridge.prepare("prompt", [])


class TestMolmoPointBridge(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(TEXT_SIDE.get("molmo_point"), "molmo-point")
        # patch keys index ABSOLUTE prompt positions: a cache-trimmed prefill
        # would leave them incomplete, so the bypass is a hard requirement
        self.assertIn("molmo-point", BYPASS_CACHE_SIDES)

    def prepare_with(self, cache, processor=None, config=MOLMO_POINT_CONFIG):
        feats = SimpleNamespace(
            inputs_embeds=mx.ones((1, 7, 4)), position_ids=None, rope_deltas=None
        )
        vlm = SimpleNamespace(get_input_embeddings=lambda *a, **k: feats)
        if cache is not None:
            vlm._image_cache = cache
        inputs = dict(
            input_ids=mx.array([[1, 2, 96, 96, 3, 4, 5]]),
            pixel_values=mx.ones((1, 3, 4)),
        )
        return run_prepare(config, inputs, vlm, processor=processor)

    def test_side_state_carried(self):
        cache = make_image_cache()
        meta = {
            "token_pooling": np.array([[0, 1, -1], [2, -1, -1]]),
            "subpatch_mapping": [np.arange(6).reshape(2, 3)],
            "image_sizes": [(60, 40)],
        }
        vp = self.prepare_with(cache, processor=SimpleNamespace(_pointing_metadata=meta))
        # the six tensors travel untouched, in _image_cache order
        for field in (
            "token_pooling",
            "vit_features",
            "image_features",
            "image_token_offsets",
            "is_image_token",
            "is_indexable_image_token",
        ):
            self.assertIs(getattr(vp, field), cache[field], field)
        # extended ids start at the model's total vocab
        self.assertEqual(vp.point_id_start, 108)
        # metadata merged with the two config flags extraction needs
        self.assertIs(vp.pointing_metadata["token_pooling"], meta["token_pooling"])
        self.assertIs(vp.pointing_metadata["subpatch_mapping"], meta["subpatch_mapping"])
        self.assertEqual(vp.pointing_metadata["image_sizes"], [(60, 40)])
        self.assertTrue(vp.pointing_metadata["no_more_points_class"])
        self.assertEqual(vp.pointing_metadata["patch_location"], "3x3")
        # cache policy + plain injection otherwise
        self.assertTrue(vp.bypass_cache)
        self.assertFalse(vp.single_prefill)
        self.assertIsNone(vp.position_ids)
        self.assertEqual(vp.embeddings.shape, (7, 4))

    def test_metadata_optional(self):
        # a processor without _pointing_metadata: side state still flows, only
        # the server-side coordinate post-process is unavailable
        vp = self.prepare_with(make_image_cache())
        self.assertIsNone(vp.pointing_metadata)
        self.assertEqual(vp.point_id_start, 108)
        self.assertIsNotNone(vp.token_pooling)

    def test_missing_image_cache_raises(self):
        with self.assertRaises(ValueError):
            self.prepare_with(None)


class TestExtractImagePoints(unittest.TestCase):
    POOLING = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, -1, -1]])  # (P=3, S=4)
    MAPPINGS = [
        np.array([[0, 1], [2, 3], [4, 5]]),  # image 0: (H=3, W=2) grid
        np.array([[6, 7, 8], [9, 10, 11]]),  # image 1: (H=2, W=3) grid
    ]
    SIZES = [(20, 30), (90, 60)]  # (w, h)

    def pointing(self, **overrides):
        base = dict(
            token_pooling=self.POOLING,
            subpatch_mapping=self.MAPPINGS,
            image_sizes=self.SIZES,
            no_more_points_class=True,
            patch_location="3x3",
        )
        base.update(overrides)
        return base

    def test_matches_reference(self):
        # differential against mlx-vlm's point_utils on identical inputs:
        # two valid triples (with and without the optional spaces), a padded
        # pooling slot (-1 matches no grid cell — skipped by both), and
        # surrounding text/digits that must not confuse the regex
        from mlx_vlm.models.molmo_point.point_utils import extract_points_from_text

        text = (
            "I see 3 things <POINT_1><POINT_5><POINT_14>12 and "
            "<POINT_2> <POINT_5> <POINT_8> 3 then <POINT_2><POINT_6><POINT_8>4 done"
        )
        pointing = self.pointing()
        mine = extract_image_points(text, pointing)
        reference = extract_points_from_text(
            text, pointing, no_more_points_class=True, patch_location="3x3"
        )
        self.assertEqual(len(mine), len(reference))
        self.assertEqual(len(mine), 2)  # the -1 pooling slot matched nowhere
        for m, r in zip(mine, reference):
            self.assertEqual(m[0], int(r[0]))
            self.assertEqual(m[1], int(r[1]))
            self.assertEqual(m[2], float(r[2]))
            self.assertEqual(m[3], float(r[3]))
        # spot-check the first: patch 1/subpatch 1 -> vit id 5 at cell (2,1) of
        # image 0, location 6 -> (+2.5, +0.5)*0.33, grid (3,2), image 20x30
        ex, img, x, y = mine[0]
        self.assertEqual((ex, img), (12, 0))
        self.assertAlmostEqual(x, (1 + 2.5 * 0.33) / 2 * 20, places=10)
        self.assertAlmostEqual(y, (2 + 0.5 * 0.33) / 3 * 30, places=10)

    def test_malformed_ids_skipped_not_raised(self):
        # out-of-range patch / negative subpatch ids can't come from the model
        # (the in-model ordering mask forbids them) but must not 500 the server
        # on adversarial text — the reference would IndexError / wrap around
        text = "<POINT_99><POINT_5><POINT_14>1 <POINT_1><POINT_2><POINT_14>5"
        self.assertEqual(extract_image_points(text, self.pointing()), [])

    def test_no_digits_no_match(self):
        text = "<POINT_1><POINT_5><POINT_14> trailing words"
        self.assertEqual(extract_image_points(text, self.pointing()), [])

    def test_render(self):
        self.assertEqual(render_image_points([]), "")
        self.assertEqual(
            render_image_points([(12, 0, 18.25, 21.65), (3, 1, 4.95, 34.95)]),
            '\n<point id="12" image="0" x="18.2" y="21.6"/>'
            '<point id="3" image="1" x="5.0" y="35.0"/>',
        )


class BufferingInner:
    """A worst-case inner detokenizer: text stays pending until finalize, so the
    wrapper's flush-before-point ordering is actually exercised."""

    VOCAB = {0: "</s>", 10: "a", 11: "b", 12: "c"}

    def __init__(self):
        self.reset()

    def reset(self):
        self.offset = 0
        self.text = ""
        self.tokens = []
        self._pending = ""

    def add_token(self, token):
        self.tokens.append(token)
        self._pending += self.VOCAB[token]

    def finalize(self):
        self.text += self._pending
        self._pending = ""

    @property
    def last_segment(self):
        segment = self.text[self.offset :]
        self.offset = len(self.text)
        return segment


class TestPointStreamingDetokenizer(unittest.TestCase):
    def test_interleaving_and_flush_order(self):
        det = PointStreamingDetokenizer(BufferingInner(), 100)
        segments = []
        for token in (10, 11, 105, 12, 103):
            det.add_token(token)
            segments.append(det.last_segment)
        det.finalize()
        segments.append(det.last_segment)
        # pending inner text ("ab", then "c") is flushed BEFORE each point text
        self.assertEqual(segments, ["", "", "ab<POINT_5>", "", "c<POINT_3>", ""])
        self.assertEqual(det.text, "ab<POINT_5>c<POINT_3>")
        # extended ids never reached the inner detokenizer
        self.assertEqual(det.tokens, [10, 11, 105, 12, 103])
        self.assertEqual(det._detokenizer.tokens, [10, 11, 12])

    def test_reset(self):
        det = PointStreamingDetokenizer(BufferingInner(), 100)
        det.add_token(101)
        det.reset()
        self.assertEqual((det.text, det.tokens, det.offset), ("", [], 0))
        det.add_token(102)
        det.finalize()
        self.assertEqual(det.text, "<POINT_2>")


# ---- end-to-end _serve_single ----------------------------------------------

POINT_VOCAB = {"a": 10, " ": 11, ",": 12, "b": 13, "1": 14, "\n": 15}


class FakeHFTokenizer:
    """The minimal HF-tokenizer surface TokenizerWrapper + _serve_single touch.
    Its vocab has NO <POINT_k> tokens, so the fallback detokenizer must fire."""

    chat_template = None
    eos_token_id = 0
    bos_token = None

    def __init__(self):
        self._rev = {v: k for k, v in POINT_VOCAB.items()}
        self._rev[0] = "</s>"

    def get_vocab(self):
        return dict(POINT_VOCAB)

    def encode(self, text, add_special_tokens=False):
        return [POINT_VOCAB[c] for c in text]

    def decode(self, ids):
        return "".join(self._rev.get(i, "") for i in ids)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._rev.get(ids)
        return [self._rev.get(i) for i in ids]


def make_generation_args(max_tokens=8):
    return GenerationArguments(
        model=ModelDescription(model="tiny", draft="tiny", adapter=None),
        sampling=SamplingArguments(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            xtc_probability=0.0,
            xtc_threshold=0.0,
        ),
        logits=LogitsProcessorArguments(
            logit_bias=None,
            repetition_penalty=0.0,
            repetition_context_size=20,
            presence_penalty=0.0,
            presence_context_size=20,
            frequency_penalty=0.0,
            frequency_context_size=20,
        ),
        stop_words=[],
        max_tokens=max_tokens,
        num_draft_tokens=0,
        logprobs=False,
        top_logprobs=-1,
        seed=None,
        chat_template_kwargs=None,
    )


class TestServeSinglePointWiring(unittest.TestCase):
    def scripted_model(self, script):
        """The real tiny molmo_point model with the last position's logits
        replaced by a scripted one-hot on every SAMPLED call (the final
        1-token prefill step and each decode step — i.e. once the cache holds
        the full prompt). The real forward still runs, so patch-key capture,
        extended-id embedding mapping and the visual state hooks are all
        exercised; the script only makes the sampled ids deterministic."""
        from mlx_lm.models import molmo_point

        args = molmo_point.ModelArgs(
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
            vit_layers=[-3, -9],
            patch_embed_dim=16,
            patch_token_id=102,
            subpatch_token_id=103,
            location_token_id=104,
        )

        class ScriptedModel(molmo_point.Model):
            def __call__(self, inputs, cache=None, input_embeddings=None):
                out = super().__call__(
                    inputs, cache=cache, input_embeddings=input_embeddings
                )
                sampled_call = (
                    inputs is not None
                    and inputs.shape[1] == 1
                    and cache is not None
                    and cache[0].offset >= self._prompt_len
                )
                if sampled_call and self._script:
                    forced = self._script.pop(0)
                    out = mx.full(out.shape, -30.0, dtype=out.dtype)
                    out = out.at[..., -1, forced].add(60.0)
                return out

        model = ScriptedModel(args)
        mx.random.seed(0)
        model.update(
            tree_map(lambda p: 0.02 * mx.random.normal(p.shape), model.parameters())
        )
        model._script = list(script)
        return model

    def vision_prompt(self, model):
        """B=1, L=10 prompt with image tokens at 2..5 (2,3,4 indexable), P=4
        pooled rows, S=3 — the tiny scenario from test_unified_molmopoint,
        carried on a VisionPrompt the way the bridge would."""
        from mlx_lm.multimodal import VisionPrompt

        mx.random.seed(1)
        tokens = [1, 2, 96, 96, 96, 97, 3, 4, 5, 6]
        model._prompt_len = len(tokens)
        embeds = model.lm.model.wte(mx.array([tokens]))
        return VisionPrompt(
            tokens=tokens,
            embeddings=embeds.squeeze(0),
            token_pooling=mx.array([[[0, 1, 2], [3, 4, -1], [5, -1, -1], [6, 7, 8]]]),
            vit_features=mx.random.normal((1, 4, 3, 12)),
            image_features=mx.random.normal((4, 64)),
            image_token_offsets=mx.array([0]),
            is_image_token=mx.array([[0, 0, 1, 1, 1, 1, 0, 0, 0, 0]]).astype(mx.bool_),
            is_indexable_image_token=mx.array(
                [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]]
            ).astype(mx.bool_),
            point_id_start=108,
            pointing_metadata={
                "token_pooling": np.array([[0, 1, 2], [3, 4, -1], [5, -1, -1], [6, 7, 8]]),
                "subpatch_mapping": [np.arange(6).reshape(2, 3)],
                "image_sizes": [(60, 40)],
                "no_more_points_class": True,
                "patch_location": "3x3",
            },
            bypass_cache=True,
            image_fingerprint="test",
        )

    def serve(self, model, vision):
        tokenizer = TokenizerWrapper(FakeHFTokenizer())
        fake = SimpleNamespace(
            model_provider=SimpleNamespace(
                model=model,
                tokenizer=tokenizer,
                draft_model=None,
                model_key=("tiny",),
            ),
            prompt_cache=None,  # bypass_cache: must never be touched
            cli_args=SimpleNamespace(prefill_step_size=3),  # chunked prefill
            _state_machine_cache={},
            _log_cache_stats=lambda: None,
            _tokenize=lambda tok, req, a: (
                vision.tokens,
                [vision.tokens],
                ["assistant"],
                "normal",
            ),
            _is_distributed=False,
        )
        fake._make_state_machine = lambda *a, **k: ResponseGenerator._make_state_machine(
            fake, *a, **k
        )
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[],
            tools=None,
            role_mapping=None,
            vision=vision,
        )
        rqueue = Queue()
        ResponseGenerator._serve_single(fake, (rqueue, request, make_generation_args()))
        items = []
        while True:
            try:
                item = rqueue.get_nowait()
            except Empty:
                self.fail("rqueue never received the None sentinel")
            if isinstance(item, Exception):
                raise item
            if item is None:
                return items
            items.append(item)

    def test_point_run_end_to_end(self):
        # patch 1 (ext 109) -> subpatch 0 (ext 113) -> location 2 (ext 118) ->
        # the example-id digit '1' (vocab 14) -> EOS
        model = self.scripted_model([109, 113, 118, 14, 0])
        vision = self.vision_prompt(model)
        items = self.serve(model, vision)

        responses = [i for i in items if isinstance(i, Response)]
        # 4 generated tokens + the finalize chunk + the trailing coords chunk
        self.assertEqual(len(responses), 6)
        # the fallback detokenizer decoded the extended ids (the fake tokenizer
        # cannot: it has no <POINT_k> tokens)
        self.assertEqual(
            [r.text for r in responses[:4]],
            ["<POINT_1>", "<POINT_5>", "<POINT_10>", "1"],
        )
        self.assertEqual([r.token for r in responses[:4]], [109, 113, 118, 14])
        # EOS ends generation through the state machine (its text is blanked
        # downstream by _process_control_tokens — which is exactly why the
        # coordinates ride a separate trailing chunk)
        self.assertEqual(responses[4].finish_reason, "stop")
        self.assertEqual(responses[4].match, (0,))
        # the trailing chunk: point run -> pixel coordinates. patch 1/subpatch 0
        # -> vit id 3 at cell (1,0) of the 2x3 grid, location 2 -> (+0.5, +2.5)
        # *0.33, image 60x40
        coords = responses[5]
        self.assertEqual(coords.text, '\n<point id="1" image="0" x="3.3" y="36.5"/>')
        self.assertIsNone(coords.finish_reason)
        self.assertIsNone(coords.match)
        # the finally block reset the visual side state
        self.assertIsNone(model.model._point_state)

    def test_no_points_no_trailing_chunk(self):
        # a text-only answer ('ab' then EOS): nothing to extract, so the output
        # must be byte-identical to a plain generation — no trailing chunk
        model = self.scripted_model([10, 13, 0])
        vision = self.vision_prompt(model)
        items = self.serve(model, vision)
        responses = [i for i in items if isinstance(i, Response)]
        self.assertEqual(len(responses), 3)
        self.assertEqual([r.text for r in responses[:2]], ["a", "b"])
        self.assertEqual(responses[2].finish_reason, "stop")
        self.assertIsNone(model.model._point_state)


if __name__ == "__main__":
    unittest.main()
