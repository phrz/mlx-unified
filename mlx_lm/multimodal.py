# Copyright © 2026 Apple Inc.
#
# mlx-unified: vision support for mlx_lm.server.
#
# The approach follows lmstudio-ai/mlx-engine (MIT): the TEXT model that generates is
# always mlx-lm's own (this package); mlx-vlm contributes the vision side. Rather than
# hand-wiring each architecture's tower/processor/merge, we call out to mlx-vlm's own
# per-arch `Model.get_input_embeddings()` — the standardized entry point every mlx-vlm
# model implements (processor → vision tower → feature splice → position computation) —
# and inject its `inputs_embeds` into generation via mlx-lm's `input_embeddings`
# parameter. New architectures work without new embedding code here.
#
# What remains irreducibly per-arch is the TEXT side: some architectures change the
# language model's forward semantics for vision (qwen-family: 3D multimodal RoPE;
# gemma4-family: bidirectional attention within image spans). Those need first-class
# support in the corresponding mlx_lm/models/<arch>.py — TEXT_SIDE below records what
# each arch needs and what this fork has implemented. Architectures whose text forward
# is unchanged by vision work with plain injection automatically.
#
# The mlx-vlm checkpoint is loaded LAZILY (mlx arrays stay unmaterialized until used),
# so only the vision components + embedding table actually occupy memory — the
# duplicate language-model weights inside the mlx-vlm object are never evaluated.
#
# build_qwen_image_mrope_state is adapted from mlx-engine's
# model_kit/batched_vision/qwen_mrope.py (MIT © LM Studio) — mlx-vlm's own position
# computation can mis-position later image runs in multi-image prompts.
#
# mlx-vlm is imported lazily — text-only use of mlx_lm never requires it.

import base64
import binascii
import hashlib
import io
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx

# What the TEXT model must support beyond plain embedding injection, per model_type.
#   "plain" — no text-side changes needed; injection just works (the default).
#   "mrope" — 3D/4D multimodal RoPE side state (set_mrope_state/reset_mrope_state on
#             the inner text model). Positions are built by the bridge: qwen-style
#             grid walks for QWEN_MROPE_TYPES, processor-supplied for hunyuan_vl.
#   "gemma-visual" — gemma4 family (implemented in this fork's gemma4_text):
#             mm_token_type_ids-driven bidirectional attention within image spans
#             (gemma4_unified), explicit per-layer inputs (E2B/E4B), an embed-scale
#             correction, and single-call prefill so an image span is never split.
#   "gemma3n-visual" — plain injection with two fixes: mlx-vlm's per_layer_inputs
#             are DISCARDED (gemma3n re-derives identical ones from the raw token
#             ids that generation passes alongside input_embeddings) and text
#             positions get the ×sqrt(hidden) embed scale mlx-vlm misses.
#   "ernie-visual" — qwen-style mrope PLUS mm_token_type_ids: image tokens route
#             through the second MoE expert group. Chunked prefill is fine.
#   "attn-mask-4d" — whole-prompt 4D attention-mask override (bidirectional
#             prefix/image spans: paligemma/gemma3/moondream) via
#             set_visual_state(attention_mask_4d=...); single-call prefill.
#   "mrope+deepstack" — qwen3_vl family: mrope plus mid-layer visual injection
#             (set_visual_state(visual_pos_masks=..., deepstack_visual_embeds=...)).
#   "granite-deepstack" — granite4_vision: deepstack with full-sequence feature
#             adds and explicit target layers; inputs_embeds arrive UNSCALED with
#             image positions zeroed (image content flows in via deepstack only).
#   "cross-attention" — mllama: fixed vision K/V states + per-text-row visibility
#             masks, consumed by interleaved cross-attention layers every forward.
#   "visual-lora" — zaya1: visual_pos_masks gate vision-LoRA weight deltas.
#   "falcon-visual" — falcon_ocr: image-collapsed 1D positions + per-head golden
#             2D rotary coords + a block-diagonal bidirectional mask, all via
#             set_visual_state (NOT set_mrope_state); single-call prefill.
TEXT_SIDE = {
    "qwen3_5": "mrope",
    "qwen3_5_moe": "mrope",
    "qwen2_vl": "mrope",
    "qwen2_5_vl": "mrope",
    "glm4v": "mrope",
    "glm4v_moe": "mrope",
    "glm_ocr": "mrope",
    "paddleocr_vl": "mrope",
    "hunyuan_vl": "mrope",
    "ernie4_5_moe_vl": "ernie-visual",
    "gemma4": "gemma-visual",
    "gemma4_unified": "gemma-visual",
    "gemma3n": "gemma3n-visual",
    "paligemma": "attn-mask-4d",
    "gemma3": "attn-mask-4d",
    "moondream2": "attn-mask-4d",
    "moondream3": "attn-mask-4d",
    "qwen3_vl": "mrope+deepstack",
    "qwen3_vl_moe": "mrope+deepstack",
    "qwen3_omni_moe": "mrope+deepstack",
    "granite4_vision": "granite-deepstack",
    "mllama": "cross-attention",
    "zaya1_vl": "visual-lora",
    "falcon_ocr": "falcon-visual",
}

# model_types whose qwen-style mrope positions we build ourselves (multi-image-correct)
# instead of trusting mlx-vlm's get_rope_index output — several archs (glm family,
# ernie, paddleocr) never return positions at all (mlx-vlm stashes them internally),
# and each was verified to reproduce build_qwen_image_mrope_state exactly.
QWEN_MROPE_TYPES = (
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
)

# Capabilities that carry multimodal rope positions in VisionPrompt.position_ids.
MROPE_SIDES = ("mrope", "ernie-visual", "mrope+deepstack")

# v1: capability classes whose side state does not yet compose with the
# image-fingerprinted prompt cache — the server runs these with a FRESH cache every
# request and inserts nothing (correctness first; cache reuse is a later
# optimization). mrope/plain/gemma-visual keep the existing cache path.
BYPASS_CACHE_SIDES = (
    "attn-mask-4d",
    "mrope+deepstack",
    "granite-deepstack",
    "cross-attention",
    "visual-lora",
    "falcon-visual",
)

# InputEmbeddingsFeatures fields that change forward semantics if dropped: consumed
# only by the capabilities listed, hard-rejected for every other arch — an
# unregistered arch silently losing one of these would generate plausible-looking
# garbage, the worst failure mode.
GUARDED_FEATURES = {
    "attention_mask_4d": (
        "bidirectional attention masks",
        ("attn-mask-4d", "falcon-visual"),
    ),
    "deepstack_visual_embeds": (
        "mid-layer visual injection",
        ("mrope+deepstack", "granite-deepstack"),
    ),
    "cross_attention_states": ("cross-attention", ("cross-attention",)),
    "per_layer_inputs": (
        "per-layer multimodal inputs",
        ("gemma-visual", "gemma3n-visual"),
    ),
}


def load_vision_encoder(model_path):
    """A vision bridge for the checkpoint at model_path, or None for text-only ones."""
    model_path = Path(model_path)
    try:
        with open(model_path / "config.json") as f:
            config = json.load(f)
    except OSError:
        return None
    # qwen3_omni_moe nests everything under thinker_config; falcon_ocr's flat TII
    # config has no vision_config at all (early fusion — vision lives inside the
    # language_model weights). molmo v1 flat configs also lack vision_config, but
    # there vision is genuinely unusable (mlx-vlm cannot load those conversions
    # either), so they stay text-only.
    if (
        "vision_config" not in config
        and "vision_config" not in (config.get("thinker_config") or {})
        and config.get("model_type") != "falcon_ocr"
    ):
        return None
    return MlxVlmBridge(model_path, config)


def images_from_message_content(content) -> List[str]:
    """Extract OpenAI image_url payloads (urls / data-URIs) from a content-part list."""
    if not isinstance(content, list):
        return []
    out = []
    for part in content:
        if part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url")
            if url:
                out.append(url)
    return out


def image_fingerprint(images: List[str]) -> str:
    """A stable identity for an ORDERED list of image payloads — hashing the raw
    payload strings directly (not the decoded pixels) is cheap and exactly as
    deterministic: same input string, same hash, every time. A NUL separator can't
    appear in base64/data-URI/URL text, so no ambiguity between e.g. ["ab","c"] and
    ["a","bc"]."""
    return hashlib.sha256("\x00".join(images).encode()).hexdigest()


def decode_image(payload: str):
    """An OpenAI image payload (data URI, raw base64, or http(s) URL) → PIL image."""
    from PIL import Image

    if payload.startswith("http://") or payload.startswith("https://"):
        import urllib.request

        with urllib.request.urlopen(payload, timeout=30) as r:
            data = r.read()
    else:
        b64 = payload.split(",", 1)[1] if payload.startswith("data:") else payload
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"invalid base64 image payload: {e}")
    return Image.open(io.BytesIO(data)).convert("RGB")


@dataclass
class VisionPrompt:
    """A processed multimodal prompt, ready for injection into generation."""

    tokens: List[int]  # image placeholders expanded to their per-patch runs
    embeddings: mx.array  # (L, hidden) merged text+image embeddings
    position_ids: Optional[mx.array] = None  # (3|4, 1, L) multimodal rope
    rope_deltas: Optional[mx.array] = None  # (1, 1)
    # gemma4 family / ernie mm-expert routing:
    mm_token_type_ids: Optional[mx.array] = None  # (1, L): 0 text / 1 image / 2 video / 3 audio
    per_layer_token_ids: Optional[List[int]] = None  # ids with image positions zeroed (E2B/E4B)
    # attn-mask-4d / falcon-visual: whole-prompt mask override, indexed by ABSOLUTE
    # prompt position — the text models slice it by cache offset themselves.
    attention_mask_4d: Optional[mx.array] = None  # (B, 1, L, L)
    # deepstack (qwen3_vl family / granite4_vision) + visual-lora (zaya1):
    visual_pos_masks: Optional[mx.array] = None  # (1, L) bool at image positions
    # one flat (n_visual, hidden) table per early layer (qwen3_vl family) or a
    # full-sequence (n_sets, L, hidden) array (granite4_vision).
    deepstack_visual_embeds: Optional[Union[List[mx.array], mx.array]] = None
    deepstack_target_layers: Optional[List[int]] = None  # granite4_vision only
    # cross-attention (mllama): fixed vision K/V + per-text-row visibility masks.
    cross_attention_states: Optional[mx.array] = None  # (B, V, hidden) — vision axis
    cross_attention_mask: Optional[mx.array] = None  # (B, 1, L, V)
    full_text_row_masked_out_mask: Optional[mx.array] = None  # (B, 1, L, 1)
    # falcon-visual: image-collapsed 1D positions + golden 2D coords, carried apart
    # from position_ids so the server's set_mrope_state branch never fires for them.
    visual_position_ids: Optional[mx.array] = None  # (1, L) int
    visual_rope_deltas: Optional[mx.array] = None  # (1, 1)
    pos_hw: Optional[mx.array] = None  # (1, L, 2) golden h/w coordinates
    single_prefill: bool = False  # image span must sit in ONE prefill forward
    # v1: this prompt's side state doesn't compose with the prompt cache yet (see
    # BYPASS_CACHE_SIDES) — the server runs it with a fresh cache, no reuse.
    bypass_cache: bool = False
    # A stable fingerprint of every image referenced so far in this conversation (not
    # just a new one this turn) — same images in the same order → same fingerprint, so
    # a follow-up turn's KV cache lookup lands in the SAME trie namespace and can reuse
    # the cached prefix through fetch_nearest_cache's normal token-prefix matching.
    # ANY difference in the image set (new/different/reordered images) changes the
    # fingerprint, landing in a fresh, isolated namespace — a stale cache from a
    # different image can never be served (see server.py's _serve_single).
    image_fingerprint: str = ""

    def sliced(self, keep_from: int) -> "VisionPrompt":
        """This prompt with its first `keep_from` positions dropped — for when
        fetch_nearest_cache finds a cached prefix and only a tail needs prefill.
        All per-position state (embeddings/positions/token types) must stay aligned
        with the trimmed token list, or the model reads misaligned vision state.
        Absolute-indexed state stays whole: attention_mask_4d and the falcon fields
        are sliced by cache offset inside the text models (and the overlay masks
        compose from the LAST rows/columns, so a full mask over a tail is exact),
        and cross_attention_states' axis is vision positions, not text."""
        if keep_from <= 0:
            return self
        deepstack = self.deepstack_visual_embeds
        if deepstack is not None:
            if isinstance(deepstack, list):
                # qwen3_vl family: flat (n_visual, hidden) tables — drop the rows
                # belonging to the sliced-away visual positions.
                n_before = int(self.visual_pos_masks[:, :keep_from].sum().item())
                deepstack = [table[n_before:] for table in deepstack]
            else:
                # granite4_vision: full-sequence (n_sets, L, hidden) feature adds.
                deepstack = deepstack[:, keep_from:]
        return replace(
            self,
            tokens=self.tokens[keep_from:],
            embeddings=self.embeddings[keep_from:],
            position_ids=self.position_ids[:, :, keep_from:] if self.position_ids is not None else None,
            mm_token_type_ids=self.mm_token_type_ids[:, keep_from:] if self.mm_token_type_ids is not None else None,
            per_layer_token_ids=self.per_layer_token_ids[keep_from:] if self.per_layer_token_ids else None,
            visual_pos_masks=self.visual_pos_masks[:, keep_from:] if self.visual_pos_masks is not None else None,
            deepstack_visual_embeds=deepstack,
            cross_attention_mask=self.cross_attention_mask[:, :, keep_from:] if self.cross_attention_mask is not None else None,
            full_text_row_masked_out_mask=self.full_text_row_masked_out_mask[:, :, keep_from:] if self.full_text_row_masked_out_mask is not None else None,
        )


class MlxVlmBridge:
    """Generic vision bridge: lazily loads the checkpoint through mlx-vlm and calls its
    per-arch get_input_embeddings. Heavy pieces load on the first image request."""

    def __init__(self, model_path: Path, config: dict):
        self.model_path = model_path
        self.config = config
        self.model_type = config.get("model_type", "")
        # qwen3_omni_moe nests the multimodal fields under thinker_config.
        thinker = config.get("thinker_config") or {}
        self.vision_config = (
            config.get("vision_config") or thinker.get("vision_config") or {}
        )
        self.image_token_id = (
            config.get("image_token_id")
            or config.get("image_token_index")
            or thinker.get("image_token_id")
            or self.vision_config.get("image_token_id")
            or config.get("im_patch_id")  # ernie4_5_moe_vl
            or config.get("img_id")  # falcon_ocr (flat TII config)
            or config.get("img_context_token_id")  # nemotron_h_nano_omni
        )
        self._vlm = None
        self._processor = None

    def _ensure_loaded(self):
        if self._vlm is not None:
            return
        from mlx_vlm.utils import load_model, load_processor

        # Refuse conversions whose vision weights were stripped (an mlx-lm-made
        # conversion keeps vision_config in config.json but drops every non-language
        # tensor). Running those would silently use RANDOMLY-initialized vision
        # projections — plausible-looking garbage, the worst failure mode.
        import glob as _glob

        if self.model_type == "falcon_ocr":
            # falcon is early-fusion: EVERY tensor lives under language_model.*;
            # the one vision-input weight is the image projector, whose absence
            # marks an mlx-lm-made (vision-stripped) conversion.
            def is_vision_key(k):
                return "img_projector" in k

        else:

            def is_vision_key(k):
                return not k.startswith("language_model.")

        has_vision_weights = False
        index = self.model_path / "model.safetensors.index.json"
        if index.exists():
            with open(index) as f:
                keys = json.load(f).get("weight_map", {}).keys()
            has_vision_weights = any(is_vision_key(k) for k in keys)
        else:
            for shard in _glob.glob(str(self.model_path / "model*.safetensors")):
                if any(is_vision_key(k) for k in mx.load(shard)):
                    has_vision_weights = True
                    break
        if not has_vision_weights:
            raise ValueError(
                "this checkpoint's config declares vision_config but its weights are "
                "text-only (an mlx-lm conversion strips vision tensors) — use a "
                "conversion made with mlx-vlm"
            )

        # lazy=True: arrays materialize only when evaluated — the duplicate language
        # model inside the mlx-vlm object never runs, so it never occupies memory.
        # strict=False: some conversions carry vestigial text-side weights that
        # mlx-vlm's language classes don't instantiate (e.g. per-layer K/V for
        # gemma-4's KV-shared layers) — harmless here, we only use the vision side.
        self._vlm = load_model(self.model_path, lazy=True, strict=False)
        self._processor = load_processor(self.model_path)

    def prepare(self, rendered_prompt: str, images: List[str]) -> VisionPrompt:
        """(chat-template-rendered prompt text, image payloads) → VisionPrompt."""
        from mlx_vlm.utils import prepare_inputs

        side = TEXT_SIDE.get(self.model_type, "plain")

        self._ensure_loaded()
        pils = [decode_image(i) for i in images]
        inputs = prepare_inputs(
            self._processor,
            images=pils,
            prompts=rendered_prompt,
            image_token_index=self.image_token_id,
        )
        input_ids = inputs["input_ids"]  # (1, L), placeholders expanded
        extra = {
            k: v
            for k, v in inputs.items()
            if k not in ("input_ids", "pixel_values", "attention_mask")
        }
        feats = self._vlm.get_input_embeddings(
            input_ids,
            inputs.get("pixel_values"),
            mask=inputs.get("attention_mask"),
            **extra,
        )
        # Normalize: some archs return a bare array instead of the dataclass.
        if isinstance(feats, mx.array):
            embeds = feats
            feats = None
            position_ids = rope_deltas = None
        else:
            embeds = feats.inputs_embeds
            position_ids = feats.position_ids
            rope_deltas = feats.rope_deltas
            # Fail loud on side state this arch's TEXT_SIDE entry doesn't consume.
            for field, (why, sides) in GUARDED_FEATURES.items():
                if getattr(feats, field, None) is not None and side not in sides:
                    raise ValueError(
                        f'architecture "{self.model_type}" produces {why} — not yet '
                        "supported by mlx-unified's injection path"
                    )

        mm_token_type_ids = None
        per_layer_token_ids = None
        attention_mask_4d = None
        visual_pos_masks = None
        deepstack_visual_embeds = None
        deepstack_target_layers = None
        cross_attention_states = None
        cross_attention_mask = None
        full_text_row_masked_out_mask = None
        visual_position_ids = None
        visual_rope_deltas = None
        pos_hw = None
        single_prefill = False

        if self.model_type in QWEN_MROPE_TYPES:
            # Multi-image-correct positions (mlx-vlm's own can drift on later runs).
            state = build_qwen_image_mrope_state(
                input_ids=input_ids,
                image_grid_thw=inputs["image_grid_thw"],
                image_token_id=self.image_token_id,
                spatial_merge_size=self.vision_config["spatial_merge_size"],
            )
            position_ids, rope_deltas = state.position_ids, state.rope_deltas
        elif self.model_type == "hunyuan_vl":
            # hunyuan's 4-axis xdrope positions come from the PROCESSOR output —
            # (1, 4, L), axis order [p, w, h, t] — not from get_input_embeddings
            # (which stashes them as mlx-vlm-internal side state). Positions are
            # never compressed after image spans, so rope_deltas is always zero.
            position_ids = mx.array(inputs["position_ids"]).transpose(1, 0, 2)
            rope_deltas = mx.zeros((1, 1), dtype=position_ids.dtype)

        if side == "gemma-visual":
            # 1) mlx-vlm returns embeddings ALREADY ×sqrt(hidden); mlx-lm's gemma4_text
            #    scales injected embeddings again — undo one scaling here.
            text_hidden = (self.config.get("text_config") or {}).get(
                "hidden_size"
            ) or self.config.get("hidden_size")
            embeds = embeds / (text_hidden**0.5)
            # 2) token types drive the bidirectional image-span mask (gemma4_unified);
            #    derive from ids if the processor didn't emit them.
            mm_token_type_ids = inputs.get("mm_token_type_ids")
            if mm_token_type_ids is None:
                mm_token_type_ids = (input_ids == self.image_token_id).astype(mx.int32)
            # 3) E2B/E4B per-layer inputs must come from the REAL ids with image
            #    positions zeroed (computed against the serving model's own weights).
            if ((self.config.get("text_config") or {}).get("hidden_size_per_layer_input") or 0) > 0:
                zeroed = mx.where(
                    input_ids == self.image_token_id, mx.zeros_like(input_ids), input_ids
                )
                per_layer_token_ids = zeroed.squeeze(0).tolist()
            # 4) an image span must never be split across prefill chunks — the
            #    bidirectional edges within a block are unrecoverable once the first
            #    chunk's KV is written causally.
            single_prefill = True

        elif side == "gemma3n-visual":
            # mlx-vlm's gemma3n returns UNSCALED text-position embeddings while
            # multimodal positions are already in final hidden space, and the text
            # model consumes injected embeddings as-is — so scale ONLY text
            # positions by sqrt(hidden). Its per_layer_inputs are discarded
            # (GUARDED_FEATURES): gemma3n re-derives identical ones from the raw
            # token ids that generation passes alongside input_embeddings.
            text_cfg = self.config.get("text_config") or self.config
            vocab_offset = self.vision_config.get("vocab_offset") or text_cfg.get(
                "vocab_size_per_layer_input"
            )
            embeds = mx.where(
                (input_ids < vocab_offset)[..., None],
                embeds * text_cfg["hidden_size"] ** 0.5,
                embeds,
            )

        elif side == "ernie-visual":
            # qwen-style mrope (positions built above) PLUS token-type routing:
            # image tokens go through the second MoE expert group.
            mm_token_type_ids = (input_ids == self.image_token_id).astype(mx.int32)

        elif side == "attn-mask-4d":
            # Whole-prompt (B, 1, L, L) mask override: bidirectional prefix/image
            # spans. The edges within a span are unrecoverable if prefill splits
            # it, so one prefill call.
            attention_mask_4d = feats.attention_mask_4d
            single_prefill = True

        elif side == "mrope+deepstack":
            # qwen3_vl family: mrope (built above) + mid-layer visual injection.
            # deepstack_visual_embeds may be absent/None (qwen3_omni_moe's mlx-vlm
            # port doesn't return the tower's multiscale features yet) — injection
            # then simply doesn't fire; mrope + the embedding splice still apply.
            visual_pos_masks = getattr(feats, "visual_pos_masks", None)
            deepstack_visual_embeds = getattr(feats, "deepstack_visual_embeds", None)

        elif side == "granite-deepstack":
            # granite4_vision: inputs_embeds arrive UNSCALED with image positions
            # ZEROED — image content reaches the model only through the deepstack
            # adds, which need explicit target layers (get_input_embeddings stashes
            # them on mlx-vlm's language model; the config carries the same map,
            # deepstack sets first, then spatial).
            visual_pos_masks = feats.visual_pos_masks
            deepstack_visual_embeds = feats.deepstack_visual_embeds
            inner = getattr(getattr(self._vlm, "language_model", None), "model", None)
            deepstack_target_layers = getattr(inner, "_deepstack_target_layers", None)
            if deepstack_target_layers is None:
                deepstack_target_layers = [
                    llm for _, llm in self.config["deepstack_layer_map"]
                ]
                if self.config.get("use_spatial_sampling") and self.config.get(
                    "spatial_target_layers"
                ):
                    deepstack_target_layers += list(self.config["spatial_target_layers"])

        elif side == "cross-attention":
            # mllama: fixed vision K/V states + per-text-row visibility masks,
            # consumed by the interleaved cross-attention layers on every forward
            # (decode included — the server resets only after generation ends).
            cross_attention_states = feats.cross_attention_states
            cross_attention_mask = feats.cross_attention_mask
            full_text_row_masked_out_mask = feats.full_text_row_masked_out_mask

        elif side == "visual-lora":
            # zaya1: the mask gates vision-LoRA weight deltas during prefill.
            visual_pos_masks = feats.visual_pos_masks

        elif side == "falcon-visual":
            # falcon_ocr: image-collapsed 1D positions + golden 2D coords + the
            # block-diagonal bidirectional mask — set_visual_state territory, NOT
            # mrope, so reroute feats.position_ids away from the mrope fields.
            visual_position_ids, visual_rope_deltas = position_ids, rope_deltas
            position_ids = rope_deltas = None
            pos_hw = feats.pos_hw
            attention_mask_4d = feats.attention_mask_4d
            single_prefill = True

        if position_ids is not None and side not in MROPE_SIDES:
            raise ValueError(
                f'architecture "{self.model_type}" requires multimodal position support '
                "that its mlx-lm text model does not implement yet"
            )

        return VisionPrompt(
            tokens=input_ids.squeeze(0).tolist(),
            embeddings=embeds.squeeze(0) if embeds.ndim == 3 else embeds,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            mm_token_type_ids=mm_token_type_ids,
            per_layer_token_ids=per_layer_token_ids,
            attention_mask_4d=attention_mask_4d,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            deepstack_target_layers=deepstack_target_layers,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            visual_position_ids=visual_position_ids,
            visual_rope_deltas=visual_rope_deltas,
            pos_hw=pos_hw,
            single_prefill=single_prefill,
            bypass_cache=side in BYPASS_CACHE_SIDES,
            image_fingerprint=image_fingerprint(images),
        )


# --- Qwen multimodal-RoPE position builder --------------------------------
# Adapted from lmstudio-ai/mlx-engine model_kit/batched_vision/qwen_mrope.py
# (MIT © LM Studio): walks the EXPANDED token sequence, giving text spans
# sequential positions on all three axes and each image run its (t, h, w)
# grid positions — multi-image correct.


@dataclass(frozen=True)
class QwenMropeState:
    position_ids: mx.array  # (3, 1, L)
    rope_deltas: mx.array  # (1, 1)


def build_qwen_image_mrope_state(
    *,
    input_ids: mx.array,
    image_grid_thw: mx.array,
    image_token_id: int,
    spatial_merge_size: int,
) -> QwenMropeState:
    token_list = input_ids.squeeze(0).tolist()
    image_runs = _find_token_runs(token_list, image_token_id)
    grid_list = image_grid_thw.tolist()
    if len(grid_list) == 3 and isinstance(grid_list[0], int):
        grid_list = [grid_list]
    if len(image_runs) != len(grid_list):
        raise ValueError(
            "Qwen image token runs do not match image_grid_thw entries: "
            f"{len(image_runs)} runs vs {len(grid_list)} grids."
        )

    positions = [[], [], []]
    token_cursor = 0
    position_cursor = 0
    for (run_start, run_end), (t, h, w) in zip(image_runs, grid_list):
        text_len = run_start - token_cursor
        for dim in range(3):
            positions[dim].extend(range(position_cursor, position_cursor + text_len))
        position_cursor += text_len

        llm_grid_t = int(t)
        llm_grid_h = int(h) // spatial_merge_size
        llm_grid_w = int(w) // spatial_merge_size
        run_length = run_end - run_start
        if run_length != llm_grid_t * llm_grid_h * llm_grid_w:
            raise ValueError(
                "Qwen image token run length does not match image_grid_thw: "
                f"run length {run_length}, expected {llm_grid_t * llm_grid_h * llm_grid_w}."
            )

        image_position_offset = position_cursor
        for t_idx in range(llm_grid_t):
            for h_idx in range(llm_grid_h):
                for w_idx in range(llm_grid_w):
                    positions[0].append(image_position_offset + t_idx)
                    positions[1].append(image_position_offset + h_idx)
                    positions[2].append(image_position_offset + w_idx)

        position_cursor = image_position_offset + max(llm_grid_t, llm_grid_h, llm_grid_w)
        token_cursor = run_end

    trailing_text_len = len(token_list) - token_cursor
    for dim in range(3):
        positions[dim].extend(range(position_cursor, position_cursor + trailing_text_len))
    position_cursor += trailing_text_len

    return QwenMropeState(
        position_ids=mx.array(positions, dtype=input_ids.dtype).reshape(3, 1, len(token_list)),
        rope_deltas=mx.array([[position_cursor - len(token_list)]], dtype=input_ids.dtype),
    )


def _find_token_runs(tokens: List[int], target_token: int) -> List[tuple]:
    """Return `[start, end)` ranges where `target_token` is contiguous."""
    runs = []
    start = None
    for idx, token in enumerate(tokens):
        if token == target_token:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(tokens)))
    return runs
