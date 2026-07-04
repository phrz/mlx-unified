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
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx

# What the TEXT model must support beyond plain embedding injection, per model_type.
#   "plain" — no text-side changes needed; injection just works.
#   "mrope" — 3D multimodal RoPE (implemented for qwen3_5/qwen3_5_moe in this fork:
#             set_mrope_state/reset_mrope_state on the text model).
#   "gemma-visual" — gemma4 family (implemented in this fork's gemma4_text):
#             mm_token_type_ids-driven bidirectional attention within image spans
#             (gemma4_unified), explicit per-layer inputs (E2B/E4B), an embed-scale
#             correction, and single-call prefill so an image span is never split.
TEXT_SIDE = {
    "qwen3_5": "mrope",
    "qwen3_5_moe": "mrope",
    "gemma4": "gemma-visual",
    "gemma4_unified": "gemma-visual",
}

# model_types whose qwen-style mrope positions we build ourselves (multi-image-correct)
# instead of trusting mlx-vlm's get_rope_index output.
QWEN_MROPE_TYPES = ("qwen3_5", "qwen3_5_moe")


def load_vision_encoder(model_path):
    """A vision bridge for the checkpoint at model_path, or None for text-only ones."""
    model_path = Path(model_path)
    try:
        with open(model_path / "config.json") as f:
            config = json.load(f)
    except OSError:
        return None
    if "vision_config" not in config:
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
    position_ids: Optional[mx.array] = None  # (3, 1, L) multimodal rope (qwen family)
    rope_deltas: Optional[mx.array] = None  # (1, 1)
    # gemma4 family:
    mm_token_type_ids: Optional[mx.array] = None  # (1, L): 0 text / 1 image / 2 video / 3 audio
    per_layer_token_ids: Optional[List[int]] = None  # ids with image positions zeroed (E2B/E4B)
    single_prefill: bool = False  # image span must sit in ONE prefill forward


class MlxVlmBridge:
    """Generic vision bridge: lazily loads the checkpoint through mlx-vlm and calls its
    per-arch get_input_embeddings. Heavy pieces load on the first image request."""

    def __init__(self, model_path: Path, config: dict):
        self.model_path = model_path
        self.config = config
        self.model_type = config.get("model_type", "")
        self.image_token_id = (
            config.get("image_token_id")
            or config.get("image_token_index")
            or (config.get("vision_config") or {}).get("image_token_id")
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

        has_vision_weights = False
        index = self.model_path / "model.safetensors.index.json"
        if index.exists():
            with open(index) as f:
                keys = json.load(f).get("weight_map", {}).keys()
            has_vision_weights = any(not k.startswith("language_model.") for k in keys)
        else:
            for shard in _glob.glob(str(self.model_path / "model*.safetensors")):
                if any(not k.startswith("language_model.") for k in mx.load(shard)):
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
            position_ids = rope_deltas = None
        else:
            embeds = feats.inputs_embeds
            position_ids = feats.position_ids
            rope_deltas = feats.rope_deltas
            for field, why in (
                ("attention_mask_4d", "bidirectional attention masks"),
                ("deepstack_visual_embeds", "mid-layer visual injection"),
                ("cross_attention_states", "cross-attention"),
            ):
                if getattr(feats, field, None) is not None:
                    raise ValueError(
                        f'architecture "{self.model_type}" produces {why} — not yet '
                        "supported by mlx-unified's injection path"
                    )
            if getattr(feats, "per_layer_inputs", None) is not None and side != "gemma-visual":
                raise ValueError(
                    f'architecture "{self.model_type}" produces per-layer multimodal '
                    "inputs — not yet supported by mlx-unified's injection path"
                )

        mm_token_type_ids = None
        per_layer_token_ids = None
        single_prefill = False

        if self.model_type in QWEN_MROPE_TYPES:
            # Multi-image-correct positions (mlx-vlm's own can drift on later runs).
            state = build_qwen_image_mrope_state(
                input_ids=input_ids,
                image_grid_thw=inputs["image_grid_thw"],
                image_token_id=self.image_token_id,
                spatial_merge_size=self.config["vision_config"]["spatial_merge_size"],
            )
            position_ids, rope_deltas = state.position_ids, state.rope_deltas

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

        if position_ids is not None and side != "mrope":
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
            single_prefill=single_prefill,
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
