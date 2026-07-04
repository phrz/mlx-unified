# Copyright © 2026 Apple Inc.
#
# mlx-unified: vision support for mlx_lm.server.
#
# The approach follows lmstudio-ai/mlx-engine (MIT): the TEXT model is always
# mlx-lm's own (this package), and mlx-vlm contributes only its vision
# components — the vision tower encodes images into embeddings that are spliced
# into the text-token embeddings at image-placeholder positions, then injected
# into generation via mlx-lm's `input_embeddings` parameter. One text
# implementation, no duplicated code paths.
#
# build_qwen_image_mrope_state is adapted from mlx-engine's
# model_kit/batched_vision/qwen_mrope.py (MIT © LM Studio).
#
# mlx-vlm is imported lazily — text-only use of mlx_lm never requires it.

import base64
import binascii
import glob
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx

# model_type → encoder class; growing this list is how new architectures gain
# vision support (each needs its arch's merge/position conventions verified).
SUPPORTED_VISION_MODEL_TYPES = ("qwen3_5", "qwen3_5_moe")


def load_vision_encoder(model_path):
    """A VisionEncoder for the checkpoint at model_path, or None.

    None when the checkpoint has no vision_config (text-only), or its
    model_type's vision conventions aren't implemented yet — the server then
    behaves exactly like stock mlx-lm (images rejected).
    """
    model_path = Path(model_path)
    try:
        with open(model_path / "config.json") as f:
            config = json.load(f)
    except OSError:
        return None
    if "vision_config" not in config:
        return None
    if config.get("model_type") not in SUPPORTED_VISION_MODEL_TYPES:
        return None
    return QwenVisionEncoder(model_path, config)


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
    position_ids: mx.array  # (3, 1, L) multimodal rope positions
    rope_deltas: mx.array  # (1, 1)


class QwenVisionEncoder:
    """Vision components for the qwen3_5 family, loaded from the same checkpoint
    the text model came from. Heavy pieces (processor, vision tower) load lazily
    on the first image request."""

    def __init__(self, model_path: Path, config: dict):
        self.model_path = model_path
        self.config = config
        self.image_token_id = config["image_token_id"]
        self.video_token_id = config.get("video_token_id")
        self.spatial_merge_size = config["vision_config"]["spatial_merge_size"]
        self._processor = None
        self._tower = None

    def _ensure_loaded(self):
        if self._tower is not None:
            return
        from mlx_vlm.models.qwen3_5 import VisionModel
        from mlx_vlm.models.qwen3_5.config import VisionConfig
        from mlx_vlm.utils import load_processor

        self._processor = load_processor(self.model_path)
        tower = VisionModel(VisionConfig.from_dict(self.config["vision_config"]))
        weights = {}
        for shard in glob.glob(str(self.model_path / "model*.safetensors")):
            for k, v in mx.load(shard).items():
                if k.startswith("vision_tower."):
                    weights[k.removeprefix("vision_tower.")] = v
        if not weights:
            raise ValueError(f"no vision_tower.* weights in {self.model_path}")
        tower.load_weights(list(weights.items()), strict=True)
        tower.eval()
        self._tower = tower

    def prepare(self, rendered_prompt: str, images: List[str], embed_tokens) -> VisionPrompt:
        """(chat-template-rendered prompt text, image payloads) → VisionPrompt.

        `embed_tokens` is the TEXT model's embedding module — the text side of
        the merged embedding always comes from mlx-lm's own weights.
        """
        from mlx_vlm.utils import prepare_inputs

        self._ensure_loaded()
        pils = [decode_image(i) for i in images]
        inputs = prepare_inputs(
            self._processor,
            images=pils,
            prompts=rendered_prompt,
            image_token_index=self.image_token_id,
        )
        input_ids = inputs["input_ids"]  # (1, L), placeholders expanded per patch
        pixel_values = inputs["pixel_values"]
        grid_thw = inputs["image_grid_thw"]

        text_embeds = embed_tokens(input_ids)
        image_embeds, _ = self._tower(
            pixel_values.astype(self._tower.patch_embed.proj.weight.dtype), grid_thw
        )
        merged = self._merge(text_embeds, input_ids, image_embeds.astype(text_embeds.dtype))

        state = build_qwen_image_mrope_state(
            input_ids=input_ids,
            image_grid_thw=grid_thw,
            image_token_id=self.image_token_id,
            spatial_merge_size=self.spatial_merge_size,
        )
        return VisionPrompt(
            tokens=input_ids.squeeze(0).tolist(),
            embeddings=merged.squeeze(0),
            position_ids=state.position_ids,
            rope_deltas=state.rope_deltas,
        )

    def _merge(self, text_embeds: mx.array, input_ids: mx.array, image_embeds: mx.array):
        from mlx_vlm.models.qwen3_vl.qwen3_vl import masked_scatter

        mask = input_ids == self.image_token_id
        if self.video_token_id is not None:
            mask = mask | (input_ids == self.video_token_id)
        mask = mx.broadcast_to(mask[..., None], text_embeds.shape)
        n_slots = int(mask.sum().item()) // text_embeds.shape[-1]
        if n_slots != image_embeds.shape[0]:
            raise ValueError(
                f"image feature count ({image_embeds.shape[0]}) does not match "
                f"placeholder token count ({n_slots})"
            )
        return masked_scatter(text_embeds, mask, image_embeds)


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
