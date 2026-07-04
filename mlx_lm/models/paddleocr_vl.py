# Copyright © 2026 Apple Inc.
#
# PaddleOCR-VL text model: an ERNIE-4.5-style GQA/RMSNorm/SwiGLU decoder with
# Qwen2-VL-style sectioned-half-split 3D multimodal RoPE (mrope_section
# [16, 24, 24]). Language architecture ported from Blaizzy/mlx-vlm's
# models/paddleocr_vl (MIT © Blaizzy/mlx-vlm contributors); the
# set_mrope_state/reset_mrope_state side-state pattern follows this fork's
# qwen3_5.py.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    # The checkpoint's config is flat (text fields at the root, next to
    # vision_config) — defaults mirror the real PaddleOCR-VL-0.9B release.
    model_type: str = "paddleocr_vl"
    hidden_size: int = 1024
    num_hidden_layers: int = 18
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-5
    vocab_size: int = 103424
    num_key_value_heads: Optional[int] = 2
    head_dim: Optional[int] = None
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict[str, Union[float, str, List[int]]]] = None
    use_bias: bool = False
    tie_word_embeddings: bool = False
    mrope_section: List[int] = field(default_factory=lambda: [16, 24, 24])

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_scaling and "mrope_section" in self.rope_scaling:
            self.mrope_section = list(self.rope_scaling["mrope_section"])
        if 2 * sum(self.mrope_section) != self.head_dim:
            raise ValueError(
                f"mrope_section {self.mrope_section} must sum to head_dim/2 "
                f"({self.head_dim // 2})"
            )


def _sectioned_rope_cos_sin(
    position_ids: mx.array,
    inv_freq: mx.array,
    mrope_section: List[int],
    dtype: mx.Dtype,
) -> Tuple[mx.array, mx.array]:
    """cos/sin (B, 1, L, head_dim) for sectioned-half-split mrope: frequency
    band i (widths from mrope_section, mirrored across the two rope halves)
    reads its positions from axis i % 3 of position_ids (3, B, L)."""
    freqs = position_ids[..., None].astype(mx.float32) * inv_freq  # (3, B, L, d/2)
    bands = []
    start = 0
    for i, width in enumerate(mrope_section):
        bands.append(freqs[i % 3, ..., start : start + width])
        start += width
    half = mx.concatenate(bands, axis=-1)
    emb = mx.concatenate([half, half], axis=-1)  # (B, L, d)
    return (
        mx.expand_dims(mx.cos(emb), 1).astype(dtype),
        mx.expand_dims(mx.sin(emb), 1).astype(dtype),
    )


def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.use_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.use_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.use_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.use_bias)

        self.rope = initialize_rope(
            head_dim,
            args.rope_theta,
            False,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if position_embeddings is not None:
            # Vision: 3D multimodal RoPE from absolute positions (no offset).
            cos, sin = position_embeddings
            queries = _apply_rotary(queries, cos, sin)
            keys = _apply_rotary(keys, cos, sin)
        elif cache is not None:
            # Text-only: plain 1D RoPE (sectioned mrope with equal t/h/w axes
            # degenerates to exactly this).
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache, position_embeddings)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class PaddleOCRVLModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Underscore attrs so MLX's module walker never registers them (no
        # weights to load/quantize). The inverse rope frequencies feed the
        # sectioned mrope path; the mm_* pair is the multimodal side state, set
        # by the vision path before generation and cleared afterwards.
        self._inv_freq = args.rope_theta ** (
            -mx.arange(0, args.head_dim, 2, dtype=mx.float32) / args.head_dim
        )
        self._mm_position_ids = None
        self._mm_rope_deltas = None

    def set_mrope_state(self, position_ids: mx.array, rope_deltas: mx.array) -> None:
        """Install 3D multimodal positions (3, B, L) + rope delta for a vision prompt."""
        self._mm_position_ids = position_ids
        self._mm_rope_deltas = rope_deltas

    def reset_mrope_state(self) -> None:
        self._mm_position_ids = None
        self._mm_rope_deltas = None

    def _compute_position_ids(self, inputs: mx.array, cache) -> Optional[mx.array]:
        """Positions for the current forward pass from stored multimodal state.

        None (the common text-only case) routes every layer to the plain
        1D-RoPE path. During chunked prefill the stored prompt positions are
        sliced by cache offset (stitching a sequential tail if a chunk crosses
        the end); once the prompt is exhausted, decode positions are
        cache_offset + rope_delta broadcast to all three mrope axes.
        """
        if self._mm_position_ids is None and self._mm_rope_deltas is None:
            return None
        if self._mm_position_ids is not None and self._mm_rope_deltas is None:
            raise ValueError(
                "MRoPE state is inconsistent: position_ids set without rope_deltas."
            )

        cache_offset = 0
        cache_offset_scalar = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0].offset
            if isinstance(offset, int):
                cache_offset = offset
                cache_offset_scalar = offset
            elif isinstance(offset, mx.array) and offset.ndim == 0:
                cache_offset = offset.item()
                cache_offset_scalar = cache_offset
            elif isinstance(offset, mx.array):
                cache_offset = offset
                cache_offset_scalar = offset[0].item()

        batch_size, seq_length = inputs.shape

        if self._mm_position_ids is not None:
            stored_seq_length = self._mm_position_ids.shape[2]
            if cache_offset_scalar < stored_seq_length:
                stored_end = min(cache_offset_scalar + seq_length, stored_seq_length)
                stored_positions = self._mm_position_ids[
                    :, :, cache_offset_scalar:stored_end
                ]
                if stored_end - cache_offset_scalar == seq_length:
                    return stored_positions
                tail_positions = self._sequential_position_ids(
                    batch_size=batch_size,
                    seq_length=seq_length - (stored_end - cache_offset_scalar),
                    start_offset=stored_seq_length,
                )
                return mx.concatenate([stored_positions, tail_positions], axis=2)

        return self._sequential_position_ids(
            batch_size=batch_size, seq_length=seq_length, start_offset=cache_offset
        )

    def _sequential_position_ids(
        self, *, batch_size: int, seq_length: int, start_offset
    ) -> mx.array:
        delta = mx.array(start_offset)
        if self._mm_rope_deltas is not None:
            delta = delta + self._mm_rope_deltas

        position_ids = mx.arange(seq_length).reshape(1, -1)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))

        if delta.ndim == 0:
            delta = mx.broadcast_to(delta.reshape(1, 1), (batch_size, 1))
        elif delta.ndim == 1:
            delta = delta[:batch_size].reshape(-1, 1)
            if delta.shape[0] == 1 and batch_size > 1:
                delta = mx.broadcast_to(delta, (batch_size, 1))
        else:
            delta = delta[:batch_size]
            if delta.shape[0] == 1 and batch_size > 1:
                delta = mx.broadcast_to(delta, (batch_size, delta.shape[1]))

        position_ids = mx.add(position_ids, delta)[None, ...]
        return mx.broadcast_to(position_ids, (3, batch_size, seq_length))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        # None for text-only requests — layers take the plain 1D-RoPE path.
        position_ids = self._compute_position_ids(inputs, cache)
        position_embeddings = None
        if position_ids is not None:
            position_embeddings = _sectioned_rope_cos_sin(
                position_ids, self._inv_freq, self.args.mrope_section, h.dtype
            )

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, position_embeddings)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = PaddleOCRVLModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # Accept both the raw PaddleOCR-VL checkpoint (text keys under model.*
        # / lm_head.*, vision tower under visual.* / mlp_AR.*) and an
        # mlx-vlm-made conversion (language_model.model.* /
        # language_model.lm_head.* / visual.*) — strip the vision tower, keep
        # the language model.
        sanitized = {}
        for k, v in weights.items():
            if (
                k.startswith(("visual.", "mlp_AR."))
                or "packing_position_embedding" in k
                or "rotary_emb.inv_freq" in k
            ):
                continue
            sanitized[k.removeprefix("language_model.")] = v
        if self.args.tie_word_embeddings:
            sanitized.pop("lm_head.weight", None)
        return sanitized

    @property
    def layers(self):
        return self.model.layers
