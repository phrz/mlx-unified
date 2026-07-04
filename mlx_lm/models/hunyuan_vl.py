# Copyright © 2026 Apple Inc.
#
# Tencent Hunyuan-VL text model with 4-axis "xdrope" multimodal RoPE side state
# (mlx-unified). The rope math is ported from Blaizzy/mlx-vlm's
# models/hunyuan_vl/language.py (MIT © Blaizzy/mlx-vlm contributors): the
# NTK-alpha-scaled frequency half is split into xdrope_section chunks (default
# [16, 16, 16, 16]) whose frequencies read the (p, w, h, t) position axes — the
# processor's axis order, which is what mlx-vlm actually feeds at runtime (its
# language.py fallback stacks [p, t, h, w]; the processor wins in prepare_inputs).
# The p axis stays the raw sequential index everywhere, so equal axes degenerate
# to exactly the DynamicNTKAlphaRoPE 1D rope that hunyuan_v1_dense uses, and
# decode positions are plain cache offsets (rope delta 0 — hunyuan never
# compresses positions after an image span).

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .hunyuan_v1_dense import DynamicNTKAlphaRoPE


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = "hunyuan_vl"
    vocab_size: int = 120818
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    intermediate_size: int = 3584
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    use_qk_norm: bool = True
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, int, bool, str, List[int]]]] = field(
        default_factory=lambda: {
            "type": "xdrope",
            "alpha": 1000.0,
            "xdrope_section": [16, 16, 16, 16],
        }
    )
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


def _scaled_rope_base(args: TextModelArgs) -> float:
    """rope_theta with hunyuan's NTK-alpha scaling (applies to xdrope too)."""
    scaling = args.rope_scaling or {}
    alpha = scaling.get("alpha")
    if scaling.get("type") in ("xdrope", "dynamic") and alpha:
        return args.rope_theta * alpha ** (args.head_dim / (args.head_dim - 2))
    return args.rope_theta


def _rotate_half(x):
    mid = x.shape[-1] // 2
    return mx.concatenate([-x[..., mid:], x[..., :mid]], axis=-1)


def _xdrope_cos_sin(position_ids, inv_freq, axis_selector):
    """(num_axes, B, L) positions -> half-split cos/sin of shape (B, 1, L, head_dim).

    Each frequency index reads the position axis its xdrope_section chunk names;
    duplicating the half then matches _rotate_half's (d, d + dim/2) pairing —
    numerically identical to mlx-vlm's split-and-reassemble of precomputed cos/sin.
    """
    positions = mx.take(position_ids, axis_selector, axis=0).transpose(1, 2, 0)
    angles = positions.astype(mx.float32) * inv_freq
    angles = mx.concatenate([angles, angles], axis=-1)
    return mx.cos(angles)[:, None], mx.sin(angles)[:, None]


def _apply_xdrope(queries, keys, cos, sin):
    q = queries.astype(mx.float32)
    k = keys.astype(mx.float32)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q.astype(queries.dtype), k.astype(keys.dtype)


class Attention(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.head_dim
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(head_dim, args.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(head_dim, args.rms_norm_eps)

        # Text-only path: NTK-alpha rope, identical to hunyuan_v1_dense (equal
        # p/w/h/t axes make xdrope degenerate to exactly this).
        self.rope = DynamicNTKAlphaRoPE(head_dim, base=_scaled_rope_base(args))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if position_embeddings is not None:
            cos, sin = position_embeddings
            queries, keys = _apply_xdrope(queries, keys, cos, sin)
        elif cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # QK norm comes AFTER rope in hunyuan (both dense and VL).
        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        dim, hidden_dim = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache, position_embeddings)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class HunyuanVLModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # XD-RoPE side state (mlx-unified), set by the vision path before generation
        # and cleared afterwards. Underscore attrs so MLX's module walker never
        # registers them as parameters.
        head_dim = args.head_dim
        half_dim = head_dim // 2
        base = _scaled_rope_base(args)
        self._inv_freq = base ** (
            -mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim
        )
        sections = (args.rope_scaling or {}).get("xdrope_section") or [half_dim]
        if sum(sections) != half_dim:
            raise ValueError(
                f"xdrope_section {sections} must sum to head_dim/2 ({half_dim})"
            )
        # Frequency-index -> position-axis map: chunks of the half in xdrope_section
        # order, axes as the processor stacks them: (p, w, h, t).
        selector = []
        for axis, length in enumerate(sections):
            selector.extend([axis] * length)
        self._axis_selector = mx.array(selector, dtype=mx.int32)
        self._num_axes = len(sections)
        self._mm_position_ids = None
        self._mm_rope_deltas = None

    def set_mrope_state(self, position_ids: mx.array, rope_deltas: mx.array) -> None:
        """Install 4-axis xdrope positions (4, B, L), processor order (p, w, h, t),
        plus the rope delta (always 0 for hunyuan) for a vision prompt."""
        self._mm_position_ids = position_ids
        self._mm_rope_deltas = rope_deltas

    def reset_mrope_state(self) -> None:
        self._mm_position_ids = None
        self._mm_rope_deltas = None

    def _compute_position_ids(self, inputs: mx.array, cache) -> Optional[mx.array]:
        """Positions for the current forward pass from stored multimodal state.

        None (the common text-only case) routes every layer to the original
        1D-RoPE path. During chunked prefill the stored prompt positions are
        sliced by cache offset (stitching a sequential tail if a chunk crosses
        the end); once the prompt is exhausted, decode positions are
        cache_offset + rope_delta broadcast to all xdrope axes.
        """
        if self._mm_position_ids is None and self._mm_rope_deltas is None:
            return None
        if self._mm_position_ids is not None and self._mm_rope_deltas is None:
            raise ValueError(
                "XD-RoPE state is inconsistent: position_ids set without rope_deltas."
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
        return mx.broadcast_to(
            position_ids, (self._num_axes, batch_size, seq_length)
        )

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

        # None for text-only requests — layers take the original 1D-RoPE path.
        position_ids = self._compute_position_ids(inputs, cache)
        position_embeddings = None
        if position_ids is not None:
            position_embeddings = _xdrope_cos_sin(
                position_ids, self._inv_freq, self._axis_selector
            )

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, position_embeddings)

        return self.norm(h)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HunyuanVLModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        # Hunyuan-VL configs keep text fields at the top level next to
        # vision_config; fold them into text_config (nested values win).
        text_config = dict(params.get("text_config") or {})
        for k, v in params.items():
            if k not in ("text_config", "vision_config") and k not in text_config:
                text_config[k] = v
        return cls(model_type=params["model_type"], text_config=text_config)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    @property
    def model(self):
        # Uniform access to the inner text model (multimodal side state lives there) —
        # the same shape as qwen3_5's Model.model property.
        return self.language_model.model

    def sanitize(self, weights):
        # Accepts mlx-vlm conversions ("language_model.model.*" + "vision_tower.*")
        # and raw Tencent checkpoints ("model.layers.*" + "vit.*").
        tied = self.language_model.args.tie_word_embeddings
        sanitized = {}
        for key, value in weights.items():
            if key.startswith(
                ("vit.", "vision_tower.", "model.vit.", "model.vision_tower.")
            ):
                continue
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model.", "language_model.model.", 1)
            elif not key.startswith("language_model."):
                key = "language_model." + key
            if "self_attn.rotary_emb.inv_freq" in key:
                continue
            if tied and key.startswith("language_model.lm_head."):
                continue
            sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.language_model.layers
