# Copyright © 2026 Apple Inc.
#
# GLM-family multimodal text models: glm4v_text (GLM-4.1V dense) and glm_ocr_text
# (GLM-OCR) — the GLM4 dense body with GLM-style 3D multimodal RoPE applied over
# PARTIAL rotary dims (glm4v: factor 0.5, mrope_section [8, 12, 12]; glm_ocr:
# factor 1.0, mrope_section [16, 24, 24]). The two bodies coincide field-for-field,
# so one file serves both, parameterized by config. Language architecture ported
# from Blaizzy/mlx-vlm's glm4v and glm_ocr models (MIT © Blaizzy/mlx-vlm
# contributors); the mrope side-state pattern follows this fork's qwen3_5.py.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = "glm4v_text"
    hidden_size: int = 4096
    num_hidden_layers: int = 40
    intermediate_size: int = 13696
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-5
    vocab_size: int = 151552
    attention_bias: bool = True
    max_position_embeddings: int = 65536
    tie_word_embeddings: bool = False
    # glm4v keeps rope fields flat (rope_scaling carries only the mrope_section);
    # glm_ocr nests everything under rope_parameters. Normalized in __post_init__.
    partial_rotary_factor: float = 0.5
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    rope_parameters: Optional[Dict] = None
    mrope_section: Optional[List[int]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        params = self.rope_parameters or self.rope_scaling or {}
        self.partial_rotary_factor = params.get(
            "partial_rotary_factor", self.partial_rotary_factor
        )
        self.rope_theta = params.get("rope_theta", self.rope_theta)
        self.mrope_section = params.get("mrope_section", [8, 12, 12])


def _rotate_half_even_odd(x):
    return mx.flatten(mx.stack([-x[..., 1::2], x[..., 0::2]], axis=-1), -2, -1)


def _apply_mrope(x, cos, sin):
    """Rotate the first cos.shape[-1] dims with even/odd (traditional) pairing."""
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_rot = x_rot * cos + _rotate_half_even_odd(x_rot) * sin
    if rotary_dim == x.shape[-1]:
        return x_rot
    return mx.concatenate([x_rot, x[..., rotary_dim:]], axis=-1)


class Attention(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # Text-only path: GLM rotates even/odd pairs, i.e. traditional RoPE over the
        # partial dims — 3D mrope with equal t/h/w axes degenerates to exactly this.
        self.rope = nn.RoPE(
            dims=int(self.head_dim * args.partial_rotary_factor),
            traditional=True,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if position_embeddings is not None:
            # Vision: 3D multimodal RoPE over the same attention weights.
            cos, sin = position_embeddings
            queries = _apply_mrope(queries, cos, sin)
            keys = _apply_mrope(keys, cos, sin)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
        elif cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            args.hidden_size, 2 * args.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x) -> mx.array:
        x = self.gate_up_proj(x)
        gate, up_states = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up_states)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_self_attn_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_mlp_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> mx.array:
        x = x + self.post_self_attn_layernorm(
            self.self_attn(self.input_layernorm(x), mask, cache, position_embeddings)
        )
        residual = x
        x = (
            self.post_mlp_layernorm(self.mlp(self.post_attention_layernorm(x)))
            + residual
        )
        return x


class Glm4vTextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Multimodal RoPE side state (mlx-unified), set by the vision path before
        # generation and cleared afterwards — same contract as qwen3_5.py. Underscore
        # attrs so MLX's module walker never registers them as parameters.
        rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self._inv_freq = 1.0 / (
            args.rope_theta
            ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim)
        )
        # Which mrope axis (t/h/w) feeds each frequency: chunked by mrope_section.
        self._axis_selector = mx.array(
            [axis for axis, n in enumerate(args.mrope_section) for _ in range(n)]
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

        None (the common text-only case) routes every layer to the original
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

    def _position_embeddings(self, position_ids: mx.array, dtype) -> tuple:
        """cos/sin (B, 1, L, rotary_dim) in even/odd layout from (3, B, L) positions."""
        positions = mx.take(position_ids, self._axis_selector, axis=0)
        angle = positions.transpose(1, 2, 0).astype(mx.float32) * self._inv_freq
        cos = mx.repeat(mx.cos(angle)[:, None], 2, axis=-1)
        sin = mx.repeat(mx.sin(angle)[:, None], 2, axis=-1)
        return cos.astype(dtype), sin.astype(dtype)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
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
            position_embeddings = self._position_embeddings(position_ids, h.dtype)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c, position_embeddings=position_embeddings)

        return self.norm(h)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Glm4vTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        # GLM-OCR ships a multi-token-prediction block as layer[num_hidden_layers].
        mtp_prefix = f"model.layers.{self.args.num_hidden_layers}."
        weights = {k: v for k, v in weights.items() if mtp_prefix not in k}
        if self.args.tie_word_embeddings:
            weights = {
                k: v for k, v in weights.items() if not k.endswith("lm_head.weight")
            }
        return weights


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


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
        sanitized = {}
        for key, value in weights.items():
            if key.startswith(("vision_tower", "visual", "model.visual")):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif not key.startswith("language_model."):
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    @property
    def layers(self):
        return self.language_model.layers
