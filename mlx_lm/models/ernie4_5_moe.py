# Copyright © 2023-2024 Apple Inc.
#
# ERNIE 4.5 VL support (3D multimodal RoPE, dual expert groups, VL-checkpoint
# sanitize) is ported from Blaizzy/mlx-vlm's ernie4_5_moe_vl (MIT © Blaizzy /
# mlx-vlm contributors), expressed as mlx-unified side state (see multimodal.py).

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    intermediate_size: int
    model_type: str
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    use_bias: bool
    tie_word_embeddings: bool
    moe_num_experts: Union[int, list[int]]  # VL checkpoints: [text, multimodal]
    moe_layer_start_index: Union[int, list[int]] = 0
    moe_intermediate_size: Union[int, list[int]] = 0
    moe_capacity: list[int] = field(default_factory=list)
    moe_k: int = 1
    moe_layer_interval: int = 1
    moe_use_aux_free: bool = False
    moe_num_shared_experts: int = 0
    moe_layer_end_index: Optional[Union[int, list[int]]] = None
    head_dim: Optional[int] = None
    moe_gate_act: str = "softmax"
    # 3D multimodal RoPE (VL checkpoints; see Ernie45Model.set_mrope_state).
    rope_scaling: Optional[dict] = None
    mrope_section: list[int] = field(default_factory=lambda: [22, 22, 20])

    def __post_init__(self):
        if self.rope_scaling and "mrope_section" in self.rope_scaling:
            self.mrope_section = list(self.rope_scaling["mrope_section"])


def _text_mm_split(value):
    """A [text, multimodal] pair from a VL config list, or (value, 0) for text-only."""
    if isinstance(value, (list, tuple)):
        return value[0], value[1]
    return value, 0


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or dim // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.use_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.use_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.use_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.use_bias)

        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=True,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
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


class Ernie4_5_MLP(nn.Module):
    def __init__(self, dim, hidden_dim, use_bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=use_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=use_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Ernie4_5_MoeMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.k = args.moe_k

        self.num_text_experts, self.num_mm_experts = _text_mm_split(
            args.moe_num_experts
        )
        text_size, mm_size = _text_mm_split(
            args.moe_intermediate_size or args.intermediate_size
        )

        self.gate = nn.Linear(args.hidden_size, self.num_text_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, text_size, self.num_text_experts, bias=args.use_bias
        )

        if self.num_mm_experts > 0:
            # VL checkpoints carry a second expert group for image/video positions
            # plus learned aux-free bias corrections (used only for selection).
            self.e_score_correction_bias = mx.zeros((self.num_text_experts,))
            self.gate_1 = nn.Linear(args.hidden_size, self.num_mm_experts, bias=False)
            self.e_score_correction_bias_1 = mx.zeros((self.num_mm_experts,))
            self.switch_mlp_1 = SwitchGLU(
                args.hidden_size, mm_size, self.num_mm_experts, bias=args.use_bias
            )

        if getattr(args, "moe_num_shared_experts", 0) > 0:
            self.shared_experts = Ernie4_5_MLP(
                args.hidden_size, text_size * args.moe_num_shared_experts, args.use_bias
            )
        else:
            self.shared_experts = None

        if args.moe_gate_act == "softmax":
            self.gate_act = nn.Softmax()
        elif args.moe_gate_act == "sigmoid":
            self.gate_act = nn.Sigmoid()
        else:
            raise ValueError(f"{args.moe_gate_act} is not supported.")

    def _route(self, x: mx.array, gate: nn.Module, bias: Optional[mx.array]) -> tuple:
        gates = self.gate_act(gate(x).astype(mx.float32))
        selection = gates if bias is None else gates + bias

        k = self.k
        inds = mx.stop_gradient(
            mx.argpartition(-selection, kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), 1e-12)
        return inds, scores

    def __call__(
        self, x: mx.array, token_type_ids: Optional[mx.array] = None
    ) -> mx.array:
        bias = self.e_score_correction_bias if self.num_mm_experts > 0 else None
        inds, scores = self._route(x, self.gate, bias)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        if self.num_mm_experts > 0 and token_type_ids is not None:
            inds, scores = self._route(x, self.gate_1, self.e_score_correction_bias_1)
            y_mm = self.switch_mlp_1(x, inds)
            y_mm = (y_mm * scores[..., None]).sum(axis=-2).astype(y_mm.dtype)
            y = mx.where((token_type_ids == 0)[..., None], y, y_mm)

        if self.shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class Ernie4_5_DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)
        # Multimodal RoPE support (mlx-unified): built lazily on the first vision
        # call — text-only use never imports mlx-vlm. Underscore attrs so MLX's
        # module walker doesn't register them (no weights to load/quantize).
        self._mrope = None
        self._rope_theta = args.rope_theta
        self._mrope_section = args.mrope_section

        moe_layer_start_index = (
            min(args.moe_layer_start_index)
            if isinstance(args.moe_layer_start_index, (tuple, list))
            else args.moe_layer_start_index
        )

        if args.moe_layer_end_index is None:
            moe_layer_end_index = args.num_hidden_layers - 1
        else:
            moe_layer_end_index = (
                max(args.moe_layer_end_index)
                if isinstance(args.moe_layer_end_index, (tuple, list))
                else args.moe_layer_end_index
            )

        if (
            ((layer_idx + 1) % args.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index
            and layer_idx <= moe_layer_end_index
        ):
            self.mlp = Ernie4_5_MoeMLP(args)
        else:
            self.mlp = Ernie4_5_MLP(
                args.hidden_size, args.intermediate_size, args.use_bias
            )

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if position_ids is None:
            # Text-only: the original traditional RoPE (3D rope with equal
            # t/h/w axes degenerates to exactly this).
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        else:
            # Vision: 3D multimodal RoPE over the SAME attention weights.
            r = self._mrope_attention(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        if isinstance(self.mlp, Ernie4_5_MoeMLP):
            r = self.mlp(self.post_attention_layernorm(h), token_type_ids)
        else:
            r = self.mlp(self.post_attention_layernorm(h))
        return h + r

    def _mrope_attention(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        position_ids: mx.array,
    ) -> mx.array:
        """MRoPE attention path reusing self.self_attn's weights.

        Mirrors ernie4_5_moe_vl's Attention.__call__ from mlx-vlm (MIT ©
        Blaizzy/mlx-vlm contributors) — in this fork it is first-class rather
        than a separate model.
        """
        # Heavy deps only on the vision path; text-only never reaches here.
        from mlx_vlm.models.ernie4_5_moe_vl.language import (
            Ernie4_5RotaryEmbedding,
            apply_rotary_pos_emb,
        )

        if self._mrope is None:
            self._mrope = Ernie4_5RotaryEmbedding(
                self.self_attn.head_dim,
                base=self._rope_theta,
                mrope_section=tuple(self._mrope_section),
            )

        attn = self.self_attn
        B, L, D = x.shape

        queries = attn.q_proj(x).reshape(B, L, attn.n_heads, -1).transpose(0, 2, 1, 3)
        keys = attn.k_proj(x).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = attn.v_proj(x).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Side-state positions are (3, B, L); ERNIE's rotary takes (B, L, 3).
        cos, sin = self._mrope(values, position_ids.transpose(1, 2, 0))
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=attn.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return attn.o_proj(output)


class Ernie45Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Ernie4_5_DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Multimodal side state (mlx-unified), set by the vision path before
        # generation and cleared afterwards. Underscore attrs so MLX's module
        # walker never registers them as parameters.
        self._mm_position_ids = None  # (3, B, L) t/h/w positions for the prompt
        self._mm_rope_deltas = None  # (B, 1) decode-position delta
        self._mm_token_type_ids = None  # (B, L): 0 text, nonzero image/video

    def set_mrope_state(self, position_ids: mx.array, rope_deltas: mx.array) -> None:
        """Install 3D multimodal positions (3, B, L) + rope delta for a vision prompt."""
        self._mm_position_ids = position_ids
        self._mm_rope_deltas = rope_deltas

    def reset_mrope_state(self) -> None:
        self._mm_position_ids = None
        self._mm_rope_deltas = None

    def set_visual_state(self, mm_token_type_ids=None) -> None:
        """Install prompt-aligned token types — VL checkpoints route image/video
        positions through the second expert group (see Ernie4_5_MoeMLP)."""
        self._mm_token_type_ids = mm_token_type_ids

    def reset_visual_state(self) -> None:
        self._mm_token_type_ids = None

    def _mm_cache_offset(self, cache) -> tuple:
        """First-entry KV offset as (array-or-int, python int) — 0 when fresh."""
        if cache is None or cache[0] is None:
            return 0, 0
        offset = cache[0].offset
        if isinstance(offset, mx.array):
            if offset.ndim == 0:
                scalar = offset.item()
                return scalar, scalar
            return offset, offset[0].item()
        return offset, offset

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

        cache_offset, cache_offset_scalar = self._mm_cache_offset(cache)
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

    def _compute_token_type_ids(self, inputs: mx.array, cache) -> Optional[mx.array]:
        """Prompt-aligned token types for the current chunk — None once every
        position in this forward is past the stored prompt (decode is text-only)."""
        if self._mm_token_type_ids is None:
            return None

        _, offset = self._mm_cache_offset(cache)
        batch_size, seq_length = inputs.shape

        stored = self._mm_token_type_ids
        if stored.shape[0] == 1 and batch_size > 1:
            stored = mx.broadcast_to(stored, (batch_size, stored.shape[1]))
        stored_len = stored.shape[1]
        if offset >= stored_len:
            return None

        chunk = stored[:, offset : min(offset + seq_length, stored_len)]
        if chunk.shape[1] < seq_length:
            pad = mx.zeros((batch_size, seq_length - chunk.shape[1]), dtype=chunk.dtype)
            chunk = mx.concatenate([chunk, pad], axis=1)
        return chunk

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

        # None for text-only requests — layers take the original 1D-RoPE path
        # and text-expert-only routing.
        position_ids = self._compute_position_ids(inputs, cache)
        token_type_ids = self._compute_token_type_ids(inputs, cache)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(
                h, mask, c, position_ids=position_ids, token_type_ids=token_type_ids
            )

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Ernie45Model(args)
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

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        # VL checkpoints (raw HF or mlx-vlm layout): strip the vision tower and
        # resampler, then remap language_model.* onto this text-only module tree.
        sanitized = {}
        for key, value in weights.items():
            if key.startswith(
                (
                    "vision_tower.",
                    "vision_model.",
                    "resampler_model.",
                    "model.resampler_model.",
                )
            ):
                continue
            if key.startswith("language_model."):
                key = key[len("language_model.") :]
            sanitized[key] = value
        weights = sanitized

        num_text_experts, num_mm_experts = _text_mm_split(self.args.moe_num_experts)

        remove_patterns = [
            "mtp_block.",
            "mtp_linear_proj.",
            "mtp_hidden_norm.",
            "mtp_emb_norm.",
        ]
        # Single-expert-group checkpoints don't model aux-free bias correction.
        if num_mm_experts == 0:
            remove_patterns.append("e_score_correction_bias")

        weights = {
            key: value
            for key, value in weights.items()
            if not any(pattern in key for pattern in remove_patterns)
        }

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"

            # Stack per-expert weights: text experts → switch_mlp, multimodal
            # experts (raw HF appends them after the text group) → switch_mlp_1.
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(num_text_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
                    if (
                        num_mm_experts > 0
                        and f"{prefix}.mlp.experts.{num_text_experts}.{m}.{k}"
                        in weights
                    ):
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(
                                num_text_experts, num_text_experts + num_mm_experts
                            )
                        ]
                        weights[f"{prefix}.mlp.switch_mlp_1.{m}.{k}"] = mx.stack(
                            to_join
                        )

            if num_mm_experts == 0:
                continue

            # Raw HF VL layouts (Paddle heritage): gates stored (hidden, experts),
            # the multimodal gate as gate.weight_1, and both groups' aux-free
            # biases stacked in moe_statics.
            gate_key = f"{prefix}.mlp.gate.weight"
            if gate_key in weights:
                w = weights[gate_key]
                if w.shape[0] > w.shape[1]:
                    weights[gate_key] = w.T
            gate_1_key = f"{prefix}.mlp.gate.weight_1"
            if gate_1_key in weights:
                w = weights.pop(gate_1_key)
                weights[f"{prefix}.mlp.gate_1.weight"] = (
                    w.T if w.shape[0] > w.shape[1] else w
                )
            bias_key = f"{prefix}.mlp.moe_statics.e_score_correction_bias"
            if bias_key in weights:
                bias = weights.pop(bias_key)
                weights[f"{prefix}.mlp.e_score_correction_bias"] = bias[0]
                weights[f"{prefix}.mlp.e_score_correction_bias_1"] = bias[1]

        return weights
