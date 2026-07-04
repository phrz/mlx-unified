# Copyright © 2026 Apple Inc.
#
# Qwen3-Omni-MoE thinker text model: the Qwen3-MoE-shaped decoder that generates
# the chat response. Language architecture ported from Blaizzy/mlx-vlm's
# models/qwen3_omni_moe (MIT © Blaizzy/mlx-vlm contributors) — thinker only; the
# Talker/Code2Wav TTS stack is out of scope. Two pieces of multimodal side state:
# interleaved 3D multimodal RoPE (set_mrope_state/reset_mrope_state, the pattern
# from this fork's qwen3_5.py) and deepstack mid-layer visual injection
# (set_visual_state/reset_visual_state: the vision tower's multiscale features
# are added to the hidden state at image positions after the first few layers).

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    head_dim: Optional[int] = None
    max_position_embeddings: int = 65536
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str, bool, List[int]]]] = None
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])

    @classmethod
    def from_dict(cls, params):
        # The checkpoint config nests the thinker's decoder config as
        # thinker_config.text_config (talker/code2wav configs are ignored).
        if "thinker_config" in params:
            params = {
                **params["thinker_config"]["text_config"],
                "model_type": params["model_type"],
            }
        elif "text_config" in params:
            params = {
                **params["text_config"],
                "model_type": params.get("model_type", "qwen3_omni_moe"),
            }
        return super().from_dict(params)

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_scaling and "mrope_section" in self.rope_scaling:
            self.mrope_section = list(self.rope_scaling["mrope_section"])
        if sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError(
                f"mrope_section {self.mrope_section} must sum to head_dim/2 "
                f"({self.head_dim // 2})"
            )


def _interleaved_selector(mrope_section: List[int], half_dim: int) -> mx.array:
    """Per-frequency position-axis selector for INTERLEAVED mrope: frequency i
    reads its position from t/h/w in a repeating t,h,w pattern for the first
    3·mrope_section[1] slots, then t for the remainder (Qwen3-Omni layout)."""
    selector = [0] * half_dim
    for axis, offset in enumerate((1, 2), start=1):
        for idx in range(offset, min(mrope_section[axis] * 3, half_dim), 3):
            selector[idx] = axis
    return mx.array(selector, dtype=mx.int32)


def _interleaved_rope_cos_sin(
    position_ids: mx.array,
    inv_freq: mx.array,
    selector: mx.array,
    dtype: mx.Dtype,
) -> Tuple[mx.array, mx.array]:
    """cos/sin (B, 1, L, head_dim) from 3D positions (3, B, L): each frequency
    band's angle uses the position axis chosen by the interleaved selector."""
    positions = mx.take(position_ids, selector, axis=0).transpose(1, 2, 0)
    freqs = positions.astype(mx.float32) * inv_freq  # (B, L, d/2)
    emb = mx.concatenate([freqs, freqs], axis=-1)
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

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if position_embeddings is not None:
            # Vision/audio: interleaved 3D mrope from explicit positions. With
            # equal t/h/w positions this reproduces the plain 1D path exactly.
            cos, sin = position_embeddings
            queries = _apply_rotary(queries, cos, sin)
            keys = _apply_rotary(keys, cos, sin)
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
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen3OmniMoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3OmniMoeSparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

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


class Qwen3OmniMoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Underscore attrs so MLX's module walker never registers them (no
        # weights to load/quantize). The inverse rope frequencies + axis
        # selector feed the interleaved mrope path; the mm_* attrs are the
        # multimodal side state, set by the vision path before generation and
        # cleared afterwards.
        self._inv_freq = args.rope_theta ** (
            -mx.arange(0, args.head_dim, 2, dtype=mx.float32) / args.head_dim
        )
        self._selector = _interleaved_selector(args.mrope_section, args.head_dim // 2)
        self._mm_position_ids = None
        self._mm_rope_deltas = None
        self._mm_visual_pos_masks = None
        self._mm_deepstack_embeds = None

    def set_mrope_state(self, position_ids: mx.array, rope_deltas: mx.array) -> None:
        """Install 3D multimodal positions (3, B, L) + rope delta for a vision prompt."""
        self._mm_position_ids = position_ids
        self._mm_rope_deltas = rope_deltas

    def reset_mrope_state(self) -> None:
        self._mm_position_ids = None
        self._mm_rope_deltas = None

    def set_visual_state(
        self,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[List[mx.array]] = None,
    ) -> None:
        """Install deepstack state for a vision prompt: a (1, L) / (L,) boolean
        mask of image-token positions and the vision tower's multiscale features
        (one (n_visual, hidden) table per early decoder layer, position-aligned
        with the mask's True entries in order)."""
        self._mm_visual_pos_masks = visual_pos_masks
        self._mm_deepstack_embeds = deepstack_visual_embeds

    def reset_visual_state(self) -> None:
        self._mm_visual_pos_masks = None
        self._mm_deepstack_embeds = None

    def _cache_offset(self, cache) -> int:
        if cache is None or cache[0] is None:
            return 0
        offset = cache[0].offset
        if isinstance(offset, int):
            return offset
        return offset.item() if offset.ndim == 0 else offset[0].item()

    def _compute_position_ids(
        self, batch_size: int, seq_length: int, cache_offset: int
    ) -> Optional[mx.array]:
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

        if self._mm_position_ids is not None:
            stored_seq_length = self._mm_position_ids.shape[2]
            if cache_offset < stored_seq_length:
                stored_end = min(cache_offset + seq_length, stored_seq_length)
                stored_positions = self._mm_position_ids[:, :, cache_offset:stored_end]
                if stored_end - cache_offset == seq_length:
                    return stored_positions
                tail_positions = self._sequential_position_ids(
                    batch_size=batch_size,
                    seq_length=seq_length - (stored_end - cache_offset),
                    start_offset=stored_seq_length,
                )
                return mx.concatenate([stored_positions, tail_positions], axis=2)

        return self._sequential_position_ids(
            batch_size=batch_size, seq_length=seq_length, start_offset=cache_offset
        )

    def _sequential_position_ids(
        self, *, batch_size: int, seq_length: int, start_offset: int
    ) -> mx.array:
        delta = mx.array(start_offset)
        if self._mm_rope_deltas is not None:
            delta = delta + self._mm_rope_deltas

        position_ids = mx.arange(seq_length).reshape(1, -1)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))

        if delta.ndim == 0:
            delta = mx.broadcast_to(delta.reshape(1, 1), (batch_size, 1))
        else:
            delta = delta.reshape(-1, 1)[:batch_size]
            if delta.shape[0] == 1 and batch_size > 1:
                delta = mx.broadcast_to(delta, (batch_size, 1))

        position_ids = mx.add(position_ids, delta)[None, ...]
        return mx.broadcast_to(position_ids, (3, batch_size, seq_length))

    def _deepstack_window(self, cache_offset: int, seq_length: int, dtype: mx.Dtype):
        """(rows, scale) aligning the deepstack tables to the current forward
        window: per-position table row indices (S,) and a (1, S, 1) 0/1 gate.
        (None, None) once the prompt — and every image span — is fully cached."""
        if self._mm_deepstack_embeds is None or self._mm_visual_pos_masks is None:
            return None, None
        mask = self._mm_visual_pos_masks.reshape(-1).astype(mx.int32)
        if cache_offset >= mask.shape[0]:
            return None, None
        window = mask[cache_offset : cache_offset + seq_length]
        if window.shape[0] < seq_length:
            window = mx.concatenate(
                [window, mx.zeros(seq_length - window.shape[0], dtype=mx.int32)]
            )
        rows = mx.cumsum(window) - 1 + mask[:cache_offset].sum()
        # Text positions gather an arbitrary (clamped) row; the gate zeroes them.
        rows = mx.maximum(rows, 0)
        return rows, window.astype(dtype)[None, :, None]

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        cache_offset = self._cache_offset(cache)
        B, S = h.shape[0], h.shape[1]

        # None for text-only requests — layers take the plain 1D-RoPE path.
        position_ids = self._compute_position_ids(B, S, cache_offset)
        position_embeddings = None
        if position_ids is not None:
            position_embeddings = _interleaved_rope_cos_sin(
                position_ids, self._inv_freq, self._selector, h.dtype
            )

        ds_rows, ds_gate = self._deepstack_window(cache_offset, S, h.dtype)
        ds_embeds = self._mm_deepstack_embeds if ds_rows is not None else []

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c, position_embeddings)
            if i < len(ds_embeds):
                h = h + mx.take(ds_embeds[i].astype(h.dtype), ds_rows, axis=0) * ds_gate

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3OmniMoeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # Accept both the raw HF checkpoint (thinker.model.* / thinker.lm_head.*
        # / thinker.visual.* / thinker.audio_tower.*) and an mlx-vlm-made
        # conversion (thinker.language_model.model.* / thinker.vision_tower.*):
        # keep only the thinker's text decoder — the vision/audio towers and
        # the Talker/Code2Wav TTS stack are dropped.
        sanitized = {}
        for k, v in weights.items():
            if k.startswith(("talker.", "code2wav.")):
                continue
            k = k.removeprefix("thinker.")
            if k.startswith(("visual.", "vision_tower.", "audio_tower.")):
                continue
            k = k.removeprefix("language_model.")
            if not k.startswith(("model.", "lm_head.")):
                continue
            sanitized[k] = v
        if self.args.tie_word_embeddings:
            sanitized.pop("lm_head.weight", None)
        # The raw checkpoint stores each expert separately; mlx-vlm conversions
        # arrive already stacked as switch_mlp.
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                if f"{prefix}.mlp.experts.0.{n}.weight" in sanitized:
                    to_join = [
                        sanitized.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    sanitized[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join)
        return sanitized

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers
