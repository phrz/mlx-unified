# Copyright © 2026 Apple Inc.
#
# Qwen3-VL text model with first-class multimodal support (mlx-unified):
# interleaved 3D multimodal RoPE (set_mrope_state/reset_mrope_state) and
# deepstack mid-layer visual injection (set_visual_state/reset_visual_state).
# Language-model architecture ported from Blaizzy/mlx-vlm's qwen3_vl
# (MIT © Blaizzy/mlx-vlm contributors).

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .qwen3 import MLP
from .qwen3_moe import Qwen3MoeSparseMoeBlock as SparseMoeBlock
from .rope_utils import initialize_rope


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = "qwen3_vl_text"
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    intermediate_size: int = 6144
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    head_dim: Optional[int] = None
    rope_theta: float = 5000000.0
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False

    # MoE fields (qwen3_vl_moe); dense checkpoints leave num_experts at 0.
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    rope_scaling: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {"type": "default", "mrope_section": [24, 20, 20]}
    )

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_scaling:
            if "type" not in self.rope_scaling and "rope_type" in self.rope_scaling:
                self.rope_scaling["type"] = self.rope_scaling.pop("rope_type")


class Attention(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        # Text-only path: interleaved mrope with equal t/h/w axes degenerates to
        # exactly this 1D RoPE ("mrope"/"default" scaling both yield nn.RoPE).
        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
        # Interleaved multimodal RoPE (mlx-unified): built lazily on the first
        # vision call — text-only use never imports mlx-vlm. Underscore attrs so
        # MLX's module walker doesn't register them (no weights to load/quantize).
        self._mrope = None
        self._rope_scaling = args.rope_scaling
        self._rope_theta = args.rope_theta
        self._max_position_embeddings = args.max_position_embeddings
        self._head_dim = head_dim

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
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

        if position_ids is None:
            # Text-only: plain 1D RoPE by cache offset.
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)
        else:
            # Vision: interleaved 3D multimodal RoPE (NOT qwen3_5's chunked style)
            # over the same attention weights. Heavy dep only on the vision path.
            if self._mrope is None:
                from mlx_vlm.models.qwen3_vl.language import Qwen3VLRotaryEmbedding

                self._mrope = Qwen3VLRotaryEmbedding(
                    self._head_dim,
                    max_position_embeddings=self._max_position_embeddings,
                    base=self._rope_theta,
                    rope_scaling=self._rope_scaling,
                )
            queries, keys = self._mrope.apply_rotary(
                queries, keys, position_ids, unsqueeze_dim=1
            )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        if (
            args.num_experts > 0
            and layer_idx not in args.mlp_only_layers
            and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3VLTextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Multimodal side state (mlx-unified), set by the vision path before
        # generation and cleared afterwards. Underscore attrs so MLX's module
        # walker never registers them as parameters. Stored arrays describe the
        # PROMPT BEING PREFILLED; the cache offset at the first forward after
        # set_* anchors them (a prompt-cache hit means prefill starts mid-cache).
        self._mm_position_ids = None  # (3, B, L) interleaved-mrope positions
        self._mm_rope_deltas = None  # (B, 1), full-prompt-relative
        self._mm_base_offset = None
        self._visual_pos_masks = None  # (B, L) bool: True at image positions
        self._visual_ordinals = None  # (B, L): row-major ordinal among visual tokens
        self._deepstack_visual_embeds = None  # per early layer: (n_visual, hidden)
        self._visual_base_offset = None

    def set_mrope_state(self, position_ids: mx.array, rope_deltas: mx.array) -> None:
        """Install 3D multimodal positions (3, B, L) + rope delta for a vision prompt."""
        self._mm_position_ids = position_ids
        self._mm_rope_deltas = rope_deltas
        self._mm_base_offset = None

    def reset_mrope_state(self) -> None:
        self._mm_position_ids = None
        self._mm_rope_deltas = None
        self._mm_base_offset = None

    def set_visual_state(
        self,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[List[mx.array]] = None,
    ) -> None:
        """Install the deepstack overlay for a vision prompt: a (B, L) bool mask of
        visual positions and one flat (n_visual, hidden) embed table per early layer
        (layer i adds deepstack_visual_embeds[i] into its output at visual positions)."""
        self._visual_pos_masks = visual_pos_masks
        self._deepstack_visual_embeds = deepstack_visual_embeds
        self._visual_base_offset = None
        if visual_pos_masks is not None:
            # Row-major cumulative count maps each masked position to its row in
            # the flat embed tables (mlx-vlm consumes them in the same order).
            flat = visual_pos_masks.reshape(-1).astype(mx.int32)
            self._visual_ordinals = (mx.cumsum(flat) - 1).reshape(
                visual_pos_masks.shape
            )
        else:
            self._visual_ordinals = None

    def reset_visual_state(self) -> None:
        self._visual_pos_masks = None
        self._visual_ordinals = None
        self._deepstack_visual_embeds = None
        self._visual_base_offset = None

    @staticmethod
    def _cache_offset(cache):
        """(offset, python-int offset) of the first layer's cache before update."""
        if cache is None or cache[0] is None:
            return 0, 0
        offset = cache[0].offset
        if isinstance(offset, int):
            return offset, offset
        if offset.ndim == 0:
            return offset.item(), int(offset.item())
        return offset, int(offset[0].item())

    def _compute_position_ids(self, inputs: mx.array, cache) -> Optional[mx.array]:
        """Positions for the current forward pass from stored multimodal state.

        None (the common text-only case) routes every layer to the original
        1D-RoPE path. During chunked prefill the stored prompt positions are
        sliced by cache offset relative to where prefill started (stitching a
        sequential tail if a chunk crosses the end); once the prompt is
        exhausted, decode positions are cache_offset + rope_delta broadcast to
        all three mrope axes.
        """
        if self._mm_position_ids is None and self._mm_rope_deltas is None:
            return None
        if self._mm_position_ids is not None and self._mm_rope_deltas is None:
            raise ValueError(
                "MRoPE state is inconsistent: position_ids set without rope_deltas."
            )

        cache_offset, cache_offset_scalar = self._cache_offset(cache)
        if self._mm_base_offset is None:
            self._mm_base_offset = cache_offset_scalar

        batch_size, seq_length = inputs.shape

        if self._mm_position_ids is not None:
            stored_seq_length = self._mm_position_ids.shape[2]
            rel_offset = cache_offset_scalar - self._mm_base_offset
            if rel_offset < stored_seq_length:
                stored_end = min(rel_offset + seq_length, stored_seq_length)
                stored_positions = self._mm_position_ids[:, :, rel_offset:stored_end]
                if stored_end - rel_offset == seq_length:
                    return stored_positions
                tail_positions = self._sequential_position_ids(
                    batch_size=batch_size,
                    seq_length=seq_length - (stored_end - rel_offset),
                    start_offset=self._mm_base_offset + stored_seq_length,
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

    def _deepstack_window(self, seq_length: int, cache):
        """(mask, ordinal) slices of the stored visual state for this forward.

        Deepstack injection happens ONLY during the prefill forwards that overlap
        the stored image span; once the cache offset moves past the stored mask —
        i.e. every decode step — this returns None and the layer loop skips it,
        so decode is byte-identical to plain mrope generation.
        """
        if self._deepstack_visual_embeds is None or self._visual_pos_masks is None:
            return None
        _, offset = self._cache_offset(cache)
        if self._visual_base_offset is None:
            self._visual_base_offset = offset
        start = offset - self._visual_base_offset
        masks = self._visual_pos_masks
        if start >= masks.shape[1]:
            return None
        window_mask = masks[:, start : start + seq_length]
        window_ordinal = self._visual_ordinals[:, start : start + seq_length]
        if window_mask.shape[1] < seq_length:
            pad = seq_length - window_mask.shape[1]
            batch = masks.shape[0]
            window_mask = mx.concatenate(
                [window_mask, mx.zeros((batch, pad), dtype=mx.bool_)], axis=1
            )
            window_ordinal = mx.concatenate(
                [window_ordinal, mx.zeros((batch, pad), dtype=window_ordinal.dtype)],
                axis=1,
            )
        return window_mask, window_ordinal

    @staticmethod
    def _apply_deepstack(h: mx.array, window, visual_embeds: mx.array) -> mx.array:
        """Add visual_embeds into h at visual positions (mlx-vlm's
        Qwen3VLModel._deepstack_process, as a sync-free gather + where)."""
        window_mask, window_ordinal = window
        n_visual = visual_embeds.shape[0]
        gathered = visual_embeds[mx.clip(window_ordinal, 0, n_visual - 1)]
        return mx.where(window_mask[..., None], h + gathered.astype(h.dtype), h)

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

        mask = create_attention_mask(h, cache[0])
        # Both None for text-only requests — the original code path, untouched.
        position_ids = self._compute_position_ids(inputs, cache)
        deepstack = self._deepstack_window(h.shape[1], cache)

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c, position_ids)
            if deepstack is not None and layer_idx < len(
                self._deepstack_visual_embeds
            ):
                h = self._apply_deepstack(
                    h, deepstack, self._deepstack_visual_embeds[layer_idx]
                )

        return self.norm(h)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3VLTextModel(args)
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
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        if self.args.num_experts <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate


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
        return self.language_model.model

    def sanitize(self, weights):
        # Loads mlx-vlm conversions (language_model.*/vision_tower.*), original HF
        # checkpoints (model.language_model.*/model.visual.*), and text-only
        # mlx-lm conversions (model.*) — vision-tower weights are stripped.
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif not key.startswith("language_model."):
                key = "language_model." + key
            sanitized[key] = value
        if self.language_model.args.tie_word_embeddings:
            sanitized.pop("language_model.lm_head.weight", None)
        return sanitized

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate
