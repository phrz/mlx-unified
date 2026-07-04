# Copyright © 2026 Apple Inc.
#
# Llama-3.2-Vision (mllama) TEXT decoder: ordinary Llama self-attention layers
# interleaved with dedicated cross-attention layers (config.cross_attention_layers)
# whose K/V come from the FIXED projected vision states, gated by tanh gates.
# Ported from Blaizzy/mlx-vlm mlx_vlm/models/mllama (MIT © Blaizzy/mlx-vlm
# contributors); vision arrives through mlx-unified's side-state hooks
# (set_visual_state/reset_visual_state) rather than forward kwargs. Unlike
# deepstack-style injection, the vision states stay live for the WHOLE
# generation: cross-attention layers consume them on every decode step, reusing
# K/V computed once at prefill (VisionKVCache). With no visual state set the
# cross-attention layers are skipped outright (HF does the same for text-only
# prompts), leaving pure-Llama semantics.

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
from .rope_utils import initialize_rope


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "mllama_text_model"
    hidden_size: int = 4096
    num_hidden_layers: int = 40
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-5
    vocab_size: int = 128256
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    rope_traditional: bool = False
    rope_scaling: Optional[dict] = None
    cross_attention_layers: List[int] = field(
        default_factory=lambda: [3, 8, 13, 18, 23, 28, 33, 38]
    )

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: Union[TextArgs, dict]

    def __post_init__(self):
        if not isinstance(self.text_config, TextArgs):
            self.text_config = TextArgs.from_dict(self.text_config)


class VisionKVCache:
    """Cross-attention K/V holder: computed ONCE from the fixed vision states at
    prefill and reused verbatim on every later forward. The sequence axis here is
    vision positions, not text tokens, so text-positional cache operations
    (trimming) leave it valid and untouched."""

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update(self, keys, values):
        self.keys, self.values = keys, values
        self.offset = keys.shape[2]

    def fetch(self):
        return self.keys, self.values

    def empty(self):
        return self.keys is None

    def size(self):
        return 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        # Absorb any text trim: vision K/V are not text-positional.
        return n

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2] if self.keys is not None else 0

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class Attention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = dim // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = initialize_rope(
            head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
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


class CrossAttention(nn.Module):
    """Text-to-vision cross-attention: queries from text hidden states, K/V from
    the projected vision states, per-head-dim RMS q/k norms, no RoPE."""

    def __init__(self, args: TextArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = dim // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        vision_states: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[VisionKVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        queries = self.q_norm(queries)

        if cache is not None and not cache.empty():
            keys, values = cache.fetch()
        else:
            keys = (
                self.k_proj(vision_states)
                .reshape(B, -1, self.n_kv_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            values = (
                self.v_proj(vision_states)
                .reshape(B, -1, self.n_kv_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            keys = self.k_norm(keys)
            if cache is not None:
                cache.update(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: TextArgs):
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
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class CrossAttentionBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.cross_attn = CrossAttention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.cross_attn_attn_gate = mx.zeros(1)
        self.cross_attn_mlp_gate = mx.zeros(1)

    def __call__(
        self,
        x: mx.array,
        vision_states: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        full_text_row_masked_out_mask: Optional[mx.array] = None,
        cache: Optional[VisionKVCache] = None,
    ) -> mx.array:
        r = self.cross_attn(self.input_layernorm(x), vision_states, mask, cache)
        h = x + mx.tanh(self.cross_attn_attn_gate) * r
        r = self.mlp(self.post_attention_layernorm(h))
        if full_text_row_masked_out_mask is not None:
            # Text rows with NO visible image get an all-zero (not -inf) mask row —
            # their attention output is meaningless — so the checkpoint semantics
            # zero the MLP branch for those rows (HF applies it in the same spot).
            r = full_text_row_masked_out_mask[:, 0].astype(r.dtype) * r
        return h + mx.tanh(self.cross_attn_mlp_gate) * r


class MllamaModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.cross_attention_layers = set(args.cross_attention_layers)
        # 8 extra rows hold mllama's special multimodal tokens (<|image|> = 128256
        # sits past vocab_size); lm_head still projects to vocab_size.
        self.embed_tokens = nn.Embedding(args.vocab_size + 8, args.hidden_size)
        self.layers = [
            (
                CrossAttentionBlock(args)
                if i in self.cross_attention_layers
                else TransformerBlock(args)
            )
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._self_idx = next(
            (
                i
                for i in range(args.num_hidden_layers)
                if i not in self.cross_attention_layers
            ),
            0,
        )

        # Multimodal side state (mlx-unified), set by the vision path before
        # generation and cleared afterwards. Underscore attrs so MLX's module
        # walker never registers them as parameters. Mask rows describe the
        # PROMPT BEING PREFILLED; the cache offset at the first forward after
        # set_visual_state anchors them (a prompt-cache hit means prefill starts
        # mid-cache).
        self._cross_attention_states = None  # (B, V, hidden) projected vision states
        self._cross_attention_mask = None  # (B, 1, L_prompt, V) additive overlay
        self._full_text_row_masked_out_mask = None  # (B, 1, L_prompt, 1)
        self._xattn_base_offset = None

    def set_visual_state(
        self,
        cross_attention_states: Optional[mx.array] = None,
        cross_attention_mask: Optional[mx.array] = None,
        full_text_row_masked_out_mask: Optional[mx.array] = None,
    ) -> None:
        """Install the vision overlay for a prompt: projected vision states
        (B, V, hidden), the PREPARED additive text→vision mask (B, 1, L, V) with
        masked-out rows already zeroed, and the (B, 1, L, 1) full-row mask that
        gates the MLP branch — all exactly as mlx-vlm's mllama
        get_input_embeddings returns them."""
        self._cross_attention_states = cross_attention_states
        self._cross_attention_mask = cross_attention_mask
        self._full_text_row_masked_out_mask = full_text_row_masked_out_mask
        self._xattn_base_offset = None

    def reset_visual_state(self) -> None:
        self._cross_attention_states = None
        self._cross_attention_mask = None
        self._full_text_row_masked_out_mask = None
        self._xattn_base_offset = None

    def _cache_offset(self, cache) -> int:
        c = cache[self._self_idx]
        if c is None:
            return 0
        offset = c.offset
        return offset if isinstance(offset, int) else int(offset.max().item())

    def _xattn_window(self, seq_length: int, dtype, cache):
        """(mask, full-row mask) rows for this forward. Prompt rows are anchored
        at the cache offset of the first forward after set_visual_state; rows past
        the stored prompt (decode steps) clamp to the LAST row — a generated token
        sees exactly what the prompt's final token saw, matching HF's
        repeat-last-row mask extension during generation."""
        if self._cross_attention_mask is None:
            return None, None
        offset = self._cache_offset(cache)
        if self._xattn_base_offset is None:
            self._xattn_base_offset = offset
        start = offset - self._xattn_base_offset
        n_rows = self._cross_attention_mask.shape[2]
        rows = mx.clip(mx.arange(start, start + seq_length), 0, n_rows - 1)
        mask = mx.take(self._cross_attention_mask, rows, axis=2).astype(dtype)
        full_row = None
        if self._full_text_row_masked_out_mask is not None:
            full_row = mx.take(self._full_text_row_masked_out_mask, rows, axis=2)
        return mask, full_row

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

        mask = create_attention_mask(h, cache[self._self_idx])
        xattn_mask, full_row = self._xattn_window(h.shape[1], h.dtype, cache)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if i in self.cross_attention_layers:
                # Text-only path: no vision state and no cached vision K/V — skip
                # the layer outright (HF does the same), pure-Llama semantics.
                if self._cross_attention_states is None and (c is None or c.empty()):
                    continue
                h = layer(
                    h,
                    vision_states=self._cross_attention_states,
                    mask=xattn_mask,
                    full_text_row_masked_out_mask=full_row,
                    cache=c,
                )
            else:
                h = layer(h, mask, cache=c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.model = MllamaModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.lm_head(self.model(inputs, cache, input_embeddings))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = LanguageModel(args.text_config)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(inputs, cache, input_embeddings)

    def sanitize(self, weights):
        new_weights = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue
            # transformers >= 4.52 nests the towers under a top-level "model."
            # and drops the inner ".model" from the language keys.
            starts_w_model = k.startswith("model.")
            k = k.removeprefix("model.")
            if k.startswith(("vision_tower", "vision_model", "multi_modal_projector")):
                continue
            if starts_w_model and k.startswith("language_model."):
                k = k.replace("language_model.", "language_model.model.", 1)
            elif k.startswith("lm_head."):
                k = "language_model." + k
            new_weights[k] = v
        return new_weights

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def model(self):
        # Uniform access to the inner text model (multimodal side state lives
        # there) — the same shape as gemma4's Model.model property.
        return self.language_model.model

    def make_cache(self):
        cross = self.language_model.model.cross_attention_layers
        return [
            VisionKVCache() if i in cross else KVCache()
            for i in range(len(self.layers))
        ]
