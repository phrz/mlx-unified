# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 256000
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0
    query_pre_attn_scalar: float = 144.0

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @classmethod
    def from_dict(cls, params):
        # PaliGemma 2 checkpoints nest the text model's params under text_config
        # (the top-level model_type stays "paligemma"); the defaults above cover
        # the fields sparse configs omit (head_dim, rms_norm_eps).
        text_config = params.get("text_config")
        if text_config:
            params = {
                **params,
                **{k: v for k, v in text_config.items() if k != "model_type"},
            }
        return super().from_dict(params)


def _overlay_prefix_mask(base_mask: mx.array, prefix_mask: mx.array) -> mx.array:
    """OR the vision forward's bidirectional prefix edges onto a boolean causal
    mask (PaliGemma prefill: every non-pad prompt position attends to every
    other). Query rows are the LAST rows of the prompt-wide prefix mask and keys
    its last columns (gemma4_text's overlay slicing convention); a prefix mask
    that does not cover the base extent (a cached-prefix tail) stays causal."""
    query_len, key_len = base_mask.shape[-2], base_mask.shape[-1]
    if prefix_mask.shape[-2] < query_len or prefix_mask.shape[-1] < key_len:
        return base_mask
    edges = prefix_mask[..., -query_len:, -key_len:].astype(mx.bool_)
    if edges.ndim == 4 and edges.shape[0] == 1 and edges.shape[1] == 1:
        edges = edges[0, 0]  # 2D broadcasts like the causal base mask
    return base_mask | edges


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.repeats = n_heads // n_kv_heads
        self.head_dim = head_dim = args.head_dim

        self.scale = 1.0 / (args.query_pre_attn_scalar**0.5)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.attn_logit_softcapping = args.attn_logit_softcapping
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
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

        queries = queries * self.scale

        if self.repeats > 1:
            queries = queries.reshape(
                B, self.n_kv_heads, self.repeats, L, self.head_dim
            )
            keys = mx.expand_dims(keys, 2)
            values = mx.expand_dims(values, 2)

        scores = queries @ keys.swapaxes(-1, -2)
        scores = mx.tanh(scores / self.attn_logit_softcapping)
        scores *= self.attn_logit_softcapping

        if mask is not None:
            if mask.dtype == mx.bool_:
                scores = mx.where(
                    mask, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype)
                )
            else:
                scores = scores + mask
        scores = mx.softmax(scores, precise=True, axis=-1)
        output = scores @ values
        if self.repeats > 1:
            output = output.reshape(B, self.n_heads, L, self.head_dim)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        out = h + self.post_feedforward_layernorm(r)
        return out


class GemmaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Multimodal side state (mlx-unified), set by the vision path before
        # generation and cleared afterwards. Underscore attr — never a parameter.
        self._mm_attention_mask_4d = None

    def set_visual_state(self, attention_mask_4d: Optional[mx.array] = None) -> None:
        self._mm_attention_mask_4d = attention_mask_4d

    def reset_visual_state(self) -> None:
        self._mm_attention_mask_4d = None

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
        h = h * (self.args.hidden_size**0.5)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0], return_array=True)

        # PaliGemma 2 vision prefill: the whole prompt attends bidirectionally;
        # decode steps (single token) revert to normal causal behavior.
        prefix_mask = self._mm_attention_mask_4d
        if prefix_mask is not None and h.shape[1] > 1:
            mask = _overlay_prefix_mask(mask, prefix_mask)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.final_logit_softcapping = args.final_logit_softcapping
        self.model = GemmaModel(args)
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        out = self.model.embed_tokens.as_linear(out)
        out = mx.tanh(out / self.final_logit_softcapping)
        out = out * self.final_logit_softcapping
        return out

    def sanitize(self, weights):
        # PaliGemma 2 checkpoints (mlx-vlm conversions) wrap this model as their
        # text tower — drop the vision side, remap language_model.* onto this
        # module tree, and drop any materialized (tied) lm_head.
        sanitized = {}
        for k, v in weights.items():
            if k.startswith(("vision_tower.", "multi_modal_projector.")):
                continue
            k = k.removeprefix("language_model.")
            if k.startswith("lm_head.") or "self_attn.rotary_emb.inv_freq" in k:
                continue
            sanitized[k] = v
        return sanitized

    @property
    def layers(self):
        return self.model.layers
