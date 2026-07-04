# Copyright © 2026 Apple Inc.
#
# moondream2 text model: a parallel-residual Phi-1.5-style decoder with a fused QKV
# projection, partial rotary embeddings (factor 0.5) and LayerNorm. Ported from
# Blaizzy/mlx-vlm's moondream2 language model (MIT © Blaizzy/mlx-vlm contributors).
# The vision tower stays in mlx-vlm; vision prompts arrive as input_embeddings plus
# a bidirectional-prefix attention mask via set_visual_state (see mlx_lm/multimodal.py).

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "moondream2"
    text_config: Optional[dict] = None
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    vocab_size: int = 51200
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    partial_rotary_factor: float = 0.5
    layer_norm_eps: float = 1e-5

    def __post_init__(self):
        # Multimodal checkpoints nest the text fields under text_config; the
        # reference config calls the LayerNorm eps "rms_norm_eps".
        for k, v in (self.text_config or {}).items():
            if k == "rms_norm_eps":
                self.layer_norm_eps = v
            elif k != "model_type" and hasattr(self, k):
                setattr(self, k, v)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim**-0.5

        qkv_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.qkv = nn.Linear(dim, qkv_dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.rope = nn.RoPE(
            int(self.head_dim * args.partial_rotary_factor),
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.qkv(x)
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        queries, keys, values = mx.split(qkv, [q_dim, q_dim + kv_dim], axis=-1)

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
        return self.proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=True)
        self.fc2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ln = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.attn = Attention(args)
        self.mlp = MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.ln(x)
        return x + self.attn(h, mask, cache) + self.mlp(h)


class Moondream2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.post_ln = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

        # Multimodal side state (mlx-unified), set by the vision path before
        # generation and cleared afterwards. Underscore attr — never registered as
        # a parameter. (1, 1, L, L) additive mask over the FULL prompt: bos + image
        # tokens form a bidirectional prefix, text stays causal.
        self._attention_mask_4d = None

    def set_visual_state(self, attention_mask_4d: Optional[mx.array] = None) -> None:
        self._attention_mask_4d = attention_mask_4d

    def reset_visual_state(self) -> None:
        self._attention_mask_4d = None

    def _make_mask(self, h: mx.array, cache) -> Optional[mx.array]:
        m = self._attention_mask_4d
        if m is not None:
            offset = cache[0].offset if cache[0] is not None else 0
            L = h.shape[1]
            # The stored mask is indexed by absolute prompt position. Decode steps
            # past its end sit beyond the bidirectional prefix — the prefix block
            # only rewrites rows INSIDE it — so those rows are plain causal.
            if offset + L <= m.shape[-1]:
                return m[..., offset : offset + L, : offset + L].astype(h.dtype)
        return create_attention_mask(h, cache[0])

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

        mask = self._make_mask(h, cache)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.post_ln(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Moondream2Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=True)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        return self.lm_head(self.model(inputs, cache, input_embeddings))

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            # mlx-vlm conversion layout: text.* beside a vision tower.
            if k.startswith(("vision.", "region.")):
                continue
            if k.startswith("text.model."):
                sanitized["model." + k[len("text.model.") :]] = v
                continue
            if k.startswith("text.lm_head."):
                sanitized["lm_head." + k[len("text.lm_head.") :]] = v
                continue
            # Raw moondream2 layout (vikhyatk/moondream2) — the same remap
            # mlx-vlm's moondream2.sanitize performs.
            if k.startswith(("vision_encoder.", "region_model.")):
                continue
            if k == "text_model.transformer.embd.wte.weight":
                sanitized["model.embed_tokens.weight"] = v
            elif k.startswith("text_model.transformer.h."):
                nk = "model.layers." + k[len("text_model.transformer.h.") :]
                nk = nk.replace(".mixer.Wqkv.", ".attn.qkv.")
                nk = nk.replace(".mixer.out_proj.", ".attn.proj.")
                sanitized[nk] = v
            elif k.startswith("text_model.lm_head.ln."):
                sanitized["model.post_ln." + k[len("text_model.lm_head.ln.") :]] = v
            elif k.startswith("text_model.lm_head.linear."):
                sanitized["lm_head." + k[len("text_model.lm_head.linear.") :]] = v
            else:
                sanitized[k] = v
        return sanitized

    @property
    def layers(self):
        return self.model.layers
