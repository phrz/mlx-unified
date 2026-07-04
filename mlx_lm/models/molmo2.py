# Copyright © 2026 Apple Inc.
#
# AllenAI Molmo2 decoder (Qwen-derived: fused att_proj, per-head Q/K RMSNorm,
# standard 1D-RoPE GQA, split vocab table), ported from mlx-vlm's
# mlx_vlm/models/molmo2 (MIT © Blaizzy/mlx-vlm contributors).

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    # Multimodal checkpoints nest the decoder config under text_config
    # (model_type "molmo2_text"); text-only extractions carry it flat.
    model_type: str = "molmo2"
    text_config: Optional[dict] = None
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    # Multimodal special tokens live in an extension table appended to the base
    # vocab (wte.new_embedding); the lm_head covers only the base vocab.
    additional_vocab_size: int = 128
    layer_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 36864
    qkv_bias: bool = False

    def __post_init__(self):
        if self.text_config:
            for key, value in self.text_config.items():
                if key != "model_type" and key in self.__dataclass_fields__:
                    setattr(self, key, value)


class Molmo2Embedding(nn.Module):
    """Base vocab table plus the extension table, concatenated at lookup time —
    matches the checkpoint's wte.{embedding,new_embedding} split."""

    def __init__(self, num_embeddings: int, num_new_embeddings: int, dims: int):
        super().__init__()
        self.embedding = mx.zeros((num_embeddings, dims))
        self.new_embedding = mx.zeros((num_new_embeddings, dims))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.concatenate([self.embedding, self.new_embedding], axis=0)[x]


class Molmo2Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.fused_dims = (
            self.n_heads * head_dim,
            self.n_kv_heads * head_dim,
            self.n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            args.hidden_size, sum(self.fused_dims), bias=args.qkv_bias
        )
        self.q_norm = nn.RMSNorm(head_dim, eps=args.layer_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.layer_norm_eps)
        self.attn_out = nn.Linear(
            self.n_heads * head_dim, args.hidden_size, bias=False
        )
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
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.att_proj(x)
        q, k, v = mx.split(
            qkv, [self.fused_dims[0], self.fused_dims[0] + self.fused_dims[1]], axis=-1
        )
        q = self.q_norm(q.reshape(B, L, self.n_heads, self.head_dim))
        k = self.k_norm(k.reshape(B, L, self.n_kv_heads, self.head_dim))
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        return self.attn_out(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class Molmo2MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Fused gate+up projection; molmo's operand order is (value, gate).
        self.ff_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.ff_out = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        z, gate = mx.split(self.ff_proj(x), 2, axis=-1)
        return self.ff_out(nn.silu(gate) * z)


class Molmo2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Molmo2Attention(args)
        self.mlp = Molmo2MLP(args.hidden_size, args.intermediate_size)
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.ff_norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.attn_norm(x), mask, cache)
        return h + self.mlp(self.ff_norm(h))


class Molmo2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.wte = Molmo2Embedding(
            args.vocab_size, args.additional_vocab_size, args.hidden_size
        )
        self.blocks = [Molmo2Block(args) for _ in range(args.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.wte(inputs)

        if cache is None:
            cache = [None] * len(self.blocks)

        mask = create_attention_mask(h, cache[0])

        for block, c in zip(self.blocks, cache):
            h = block(h, mask, cache=c)

        return self.ln_f(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Molmo2Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.lm_head(self.model(inputs, cache, input_embeddings))

    def sanitize(self, weights):
        new_weights = {}
        for k, v in weights.items():
            # Raw HF checkpoints: model.transformer.* / model.vision_backbone.* /
            # top-level lm_head.* (already in place).
            if k.startswith("model.transformer."):
                k = "model." + k[len("model.transformer.") :]
            # mlx-vlm conversions: language_model.{model,lm_head}.* / vision_tower.*
            elif k.startswith("language_model."):
                k = k[len("language_model.") :]
            if k.startswith(("vision_tower.", "model.vision_backbone.")):
                continue
            new_weights[k] = v
        return new_weights

    @property
    def layers(self):
        return self.model.blocks
