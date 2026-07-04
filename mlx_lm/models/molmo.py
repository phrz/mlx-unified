# Copyright © 2026 Apple Inc.
#
# AllenAI Molmo's bespoke OLMo-variant decoder, ported from mlx-vlm's
# mlx_vlm/models/molmo (MIT © Blaizzy/mlx-vlm contributors).

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    # Molmo checkpoints ship a FLAT config (no text_config); defaults are Molmo-7B-D.
    model_type: str = "molmo"
    hidden_size: int = 3584
    # Fused SwiGLU width (the ff_proj output); the actual ffn hidden is half of it.
    intermediate_size: int = 37888
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    layer_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    vocab_size: int = 152064
    embedding_size: int = 152064
    # Multimodal special tokens live in a separately-initialized extension table
    # appended to the base vocab (wte.new_embedding); not in the checkpoint config.
    additional_vocab_size: int = 128
    qkv_bias: bool = True
    weight_tying: bool = False


class MolmoEmbedding(nn.Module):
    """Base vocab table plus the extension table, concatenated at lookup time —
    matches the checkpoint's wte.{embedding,new_embedding} split."""

    def __init__(self, num_embeddings: int, num_new_embeddings: int, dims: int):
        super().__init__()
        self.embedding = mx.zeros((num_embeddings, dims))
        self.new_embedding = mx.zeros((num_new_embeddings, dims))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.concatenate([self.embedding, self.new_embedding], axis=0)[x]


class MolmoBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = dim // args.num_attention_heads
        self.scale = head_dim**-0.5

        # Single fused projection: full-width Q, then K and V at kv-head width.
        self.fused_dims = (dim, self.n_kv_heads * head_dim, self.n_kv_heads * head_dim)
        self.att_proj = nn.Linear(dim, sum(self.fused_dims), bias=args.qkv_bias)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.ff_proj = nn.Linear(dim, args.intermediate_size, bias=False)
        self.ff_out = nn.Linear(args.intermediate_size // 2, dim, bias=False)
        self.attn_norm = nn.RMSNorm(dim, eps=args.layer_norm_eps)
        self.ff_norm = nn.RMSNorm(dim, eps=args.layer_norm_eps)
        self.rope = nn.RoPE(head_dim, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.att_proj(self.attn_norm(x))
        q, k, v = mx.split(
            qkv, [self.fused_dims[0], self.fused_dims[0] + self.fused_dims[1]], axis=-1
        )
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

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
        h = x + self.attn_out(out.transpose(0, 2, 1, 3).reshape(B, L, D))

        # SwiGLU with molmo's operand order: (value, gate) = split(ff_proj(x)).
        z = self.ff_proj(self.ff_norm(h))
        z, gate = mx.split(z, 2, axis=-1)
        return h + self.ff_out(nn.silu(gate) * z)


class MolmoModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.wte = MolmoEmbedding(
            args.embedding_size, args.additional_vocab_size, args.hidden_size
        )
        self.blocks = [MolmoBlock(args) for _ in range(args.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_eps)
        if not args.weight_tying:
            self.ff_out = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

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
        self.model = MolmoModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.weight_tying:
            return out @ self.model.wte.embedding.T
        return self.model.ff_out(out)

    def sanitize(self, weights):
        new_weights = {}
        for k, v in weights.items():
            # Raw HF checkpoints: model.transformer.* / model.vision_backbone.*
            if k.startswith("model.transformer."):
                k = "model." + k[len("model.transformer.") :]
            # mlx-vlm conversions: language_model.model.* / vision_tower.*
            elif k.startswith("language_model.model."):
                k = "model." + k[len("language_model.model.") :]
            if k.startswith(("vision_tower.", "model.vision_backbone.")):
                continue
            new_weights[k] = v
        return new_weights

    @property
    def layers(self):
        return self.model.blocks
