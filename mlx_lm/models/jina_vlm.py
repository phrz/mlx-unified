# Copyright © 2026 Apple Inc.
#
# Text-side decoder for Jina VLM (checkpoint model_type "jvlm"), ported from
# Blaizzy/mlx-vlm models/jina_vlm/language.py (MIT © Blaizzy/mlx-vlm contributors).
# Loads the language half of an mlx-vlm-converted checkpoint (sanitize strips the
# vision tower); vision is plain embedding injection via input_embeddings — the text
# forward semantics are unchanged by images, so no side-state hooks are needed.

from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "jvlm"
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    additional_vocab_size: int = 128
    intermediate_size: int = 6144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960
    use_qk_norm: bool = True

    @classmethod
    def from_dict(cls, params):
        # jvlm checkpoints nest the per-block settings under block_config.
        block = params.get("block_config", {})
        attn = block.get("attn_config", {})
        ffn = block.get("ffn_config", {})
        lnorm = block.get("lnorm_config", {})
        return cls(
            model_type=params.get("model_type", "jvlm"),
            hidden_size=params.get("hidden_size", 2048),
            num_hidden_layers=params.get(
                "n_layers", params.get("num_hidden_layers", 28)
            ),
            num_attention_heads=attn.get("n_heads", 16),
            num_key_value_heads=attn.get("n_kv_heads", 8),
            head_dim=attn.get("head_dim", 128),
            vocab_size=params.get("vocab_size", 151936),
            additional_vocab_size=params.get("additional_vocab_size", 128),
            intermediate_size=ffn.get("size", 6144),
            rms_norm_eps=lnorm.get("eps", 1e-6),
            rope_theta=params.get("rope_theta", 1000000.0),
            max_position_embeddings=params.get("max_sequence_length", 40960),
            use_qk_norm=attn.get("q_lnorm", True),
        )


class Attention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()

        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        # Single fused projection, split q/k/v at call time (checkpoint key attn.qkv).
        self.qkv = nn.Linear(
            args.hidden_size, (n_heads + 2 * n_kv_heads) * head_dim, bias=False
        )
        self.out = nn.Linear(n_heads * head_dim, args.hidden_size, bias=False)

        if args.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

        self.rope = initialize_rope(
            head_dim, args.rope_theta, False, None, args.max_position_embeddings
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        queries, keys, values = mx.split(
            self.qkv(x), [q_size, q_size + kv_size], axis=-1
        )

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)
        values = values.reshape(B, L, self.n_kv_heads, -1)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

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
        return self.out(output)


class MLP(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate_up = nn.Linear(
            args.hidden_size, 2 * args.intermediate_size, bias=False
        )
        self.down = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # jvlm packs the fused projection as [up, gate]: the value half FIRST and the
        # activated (gate) half SECOND — the reverse of the usual gate_up convention.
        up, gate = mx.split(self.gate_up(x), 2, axis=-1)
        return self.down(nn.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn = Attention(args)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn = MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.attn(self.attn_norm(x), mask=mask, cache=cache)
        return h + self.ffn(self.ffn_norm(h))


class ExtendedEmbedding(nn.Module):
    """Base vocabulary plus additional_vocab_size image-token rows — ids continue past
    vocab_size into new_embedding, stored separately to match the checkpoint keys
    embedding / new_embedding."""

    def __init__(self, vocab_size: int, additional_size: int, dims: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = mx.zeros((vocab_size, dims))
        self.new_embedding = mx.zeros((additional_size, dims))

    def __call__(self, x: mx.array) -> mx.array:
        # Two masked gathers + select rather than concatenating the tables: the concat
        # would materialize the full (vocab+extra, dims) buffer on every decode step.
        in_base = x < self.vocab_size
        base = self.embedding[mx.where(in_base, x, 0)]
        extra = self.new_embedding[mx.where(in_base, 0, x - self.vocab_size)]
        return mx.where(in_base[..., None], base, extra)


class LanguageModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        if args.additional_vocab_size > 0:
            self.embedding = ExtendedEmbedding(
                args.vocab_size, args.additional_vocab_size, args.hidden_size
            )
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # lm_head covers the base vocabulary only — image-token rows are never emitted.
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embedding(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)

        return self.lm_head(self.ln_f(h))


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "jvlm"
    text_config: dict = field(default_factory=dict)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = LanguageModel(TextArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if k.startswith(("vision_model.", "vl_connector.")):
                continue
            if k.startswith("lm_head."):
                # Some jvlm conversions keep lm_head at the top level (mlx-vlm's own
                # sanitize does the same move).
                k = "language_model." + k
            sanitized[k] = v
        return sanitized

    @property
    def model(self):
        # Uniform access to the inner text model (same shape as qwen3_5/gemma4) —
        # jvlm's vision contract is plain injection, so no side-state hooks live here.
        return self.language_model

    @property
    def layers(self):
        return self.language_model.layers
