# Copyright © 2026 Apple Inc.
#
# moondream3 text model: a parallel-residual MoE decoder (dense below
# moe_start_layer, 64-expert gated-GELU MoE above) whose attention applies Tau —
# a learned position- and data-dependent multiplicative temperature on Q and V per
# head. Ported from Blaizzy/mlx-vlm's moondream3 language model (MIT ©
# Blaizzy/mlx-vlm contributors). The vision tower stays in mlx-vlm; vision prompts
# arrive as input_embeddings plus a bidirectional-prefix attention mask via
# set_visual_state (see mlx_lm/multimodal.py).

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .switch_layers import SwitchLinear, _gather_sort, _scatter_unsort


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "moondream3"
    text_config: Optional[dict] = None
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    vocab_size: int = 51200
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 64
    rope_theta: float = 1500000.0
    rope_dim: int = 32
    layer_norm_eps: float = 1e-5
    num_experts: int = 64
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1024
    moe_start_layer: int = 4
    attention_bias: bool = True

    def __post_init__(self):
        # Multimodal checkpoints nest the text fields under text_config; the
        # reference config calls the LayerNorm eps "rms_norm_eps".
        for k, v in (self.text_config or {}).items():
            if k == "rms_norm_eps":
                self.layer_norm_eps = v
            elif k != "model_type" and hasattr(self, k):
                setattr(self, k, v)


class Tau(nn.Module):
    """Learned position- and data-dependent temperature scaling for Q and V:
    per head, a tanh token term computed from the raw fused-QKV activations plus a
    sigmoid(alpha · log position) term, applied multiplicatively."""

    def __init__(self, n_heads: int, qkv_dim: int):
        super().__init__()
        self.wq = mx.zeros((n_heads, qkv_dim))
        self.wv = mx.zeros((n_heads, qkv_dim))
        self.alpha = mx.zeros((n_heads,))

    def __call__(self, qkv: mx.array, positions: mx.array) -> tuple:
        h = nn.gelu(qkv)
        tok_q = mx.tanh(h @ self.wq.T)
        tok_v = mx.tanh(h @ self.wv.T)

        log_pos = mx.log(positions + 1.0)
        alpha_log_pos = self.alpha[:, None] * log_pos[None, :]
        tau_pos = (1.0 + (mx.sigmoid(alpha_log_pos) - 0.5)).astype(qkv.dtype)

        tau_q = tok_q.transpose(0, 2, 1) + tau_pos[None]
        tau_v = tok_v.transpose(0, 2, 1) + tau_pos[None]
        return tau_q[..., None], tau_v[..., None]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        qkv_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.qkv = nn.Linear(dim, qkv_dim, bias=args.attention_bias)
        self.proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )
        self.tau = Tau(self.n_heads, qkv_dim)
        self.rope = nn.RoPE(args.rope_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.qkv(x)

        offset = cache.offset if cache is not None else 0
        positions = mx.arange(offset, offset + L)
        tau_q, tau_v = self.tau(qkv, positions)

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        queries, keys, values = mx.split(qkv, [q_dim, q_dim + kv_dim], axis=-1)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = queries * tau_q
        values = values * tau_v

        if cache is not None:
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.proj(output)


class DenseMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=True)
        self.fc2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class MoEMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        inner_dim = args.moe_intermediate_size
        num_experts = args.num_experts
        self.num_experts_per_tok = args.num_experts_per_tok

        self.router = nn.Linear(dim, num_experts, bias=True)
        # fc1 packs [hidden, gate]; the expert nonlinearity is gelu(h) * (g + 1).
        self.fc1 = SwitchLinear(dim, 2 * inner_dim, num_experts, bias=False)
        self.fc2 = SwitchLinear(inner_dim, dim, num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok

        gates = self.router(x)
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne - 1, axis=-1)[..., :ne])
        scores = mx.softmax(
            mx.take_along_axis(gates, inds, axis=-1), axis=-1, precise=True
        )

        x = mx.expand_dims(x, (-2, -3))

        do_sort = inds.size >= 64
        idx = inds
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, inds)

        h = self.fc1(x, idx, sorted_indices=do_sort)
        h1, g = mx.split(h, 2, axis=-1)
        h = nn.gelu(h1) * (g + 1.0)
        y = self.fc2(h, idx, sorted_indices=do_sort)

        if do_sort:
            y = _scatter_unsort(y, inv_order, inds.shape)

        y = y.squeeze(-2)
        return (y * scores[..., None]).sum(axis=-2)


class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.ln = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.attn = Attention(args)
        if layer_idx < args.moe_start_layer:
            self.mlp = DenseMLP(args)
        else:
            self.mlp = MoEMLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.ln(x)
        return x + self.attn(h, mask, cache) + self.mlp(h)


class Moondream3TextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.blocks = [
            DecoderBlock(args, idx) for idx in range(args.num_hidden_layers)
        ]
        self.post_ln = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

        # Multimodal side state (mlx-unified), set by the vision path before
        # generation and cleared afterwards. Underscore attr — never registered as
        # a parameter. (1, 1, L, L) additive mask over the FULL prompt: bos + image
        # tokens form a bidirectional prefix, text stays causal.
        self._attention_mask_4d = None

    @property
    def layers(self):
        return self.blocks

    @property
    def embed_tokens(self):
        return self.wte

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
            h = self.wte(inputs)

        if cache is None:
            cache = [None] * len(self.blocks)

        mask = self._make_mask(h, cache)
        for block, c in zip(self.blocks, cache):
            h = block(h, mask, c)
        return self.post_ln(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Moondream3TextModel(args)
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
            # Raw moondream3 layout nests everything under a "model." prefix.
            k = k.removeprefix("model.")
            if "position_ids" in k:
                continue
            if k.startswith(("vision.", "region.")):
                continue
            if k.startswith("text.model."):
                # mlx-vlm conversion layout — already module-shaped.
                sanitized["model." + k[len("text.model.") :]] = v
            elif k == "text.wte":
                # Raw layout stores the embedding table bare, without ".weight".
                sanitized["model.wte.weight"] = v
            elif k.startswith("text.lm_head."):
                sanitized["lm_head." + k[len("text.lm_head.") :]] = v
            elif k.startswith("text."):
                sanitized["model." + k[len("text.") :]] = v
            else:
                sanitized[k] = v
        return sanitized

    @property
    def layers(self):
        return self.model.blocks
