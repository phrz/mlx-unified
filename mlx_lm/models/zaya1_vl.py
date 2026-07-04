# Copyright © 2026 Apple Inc.
#
# Zyphra Zaya-1 text side, ported from mlx-vlm's models/zaya1_vl/language.py
# (MIT © Blaizzy / mlx-vlm contributors): CCA "Compressed Convolutional
# Attention" (depthwise causal-conv-mixed packed q/k with a qk-mean shortcut and
# per-kv-head norm + learned temperature scaling), a Mixture-of-Depths MoE router
# (an extra virtual expert that skips the MLP) with EDA router state passed
# layer-to-layer, and learned residual scaling around every sublayer.
#
# mlx-unified vision integration: visual_pos_masks side state
# (set_visual_state/reset_visual_state on the inner model) gates the checkpoint's
# vision-LoRA deltas (lora_linear_q/k, lora_val_proj1/2, lora_linear_o,
# lora_fc1/fc2) inside every attention and MLP sublayer during prefill. Without
# state the LoRA branches are never evaluated, so text-only behavior is
# byte-identical.

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import ArraysCache, CacheList, KVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchLinear


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "zaya1_vl"
    vocab_size: int = 262272
    hidden_size: int = 2048
    ffn_hidden_size: int = 4096
    num_hidden_layers: int = 40
    num_experts: int = 16
    num_attention_heads: int = 8
    num_key_value_heads: Optional[int] = None
    num_query_groups: Optional[int] = None
    head_dim: int = 128
    max_position_embeddings: int = 32768
    norm_epsilon: float = 1e-5
    attention_bias: bool = False
    lm_head_bias: bool = False
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    activation_func: str = "swiglu"
    moe_router_topk: int = 1
    zaya_mlp_expansion: int = 256
    zaya_use_mod: bool = True
    zaya_use_eda: bool = True
    scale_residual_merge: bool = True
    rope_theta: float = 1000000.0
    rotary_base: Optional[float] = None
    partial_rotary_factor: float = 0.5
    rope_pct: Optional[float] = None
    rope_scaling: Optional[dict] = None
    rope_parameters: Optional[dict] = None
    cca_time0: int = 2
    cca_time1: int = 2
    tie_word_embeddings: bool = True
    vision_lora: bool = True
    vision_lora_rank_attn: int = 8
    vision_lora_rank_mlp: int = 32

    @classmethod
    def from_dict(cls, params):
        # zaya1_vl checkpoints keep the text fields flat at the top level (mlx-vlm's
        # convention for this arch); tolerate a nested text_config too.
        if "text_config" in params:
            params = {**params, **params["text_config"]}
        return super().from_dict(params)

    def __post_init__(self):
        # Megatron-style aliases used by the released config.
        if self.rotary_base is not None:
            self.rope_theta = self.rotary_base
        if self.rope_pct is not None:
            self.partial_rotary_factor = self.rope_pct
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_query_groups or self.num_attention_heads

        rope_parameters = dict(self.rope_parameters or self.rope_scaling or {})
        if "type" in rope_parameters and "rope_type" not in rope_parameters:
            rope_parameters["rope_type"] = rope_parameters.pop("type")
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters.setdefault("rope_theta", self.rope_theta)
        rope_parameters.setdefault("partial_rotary_factor", self.partial_rotary_factor)
        self.rope_parameters = rope_parameters


def _split_cache(cache):
    """Per-layer cache is CacheList(KVCache, ArraysCache(2)) — see make_cache."""
    if cache is None:
        return None, None
    return cache[0], cache[1]


def _cache_is_empty(cache) -> bool:
    return cache is None or (hasattr(cache, "empty") and cache.empty())


def _kv_offset(kv_cache) -> int:
    if kv_cache is None:
        return 0
    offset = kv_cache.offset
    if isinstance(offset, mx.array):
        return int(offset.item()) if offset.ndim == 0 else int(offset[0].item())
    return offset


def _cache_position_mask(hidden_states: mx.array, cache) -> Optional[mx.array]:
    """(B, L) validity mask for batched left-padded prompts — padded positions are
    zeroed before the causal conv so they can't smear into real tokens."""
    left_padding = getattr(cache, "left_padding", None)
    if left_padding is None:
        return None
    positions = mx.arange(hidden_states.shape[1]) + (cache.size() if hasattr(cache, "size") else 0)
    return positions[None, :] >= left_padding[:, None]


def _causal_conv1d_stack(conv_layers, x: mx.array, state, state_size: int, use_state: bool):
    """Apply the stacked causal convs to (B, L, C) input, threading the last
    `state_size` inputs across forwards so chunked prefill and decode see the
    same left context as a single full-prompt pass."""
    if use_state:
        if state is None or state.shape[0] != x.shape[0] or state.shape[1] != state_size:
            state = mx.zeros((x.shape[0], state_size, x.shape[-1]), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        state_source = conv_input
    else:
        conv_input = mx.pad(x, ((0, 0), (state_size, 0), (0, 0)))
        state_source = x

    y = conv_input
    for conv in conv_layers:
        y = conv(y)

    if state_size == 0:
        next_state = mx.zeros((x.shape[0], 0, x.shape[-1]), dtype=x.dtype)
    else:
        if state_source.shape[1] < state_size:
            state_source = mx.pad(
                state_source, ((0, 0), (state_size - state_source.shape[1], 0), (0, 0))
            )
        next_state = mx.contiguous(state_source[:, -state_size:, :])

    return y, next_state


class ResidualScaling(nn.Module):
    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.not_first_layer = layer_n != 0
        self.hidden_states_scale = mx.ones((args.hidden_size,))
        self.hidden_states_bias = mx.zeros((args.hidden_size,))
        if self.not_first_layer:
            self.residual_scale = mx.ones((args.hidden_size,))
            self.residual_bias = mx.zeros((args.hidden_size,))

    def __call__(self, residual: Optional[mx.array], hidden_states: mx.array):
        hidden_states = (hidden_states + self.hidden_states_bias) * self.hidden_states_scale
        if self.not_first_layer and residual is not None:
            residual = (residual + self.residual_bias) * self.residual_scale
        return residual, hidden_states


class CCA(nn.Module):
    """Compressed Convolutional Attention: packed q/k are mixed by two causal
    depthwise/grouped convolutions along time, added back to a q/k-mean shortcut,
    then norm-and-temperature scaled. Values concatenate a projection of the
    current token with one of the PREVIOUS token (hs_d)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        # Both convs are kernel-2 in the released model; the state carries the
        # combined receptive-field overhang across forwards.
        self.total_padding = args.cca_time0 + args.cca_time1 - 2

        self.num_kv_heads = args.num_key_value_heads
        self.num_q_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.latent_k_dim = self.num_kv_heads * self.head_dim
        self.latent_q_dim = self.num_q_heads * self.head_dim
        self.gqa_groups = self.num_q_heads // self.num_kv_heads
        self.sqrt_head_dim = math.sqrt(self.head_dim)

        self.linear_q = nn.Linear(self.hidden_size, self.latent_q_dim, bias=args.attention_bias)
        self.linear_k = nn.Linear(self.hidden_size, self.latent_k_dim, bias=args.attention_bias)
        self.val_proj1 = nn.Linear(
            self.hidden_size, self.latent_k_dim // 2, bias=args.attention_bias
        )
        self.val_proj2 = nn.Linear(
            self.hidden_size, self.latent_k_dim // 2, bias=args.attention_bias
        )

        if args.vision_lora:
            r = args.vision_lora_rank_attn
            self.lora_linear_q = [
                nn.Linear(self.hidden_size, r, bias=False),
                nn.Linear(r, self.latent_q_dim, bias=False),
            ]
            self.lora_linear_k = [
                nn.Linear(self.hidden_size, r, bias=False),
                nn.Linear(r, self.latent_k_dim, bias=False),
            ]
            self.lora_val_proj1 = [
                nn.Linear(self.hidden_size, r, bias=False),
                nn.Linear(r, self.latent_k_dim // 2, bias=False),
            ]
            self.lora_val_proj2 = [
                nn.Linear(self.hidden_size, r, bias=False),
                nn.Linear(r, self.latent_k_dim // 2, bias=False),
            ]

        in_out_ch = self.latent_k_dim + self.latent_q_dim
        self.conv_qk = [
            nn.Conv1d(in_out_ch, in_out_ch, kernel_size=args.cca_time0, groups=in_out_ch),
            nn.Conv1d(
                in_out_ch,
                in_out_ch,
                kernel_size=args.cca_time1,
                groups=self.num_kv_heads + self.num_q_heads,
            ),
        ]
        self.temp = mx.zeros((self.num_kv_heads,))

    @staticmethod
    def _apply_lora(layers, x):
        return layers[1](layers[0](x))

    def _conv(self, qk_packed0: mx.array, aux_cache, kv_cache):
        x = qk_packed0.transpose(1, 0, 2)
        state = aux_cache[0] if aux_cache is not None else None
        y, state = _causal_conv1d_stack(
            self.conv_qk,
            x,
            state,
            self.total_padding,
            use_state=aux_cache is not None and not _cache_is_empty(kv_cache),
        )
        if aux_cache is not None:
            aux_cache[0] = state
        return y.transpose(1, 0, 2)

    def __call__(
        self,
        hidden_states: mx.array,
        cache=None,
        cca_mask: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
    ):
        kv_cache, aux_cache = _split_cache(cache)

        if cca_mask is not None and hidden_states.shape[1] > 1:
            hidden_states = hidden_states * cca_mask[..., None].astype(hidden_states.dtype)

        # (L, B, D) layout throughout; hs_d is the one-token-delayed stream for v2.
        hs = hidden_states.transpose(1, 0, 2)
        if hs.shape[0] > 1:
            hs_d = mx.concatenate([mx.zeros_like(hs[:1]), hs[:-1]], axis=0)
        else:
            hs_d = mx.zeros_like(hs)

        q = self.linear_q(hs)
        k = self.linear_k(hs)
        lora_mask = None
        if self.args.vision_lora and image_mask is not None:
            lora_mask = image_mask.transpose(1, 0)[..., None].astype(q.dtype)
            q = q + self._apply_lora(self.lora_linear_q, hs) * lora_mask
            k = k + self._apply_lora(self.lora_linear_k, hs) * lora_mask

        query_pre = q.reshape(*q.shape[:2], self.num_q_heads, self.head_dim)
        key_pre = k.reshape(*k.shape[:2], self.num_kv_heads, self.head_dim)
        key_pre = mx.repeat(key_pre, self.gqa_groups, axis=2)
        qk_mean_q = (query_pre + key_pre) / 2
        qk_mean_k = qk_mean_q.reshape(
            *qk_mean_q.shape[:2], self.num_kv_heads, self.gqa_groups, self.head_dim
        ).mean(axis=3)

        qk_packed0 = mx.concatenate([q, k], axis=-1)
        qk_packed3 = self._conv(qk_packed0, aux_cache, kv_cache)

        query = (
            qk_packed3[..., : self.latent_q_dim].reshape(
                *qk_packed3.shape[:2], self.num_q_heads, self.head_dim
            )
            + qk_mean_q
        )
        key = (
            qk_packed3[..., self.latent_q_dim :].reshape(
                *qk_packed3.shape[:2], self.num_kv_heads, self.head_dim
            )
            + qk_mean_k
        )

        v1 = self.val_proj1(hs)
        if self.args.vision_lora and image_mask is not None:
            v1 = v1 + self._apply_lora(self.lora_val_proj1, hs) * lora_mask

        # Continue hs_d across forwards: aux_cache[1] holds the previous forward's
        # last hidden state.
        if aux_cache is not None and not _cache_is_empty(kv_cache) and aux_cache[1] is not None:
            hs_d = mx.concatenate([aux_cache[1][None, ...], hs[:-1]], axis=0)
        if aux_cache is not None:
            aux_cache[1] = hs[-1]

        v2 = self.val_proj2(hs_d)
        if self.args.vision_lora and image_mask is not None:
            v2 = v2 + self._apply_lora(self.lora_val_proj2, hs_d) * lora_mask

        value = mx.concatenate([v1, v2], axis=-1).reshape(
            *hs.shape[:2], self.num_kv_heads, self.head_dim
        )

        norm_eps = mx.finfo(query.dtype).eps
        query_norm = mx.maximum(mx.sqrt(mx.sum(query * query, axis=-1, keepdims=True)), norm_eps)
        key_norm = mx.maximum(mx.sqrt(mx.sum(key * key, axis=-1, keepdims=True)), norm_eps)
        query = query * (self.sqrt_head_dim / query_norm)
        key = key * (self.sqrt_head_dim / key_norm) * self.temp[None, None, :, None]

        query = query.reshape(*query.shape[:2], self.num_q_heads * self.head_dim)
        key = key.reshape(*key.shape[:2], self.num_kv_heads * self.head_dim)
        value = value.reshape(*value.shape[:2], self.num_kv_heads * self.head_dim)
        return (
            query.transpose(1, 0, 2),
            key.transpose(1, 0, 2),
            value.transpose(1, 0, 2),
        )


class ZayaAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )
        self.qkv = CCA(args)
        self.rope = initialize_rope(
            int(args.head_dim * args.rope_parameters["partial_rotary_factor"]),
            base=args.rope_parameters["rope_theta"],
            traditional=False,
            scaling_config=args.rope_parameters,
            max_position_embeddings=args.max_position_embeddings,
        )

        if args.vision_lora:
            r = args.vision_lora_rank_attn
            self.lora_linear_o = [
                nn.Linear(self.num_attention_heads * self.head_dim, r, bias=False),
                nn.Linear(r, args.hidden_size, bias=False),
            ]

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cca_mask: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
        cache=None,
    ):
        B, L, _ = hidden_states.shape
        kv_cache, _ = _split_cache(cache)
        q, k, v = self.qkv(hidden_states, cache, cca_mask, image_mask)

        q = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = kv_cache.offset if kv_cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)

        if isinstance(mask, mx.array):
            mask = mask[..., : k.shape[-2]]

        out = scaled_dot_product_attention(q, k, v, cache=kv_cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)

        projected = self.o_proj(out)
        if self.args.vision_lora and image_mask is not None:
            addon = self.lora_linear_o[1](self.lora_linear_o[0](out))
            projected = projected + addon * image_mask[..., None].astype(projected.dtype)
        return projected


class ZayaRouter(nn.Module):
    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.use_mod = args.zaya_use_mod
        self.num_local_experts = args.num_experts
        # Mixture-of-Depths: one extra virtual expert whose selection skips the MLP.
        self.num_experts = args.num_experts + (1 if self.use_mod else 0)
        self.topk = args.moe_router_topk
        self.use_eda = args.zaya_use_eda and layer_n != 0

        self.down_proj = nn.Linear(args.hidden_size, args.zaya_mlp_expansion, bias=True)
        self.rmsnorm_eda = nn.RMSNorm(args.zaya_mlp_expansion, eps=args.norm_epsilon)
        if self.use_eda:
            self.router_states_scale = mx.ones((args.zaya_mlp_expansion,))

        self.router_mlp = [
            nn.Linear(args.zaya_mlp_expansion, args.zaya_mlp_expansion, bias=True),
            nn.GELU(),
            nn.Linear(args.zaya_mlp_expansion, args.zaya_mlp_expansion, bias=True),
            nn.GELU(),
            nn.Linear(args.zaya_mlp_expansion, self.num_experts, bias=False),
        ]
        self.balancing_biases = mx.zeros((self.num_experts,), dtype=mx.float32)
        if self.use_mod:
            self.balancing_biases[-1] = -1.0

    def __call__(self, hidden_states: mx.array, router_states: Optional[mx.array] = None):
        hs = self.down_proj(hidden_states)
        # EDA: the previous layer's pre-norm router activation feeds forward.
        if self.use_eda and router_states is not None:
            hs = hs + router_states * self.router_states_scale
        next_router_states = hs
        hs = self.rmsnorm_eda(hs)
        for layer in self.router_mlp:
            hs = layer(hs)

        expert_prob = mx.softmax(hs.astype(mx.float32), axis=-1).astype(hidden_states.dtype)
        biased = expert_prob.astype(mx.float32) + self.balancing_biases
        if self.topk == 1:
            expert_choice = mx.expand_dims(mx.argmax(biased, axis=-1), axis=-1)
        else:
            expert_choice = mx.argpartition(biased, kth=-self.topk, axis=-1)[..., -self.topk :]
        route_prob = mx.take_along_axis(expert_prob, expert_choice, axis=-1)
        return route_prob, expert_choice, next_router_states


class ZayaSwitchMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_experts = args.num_experts
        self.ffn_hidden_size_out = (
            args.ffn_hidden_size // 2 if args.gated_linear_unit else args.ffn_hidden_size
        )
        self.linear_fc1 = SwitchLinear(
            args.hidden_size, args.ffn_hidden_size, args.num_experts, bias=args.add_bias_linear
        )
        self.linear_fc2 = SwitchLinear(
            self.ffn_hidden_size_out, args.hidden_size, args.num_experts, bias=args.add_bias_linear
        )

        if args.vision_lora:
            r = args.vision_lora_rank_mlp
            self.lora_fc1 = [
                SwitchLinear(args.hidden_size, r, args.num_experts, bias=False),
                SwitchLinear(r, args.ffn_hidden_size, args.num_experts, bias=False),
            ]
            self.lora_fc2 = [
                SwitchLinear(self.ffn_hidden_size_out, r, args.num_experts, bias=False),
                SwitchLinear(r, args.hidden_size, args.num_experts, bias=False),
            ]

    def __call__(
        self,
        hidden_states: mx.array,
        expert_choice: mx.array,
        route_prob: mx.array,
        image_mask: Optional[mx.array] = None,
    ):
        # The virtual MoD expert (index num_experts) is clamped to a real expert for
        # the gather, then its output is replaced by the residual stream below.
        skip_mask = expert_choice == self.num_experts
        expert_indices = mx.minimum(expert_choice, self.num_experts - 1)

        routed_hidden_states = hidden_states[..., None, None, :]
        x = self.linear_fc1(routed_hidden_states, expert_indices, sorted_indices=False).squeeze(-2)
        if self.args.vision_lora and image_mask is not None:
            addon = self.lora_fc1[0](
                routed_hidden_states, expert_indices, sorted_indices=False
            ).squeeze(-2)
            addon = self.lora_fc1[1](
                addon[..., None, :], expert_indices, sorted_indices=False
            ).squeeze(-2)
            x = x + addon * image_mask[..., None, None].astype(x.dtype)

        if self.args.gated_linear_unit:
            x1, x2 = mx.split(x, 2, axis=-1)
            x = nn.silu(x1) * x2
        elif self.args.activation_func == "gelu":
            x = nn.gelu(x)
        else:
            x = nn.silu(x)

        y = self.linear_fc2(x[..., None, :], expert_indices, sorted_indices=False).squeeze(-2)
        if self.args.vision_lora and image_mask is not None:
            addon = self.lora_fc2[0](x[..., None, :], expert_indices, sorted_indices=False).squeeze(
                -2
            )
            addon = self.lora_fc2[1](
                addon[..., None, :], expert_indices, sorted_indices=False
            ).squeeze(-2)
            y = y + addon * image_mask[..., None, None].astype(y.dtype)

        if self.args.zaya_use_mod:
            y = mx.where(skip_mask[..., None], hidden_states[..., None, :], y)

        y = y * route_prob[..., None]
        return y.sum(axis=-2)


class ZayaBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.router = ZayaRouter(args, layer_n)
        self.experts = ZayaSwitchMLP(args)

    def __call__(
        self,
        hidden_states: mx.array,
        prev_router_hidden_states: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
    ):
        route_prob, expert_choice, prev_router_hidden_states = self.router(
            hidden_states, prev_router_hidden_states
        )
        output = self.experts(hidden_states, expert_choice, route_prob, image_mask)
        return output, prev_router_hidden_states


class ZayaDecoderATTLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.args = args
        self.self_attn = ZayaAttention(args)
        self.input_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, 2 * layer_n)

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        mask: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
        cache=None,
        cca_mask: Optional[mx.array] = None,
    ):
        if self.args.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.input_norm(residual)
        hidden_states = self.self_attn(hidden_states, mask, cca_mask, image_mask, cache)
        return hidden_states, residual


class ZayaDecoderMLPLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.args = args
        self.zaya_block = ZayaBlock(args, layer_n)
        self.input_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, 2 * layer_n + 1)

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        image_mask: Optional[mx.array] = None,
        prev_router_hidden_states: Optional[mx.array] = None,
    ):
        if self.args.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.input_norm(residual)
        hidden_states, prev_router_hidden_states = self.zaya_block(
            hidden_states, prev_router_hidden_states, image_mask
        )
        return hidden_states, residual, prev_router_hidden_states


class ZayaDecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_n: int):
        super().__init__()
        self.attn = ZayaDecoderATTLayer(args, layer_n)
        self.mlp = ZayaDecoderMLPLayer(args, layer_n)

    def __call__(
        self,
        hidden_states: mx.array,
        residual: Optional[mx.array],
        mask: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
        cache=None,
        prev_router_hidden_states: Optional[mx.array] = None,
        cca_mask: Optional[mx.array] = None,
    ):
        hidden_states, residual = self.attn(
            hidden_states,
            residual,
            mask=mask,
            image_mask=image_mask,
            cache=cache,
            cca_mask=cca_mask,
        )
        hidden_states, residual, prev_router_hidden_states = self.mlp(
            hidden_states,
            residual,
            image_mask=image_mask,
            prev_router_hidden_states=prev_router_hidden_states,
        )
        return hidden_states, residual, prev_router_hidden_states


class ZayaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ZayaDecoderBlock(args, layer_n=i) for i in range(args.num_hidden_layers)]
        if args.scale_residual_merge:
            self.res_scale = ResidualScaling(args, args.num_hidden_layers)
        self.final_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_epsilon)

        # Multimodal side state (mlx-unified), set by the vision path before
        # generation and cleared afterwards. Underscore attrs — never parameters.
        self._visual_pos_masks = None  # (1, L) bool over the prompt chunk being prefilled
        self._visual_anchor = None  # KV offset at first consumption, for chunk alignment

    def set_visual_state(self, visual_pos_masks=None) -> None:
        self._visual_pos_masks = visual_pos_masks
        self._visual_anchor = None

    def reset_visual_state(self) -> None:
        self._visual_pos_masks = None
        self._visual_anchor = None

    def _image_mask(self, seq_length: int, offset: int) -> Optional[mx.array]:
        """The chunk of visual_pos_masks aligned with this forward. The mask is
        anchored to the KV offset at first consumption (a prompt-cache hit means
        prefill starts mid-cache with an already-trimmed mask); positions past its
        end — decode steps — get no mask, so the LoRA path is prefill-only."""
        masks = self._visual_pos_masks
        if masks is None:
            return None
        if self._visual_anchor is None:
            self._visual_anchor = offset
        start = offset - self._visual_anchor
        if start >= masks.shape[1]:
            return None
        chunk = masks[:, start : start + seq_length]
        if chunk.shape[1] < seq_length:
            chunk = mx.pad(chunk, ((0, 0), (0, seq_length - chunk.shape[1])))
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

        first_kv_cache, _ = _split_cache(cache[0])
        attn_mask = create_attention_mask(h, first_kv_cache)
        cca_mask = _cache_position_mask(h, first_kv_cache)
        image_mask = self._image_mask(h.shape[1], _kv_offset(first_kv_cache))

        residual = None
        prev_router_hidden_states = None
        for layer, layer_cache in zip(self.layers, cache):
            h, residual, prev_router_hidden_states = layer(
                h,
                residual,
                mask=attn_mask,
                image_mask=image_mask,
                cache=layer_cache,
                prev_router_hidden_states=prev_router_hidden_states,
                cca_mask=cca_mask,
            )

        if self.args.scale_residual_merge:
            residual, h = self.res_scale(residual, h)
        residual = h if residual is None else h + residual
        return self.final_norm(residual)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ZayaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=args.lm_head_bias)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # Per layer: KVCache plus a 2-slot ArraysCache — [0] the causal-conv input
        # overhang, [1] the previous forward's last hidden state (for val_proj2).
        return [CacheList(KVCache(), ArraysCache(2)) for _ in self.layers]

    def sanitize(self, weights):
        # Accept both an mlx-vlm conversion (language_model.* prefix, experts
        # already stacked, mlx conv layout) and a raw checkpoint (model.* prefix,
        # per-expert local_experts, torch conv layout).
        out = {}
        for k, v in weights.items():
            if not k.startswith(("model.", "language_model.", "lm_head.")):
                continue
            k = k.removeprefix("language_model.")
            if k == "lm_head.weight" and self.args.tie_word_embeddings:
                continue
            out[k] = v

        for i in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{i}.mlp.zaya_block.experts"
            names = ["linear_fc1", "linear_fc2"]
            if self.args.vision_lora:
                names += ["lora_fc1.0", "lora_fc1.1", "lora_fc2.0", "lora_fc2.1"]
            for name in names:
                keys = [
                    f"{prefix}.local_experts.{e}.{name}.weight"
                    for e in range(self.args.num_experts)
                ]
                if keys[0] in out:
                    out[f"{prefix}.{name}.weight"] = mx.stack([out.pop(k) for k in keys], axis=0)

        # torch Conv1d stores (C_out, C_in/groups, kernel); mlx wants
        # (C_out, kernel, C_in/groups).
        kernels = (self.args.cca_time0, self.args.cca_time1)
        for k in list(out):
            if ".conv_qk." in k and k.endswith(".weight") and out[k].ndim == 3:
                kernel = kernels[int(k.rsplit(".", 2)[-2])]
                if out[k].shape[1] != kernel and out[k].shape[2] == kernel:
                    out[k] = out[k].transpose(0, 2, 1)
        return out
