# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    num_attention_heads: int
    n_group: int
    head_dim: int
    topk_group: int
    n_shared_experts: int
    n_routed_experts: int
    routed_scaling_factor: float
    num_experts_per_tok: int
    first_k_dense_replace: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[Dict]
    use_qk_norm: bool
    tie_word_embeddings: bool
    attention_bias: bool
    partial_rotary_factor: float
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"

    @classmethod
    def from_dict(cls, params):
        # A GLM-4.5V (glm4v_moe) checkpoint nests the language model under
        # text_config, with a few fields (e.g. tie_word_embeddings) only at the
        # top level — flatten it; the MoE body matches glm4_moe field-for-field.
        if "text_config" in params:
            params = {**params, **params["text_config"]}
        return super().from_dict(params)


def _rotate_half(x):
    mid = x.shape[-1] // 2
    return mx.concatenate([-x[..., mid:], x[..., :mid]], axis=-1)


def _apply_mrope(x, cos, sin):
    """Rotate the first cos.shape[-1] dims with half-split pairing (GLM-4.5V style)."""
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return mx.concatenate([x_rot, x[..., rotary_dim:]], axis=-1)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            int(head_dim * args.partial_rotary_factor),
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if position_embeddings is not None:
            # Vision: 3D multimodal RoPE over the same attention weights.
            cos, sin = position_embeddings
            queries = _apply_mrope(queries, cos, sin)
            keys = _apply_mrope(keys, cos, sin)
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
    def __init__(
        self, config: ModelArgs, hidden_size: int = None, intermediate_size: int = None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x):
        down_proj = self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
        return down_proj


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):

    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / denominator
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))
        assert config.topk_method == "noaux_tc", "Unsupported topk method."

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MLP(
                config=config, intermediate_size=intermediate_size
            )

        self.sharding_group = None

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = (
            MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
            )
            else MLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache, position_embeddings)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LanguageModel(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Multimodal RoPE side state (mlx-unified), set by the vision path before
        # generation and cleared afterwards — same contract as qwen3_5.py. Underscore
        # attrs so MLX's module walker never registers them as parameters.
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        self._inv_freq = 1.0 / (
            config.rope_theta
            ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim)
        )
        # Which mrope axis (t/h/w) feeds each frequency: chunked by mrope_section
        # (GLM-4.5V: [8, 12, 12]). None for text-only GLM-4.5 checkpoints.
        mrope_section = (config.rope_scaling or {}).get("mrope_section")
        self._axis_selector = (
            mx.array([axis for axis, n in enumerate(mrope_section) for _ in range(n)])
            if mrope_section
            else None
        )
        self._mm_position_ids = None
        self._mm_rope_deltas = None

    def set_mrope_state(self, position_ids: mx.array, rope_deltas: mx.array) -> None:
        """Install 3D multimodal positions (3, B, L) + rope delta for a vision prompt."""
        if self._axis_selector is None:
            raise ValueError(
                "this checkpoint's rope_scaling has no mrope_section — it cannot "
                "consume multimodal positions"
            )
        self._mm_position_ids = position_ids
        self._mm_rope_deltas = rope_deltas

    def reset_mrope_state(self) -> None:
        self._mm_position_ids = None
        self._mm_rope_deltas = None

    def _compute_position_ids(self, inputs: mx.array, cache) -> Optional[mx.array]:
        """Positions for the current forward pass from stored multimodal state.

        None (the common text-only case) routes every layer to the original
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

        cache_offset = 0
        cache_offset_scalar = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0].offset
            if isinstance(offset, int):
                cache_offset = offset
                cache_offset_scalar = offset
            elif isinstance(offset, mx.array) and offset.ndim == 0:
                cache_offset = offset.item()
                cache_offset_scalar = cache_offset
            elif isinstance(offset, mx.array):
                cache_offset = offset
                cache_offset_scalar = offset[0].item()

        batch_size, seq_length = inputs.shape

        if self._mm_position_ids is not None:
            stored_seq_length = self._mm_position_ids.shape[2]
            if cache_offset_scalar < stored_seq_length:
                stored_end = min(cache_offset_scalar + seq_length, stored_seq_length)
                stored_positions = self._mm_position_ids[
                    :, :, cache_offset_scalar:stored_end
                ]
                if stored_end - cache_offset_scalar == seq_length:
                    return stored_positions
                tail_positions = self._sequential_position_ids(
                    batch_size=batch_size,
                    seq_length=seq_length - (stored_end - cache_offset_scalar),
                    start_offset=stored_seq_length,
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

    def _position_embeddings(self, position_ids: mx.array, dtype) -> tuple:
        """cos/sin (B, 1, L, rotary_dim) in half-split layout from (3, B, L) positions."""
        positions = mx.take(position_ids, self._axis_selector, axis=0)
        angle = positions.transpose(1, 2, 0).astype(mx.float32) * self._inv_freq
        cos = mx.tile(mx.cos(angle)[:, None], (1, 1, 1, 2))
        sin = mx.tile(mx.sin(angle)[:, None], (1, 1, 1, 2))
        return cos.astype(dtype), sin.astype(dtype)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)
        mask = create_attention_mask(h, cache[0])

        # None for text-only requests — layers take the original 1D-RoPE path.
        position_ids = self._compute_position_ids(x, cache)
        position_embeddings = None
        if position_ids is not None:
            position_embeddings = self._position_embeddings(position_ids, h.dtype)

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for l, c in zip(self.pipeline_layers, cache):
            h = l(h, mask, cache=c, position_embeddings=position_embeddings)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = LanguageModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        return self.lm_head(out)

    def sanitize(self, weights):
        # A GLM-4.5V (glm4v_moe) checkpoint carries a vision tower and prefixes the
        # language model — strip/remap down to this text-only structure first.
        def to_text_key(k):
            if k.startswith("model.language_model."):
                return "model." + k[len("model.language_model.") :]
            if k.startswith("language_model."):
                return k[len("language_model.") :]
            return k

        weights = {
            to_text_key(k): v
            for k, v in weights.items()
            if not k.startswith(("model.visual", "visual", "vision_tower"))
        }

        mpt_layer = self.args.num_hidden_layers

        # Stack experts
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        # Remove multi-token prediction layer
        return {
            k: v
            for k, v in weights.items()
            if not k.startswith(f"model.layers.{mpt_layer}")
        }

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            # Shard the self attention
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.n_heads //= N
            layer.self_attn.n_kv_heads //= N

            # Shard the MLP
            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # Shard the MoE. Shard in place since the MoE should be responsible
            # for aggregating the results.
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_experts.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_experts.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_experts.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate
