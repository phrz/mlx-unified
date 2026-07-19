# Copyright © 2026 Apple Inc.
#
# MiniMax M3 decoder: a sparse MoE (sigmoid routing, routed scaling, one shared
# expert packed into the switch) with MiniMax's block-sparse attention — an
# indexer projects per-token index queries/keys and picks top-k key BLOCKS per
# query, so sparse layers carry an extra rotated index-key cache alongside KV.
# Ported from Blaizzy/mlx-vlm models/minimax_m3_vl/language.py
# (MIT © Blaizzy/mlx-vlm contributors).
#
# Not ported: the fused Metal sparse-attention kernel. Sparse layers use a
# pure-MLX gather over the selected blocks (decode) or a block mask over dense
# SDPA (prefill) — slower than the kernel but numerically exact.
#
# minimax_m3_vl checkpoints reuse this decoder unchanged: vision only injects
# input_embeddings, so `from_dict` flattens a nested text_config and
# `sanitize` strips the vision tower / remaps language_model.* prefixes.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU, SwitchLinear, _gather_sort, _scatter_unsort


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "minimax_m3"
    hidden_size: int = 6144
    intermediate_size: int = 3072
    dense_intermediate_size: int = 12288
    shared_intermediate_size: int = 3072
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: Optional[int] = 128
    num_hidden_layers: int = 60
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000
    rotary_dim: Optional[int] = None
    partial_rotary_factor: float = 0.5
    rope_scaling: Optional[Dict[str, Any]] = None
    max_position_embeddings: int = 1048576
    vocab_size: int = 200064
    tie_word_embeddings: bool = False
    swiglu_alpha: float = 1.702
    swiglu_beta: float = 1.0
    swiglu_limit: float = 7.0
    use_qk_norm: bool = True
    use_gemma_norm: bool = True
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    scoring_func: str = "sigmoid"
    use_routing_bias: bool = True
    routed_scaling_factor: float = 2.0
    moe_layer_freq: List[int] = field(default_factory=list)
    mlp_layer_types: Optional[List[str]] = None
    sparse_attention_config: Optional[Dict[str, Any]] = None
    layer_types: Optional[List[str]] = None
    index_n_heads: Optional[int] = None
    index_head_dim: Optional[int] = None
    index_block_size: Optional[int] = None
    index_topk_blocks: Optional[int] = None
    index_local_blocks: Optional[int] = None

    @classmethod
    def from_dict(cls, params):
        # minimax_m3_vl nests the decoder config under text_config.
        text_config = params.get("text_config")
        if isinstance(text_config, dict) and text_config:
            params = {**params, **text_config}
        return super().from_dict(params)

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rotary_dim is None:
            self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if isinstance(self.rope_scaling, dict) and "type" not in self.rope_scaling:
            self.rope_scaling = dict(self.rope_scaling)
            if "rope_type" in self.rope_scaling:
                self.rope_scaling["type"] = self.rope_scaling["rope_type"]
        if not self.moe_layer_freq:
            if self.mlp_layer_types is not None:
                self.moe_layer_freq = [
                    1 if layer_type == "sparse" else 0
                    for layer_type in self.mlp_layer_types
                ]
            else:
                self.moe_layer_freq = self._default_layer_frequency()
        sparse_freq = self._sparse_frequency_from_layer_types()
        if self.sparse_attention_config is None:
            if sparse_freq is None:
                sparse_freq = self._default_layer_frequency()
            self.sparse_attention_config = {
                "use_sparse_attention": True,
                "sparse_index_dim": (
                    self.index_head_dim if self.index_head_dim is not None else 128
                ),
                "sparse_num_index_heads": (
                    self.index_n_heads if self.index_n_heads is not None else 4
                ),
                "sparse_topk_blocks": (
                    self.index_topk_blocks if self.index_topk_blocks is not None else 16
                ),
                "sparse_block_size": (
                    self.index_block_size if self.index_block_size is not None else 128
                ),
                "sparse_score_type": "max",
                "sparse_init_block": 0,
                "sparse_local_block": (
                    self.index_local_blocks
                    if self.index_local_blocks is not None
                    else 1
                ),
                "sparse_attention_freq": sparse_freq,
            }
        else:
            self.sparse_attention_config = dict(self.sparse_attention_config)
            if (
                sparse_freq is not None
                and "sparse_attention_freq" not in self.sparse_attention_config
            ):
                self.sparse_attention_config["sparse_attention_freq"] = sparse_freq
            if sparse_freq is not None:
                self.sparse_attention_config.setdefault("use_sparse_attention", True)
            self._apply_sparse_index_aliases()
            # Older configs record the per-layer schedule under
            # sparse_disable_index_value instead of sparse_attention_freq.
            sparse_freq = self.sparse_attention_config.get("sparse_attention_freq")
            if sparse_freq is None and isinstance(
                self.sparse_attention_config.get("sparse_disable_index_value"), list
            ):
                sparse_freq = self.sparse_attention_config["sparse_disable_index_value"]
                self.sparse_attention_config["sparse_attention_freq"] = sparse_freq
                self.sparse_attention_config.setdefault("use_sparse_attention", True)

    def _default_layer_frequency(self) -> List[int]:
        dense_layers = min(3, self.num_hidden_layers)
        return [0] * dense_layers + [1] * (self.num_hidden_layers - dense_layers)

    def _sparse_frequency_from_layer_types(self) -> Optional[List[int]]:
        if self.layer_types is None:
            return None
        return [
            1 if layer_type == "minimax_m3_sparse" else 0
            for layer_type in self.layer_types
        ]

    def _apply_sparse_index_aliases(self):
        aliases = {
            "sparse_index_dim": self.index_head_dim,
            "sparse_num_index_heads": self.index_n_heads,
            "sparse_topk_blocks": self.index_topk_blocks,
            "sparse_block_size": self.index_block_size,
            "sparse_local_block": self.index_local_blocks,
        }
        for key, value in aliases.items():
            if value is not None and key not in self.sparse_attention_config:
                self.sparse_attention_config[key] = value

    def is_moe_layer(self, layer_idx: int) -> bool:
        if layer_idx >= len(self.moe_layer_freq):
            return True
        return bool(self.moe_layer_freq[layer_idx])

    def has_sparse_index(self, layer_idx: int) -> bool:
        if not self.sparse_attention_config.get("use_sparse_attention", False):
            return False
        freq = self.sparse_attention_config.get("sparse_attention_freq")
        if isinstance(freq, list) and layer_idx < len(freq):
            return bool(freq[layer_idx])
        return False


def _is_bool_mask(mask: mx.array) -> bool:
    return mask.dtype == mx.bool_


@mx.compile
def _minimax_moe_select(
    gates: mx.array,
    correction_bias: mx.array,
    k: int,
    routed_scaling_factor: float,
    scoring_func: str,
):
    if scoring_func == "sigmoid":
        scores = mx.sigmoid(gates)
    else:
        scores = mx.softmax(gates, axis=-1, precise=True)

    biased_scores = scores + correction_bias
    inds = mx.argpartition(-biased_scores, kth=k - 1, axis=-1)[..., :k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    return inds, weights * routed_scaling_factor


@mx.compile
def _select_sparse_block_indices_compiled(
    idx_queries: mx.array,
    idx_keys: mx.array,
    q_positions: mx.array,
    scale: float,
    block_size: int,
    sparse_topk_blocks: int,
    sparse_init_blocks: int,
    sparse_local_blocks: int,
):
    B, H_idx, L, _ = idx_queries.shape
    total_len = idx_keys.shape[2]
    neg = mx.array(-float("inf"), dtype=mx.float32)

    scores = mx.matmul(
        idx_queries.astype(mx.float32),
        idx_keys.astype(mx.float32).swapaxes(-1, -2),
    )
    scores = scores * scale

    qpos = q_positions
    kpos = mx.arange(total_len)
    causal = kpos[None, None, :] <= qpos[:, :, None]
    scores = mx.where(causal[:, None], scores, neg)

    num_blocks = (total_len + block_size - 1) // block_size
    pad = num_blocks * block_size - total_len
    pad_values = mx.full((*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype)
    scores = mx.concatenate([scores, pad_values], axis=-1)

    blocks = mx.arange(num_blocks)
    cur_block = qpos // block_size
    causal_block = blocks[None, None, :] <= cur_block[:, :, None]
    valid_blocks = causal_block[:, None]

    scores = scores.reshape(B, H_idx, L, num_blocks, block_size)
    block_scores = mx.max(scores, axis=-1)
    block_scores = mx.max(block_scores, axis=1)
    # NaN-scrub: a fully-masked block maxes to NaN through -inf arithmetic.
    selected_scores = mx.where(block_scores == block_scores, block_scores, neg)
    valid_blocks = mx.broadcast_to(valid_blocks[:, 0], selected_scores.shape)
    selected_scores = mx.where(valid_blocks, selected_scores, neg)

    if sparse_init_blocks > 0:
        init_blocks = blocks[None, None, :] < sparse_init_blocks
        selected_scores = mx.where(
            (init_blocks & causal_block) & valid_blocks,
            mx.array(1e30, dtype=selected_scores.dtype),
            selected_scores,
        )

    if sparse_local_blocks > 0:
        local_start = mx.maximum(cur_block - sparse_local_blocks + 1, 0)
        local_blocks = (blocks[None, None, :] >= local_start[:, :, None]) & (
            blocks[None, None, :] <= cur_block[:, :, None]
        )
        selected_scores = mx.where(
            (local_blocks & causal_block) & valid_blocks,
            mx.array(1e29, dtype=selected_scores.dtype),
            selected_scores,
        )

    topk_idx = mx.argpartition(-selected_scores, kth=sparse_topk_blocks - 1, axis=-1)[
        ..., :sparse_topk_blocks
    ]
    topk_valid = mx.take_along_axis(valid_blocks, topk_idx, axis=-1)
    invalid = mx.full(topk_idx.shape, num_blocks, dtype=topk_idx.dtype)
    block_indices = mx.where(topk_valid, topk_idx, invalid)
    order = mx.argsort(block_indices, axis=-1)
    block_indices = mx.take_along_axis(block_indices, order, axis=-1)
    return mx.where(block_indices == num_blocks, mx.array(-1), block_indices)


@mx.compile
def _swiglu_oai(x_linear, x_glu, alpha: float, limit: float, beta: float):
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
    return x_glu * mx.sigmoid(alpha * x_glu) * (x_linear + beta)


class MiniMaxSwiGLUOAI(nn.Module):
    def __init__(self, alpha: float, limit: float, beta: float):
        super().__init__()
        self.alpha = alpha
        self.limit = limit
        self.beta = beta

    def __call__(self, x, gate):
        return _swiglu_oai(x, gate, self.alpha, self.limit, self.beta)


class MiniMaxRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6, gemma: bool = True):
        super().__init__()
        self.weight = mx.zeros((dims,)) if gemma else mx.ones((dims,))
        self.eps = eps
        self.gemma = gemma

    def __call__(self, x):
        weight = self.weight + 1 if self.gemma else self.weight
        return mx.fast.rms_norm(x, weight, self.eps)


class MiniMaxM3KVCache:
    """A KVCache plus the indexer's own roped index-key cache for sparse
    layers. Trimming is supported: both offsets move back together, and the
    preallocated buffers are simply overwritten on the next update."""

    step = KVCache.step

    def __init__(self):
        self.kv_cache = KVCache()
        self.index_keys = None
        self.index_offset = 0

    @property
    def offset(self):
        return self.kv_cache.offset

    @offset.setter
    def offset(self, value):
        self.kv_cache.offset = int(value)
        self.index_offset = int(value)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        return self.kv_cache.update_and_fetch(keys, values)

    def update_index_and_fetch(self, keys: mx.array):
        prev = self.index_offset
        incoming = keys.shape[2]
        if self.index_keys is None or (prev + incoming) > self.index_keys.shape[2]:
            B, n_heads, _, head_dim = keys.shape
            n_steps = (self.step + incoming - 1) // self.step
            new_keys = mx.zeros((B, n_heads, n_steps * self.step, head_dim), keys.dtype)
            if self.index_keys is not None:
                if prev % self.step != 0:
                    self.index_keys = self.index_keys[..., :prev, :]
                self.index_keys = mx.concatenate([self.index_keys, new_keys], axis=2)
            else:
                self.index_keys = new_keys

        self.index_offset += incoming
        self.index_keys[..., prev : self.index_offset, :] = keys
        return self.index_keys[..., : self.index_offset, :]

    def make_mask(self, *args, **kwargs):
        return self.kv_cache.make_mask(*args, **kwargs)

    def size(self):
        return self.kv_cache.size()

    def empty(self):
        return self.kv_cache.empty()

    def is_trimmable(self):
        return True

    def trim(self, n):
        trimmed = self.kv_cache.trim(n)
        self.index_offset = max(0, self.index_offset - trimmed)
        return trimmed

    @property
    def state(self):
        kv_state = None if self.kv_cache.empty() else self.kv_cache.state
        index_state = (
            None
            if self.index_keys is None
            else self.index_keys[..., : self.index_offset, :]
        )
        return kv_state, index_state

    @state.setter
    def state(self, value):
        kv_state, index_state = value
        self.kv_cache = KVCache()
        if kv_state is not None:
            self.kv_cache.state = kv_state
        self.index_keys = index_state
        self.index_offset = 0 if index_state is None else index_state.shape[2]

    @property
    def meta_state(self):
        return str(self.index_offset)

    @meta_state.setter
    def meta_state(self, value):
        self.index_offset = int(value) if value else 0

    @property
    def nbytes(self):
        index_nbytes = 0 if self.index_keys is None else self.index_keys.nbytes
        return self.kv_cache.nbytes + index_nbytes


class MiniMaxMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        alpha: float,
        limit: float,
        beta: float,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = MiniMaxSwiGLUOAI(alpha, limit, beta)

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x), self.gate_proj(x)))


class MiniMaxPackedSwitchGLU(nn.Module):
    """SwitchGLU with fused gate/up and the shared expert packed as the last
    expert — matches the checkpoint's switch_mlp.gate_up_proj layout."""

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation: MiniMaxSwiGLUOAI,
    ):
        super().__init__()
        self.gate_up_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=False
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=False)
        self.activation = activation

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        gate, up = mx.split(gate_up, 2, axis=-1)
        x = self.down_proj(self.activation(up, gate), idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class MiniMaxAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = args.use_qk_norm

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        if self.use_qk_norm:
            self.q_norm = MiniMaxRMSNorm(
                self.head_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )
            self.k_norm = MiniMaxRMSNorm(
                self.head_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )

        self.has_sparse_index = args.has_sparse_index(layer_idx)
        if self.has_sparse_index:
            sparse_config = args.sparse_attention_config
            self.sparse_block_size = sparse_config.get("sparse_block_size", 128)
            self.sparse_topk_blocks = sparse_config.get("sparse_topk_blocks", 16)
            self.sparse_init_blocks = sparse_config.get("sparse_init_block", 0)
            self.sparse_local_blocks = sparse_config.get("sparse_local_block", 1)
            self.sparse_score_type = sparse_config.get("sparse_score_type", "max")
            self.index_dim = sparse_config.get("sparse_index_dim", self.head_dim)
            self.index_heads = sparse_config.get("sparse_num_index_heads", 4)
            self.index_q_proj = nn.Linear(
                args.hidden_size, self.index_heads * self.index_dim, bias=False
            )
            self.index_k_proj = nn.Linear(args.hidden_size, self.index_dim, bias=False)
            self.index_q_norm = MiniMaxRMSNorm(
                self.index_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )
            self.index_k_norm = MiniMaxRMSNorm(
                self.index_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )
            self.indexer = MiniMaxM3Indexer(self)
        else:
            self.indexer = None

        self.rope = initialize_rope(
            args.rotary_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    @staticmethod
    def _normalize_attention_mask(
        mask: Optional[mx.array], B: int, L: int, total_len: int
    ):
        """Array masks → boolean-compatible (B or 1, H or 1, L, total_len)."""
        if mask is None or isinstance(mask, str):
            return mask
        if mask.dtype in (mx.int8, mx.int16, mx.int32, mx.int64, mx.uint8):
            mask = mask.astype(mx.bool_)
        if mask.ndim == 2:
            if mask.shape == (L, total_len) or mask.shape[0] != B:
                mask = mask[None, None, :, :]
            else:
                mask = mask[:, None, None, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :] if mask.shape[0] == B else mask[None, :, :, :]
        if mask.shape[-1] != total_len:
            mask = mask[..., :total_len]
        return mask

    @staticmethod
    def _selection_valid_mask(
        mask: Optional[mx.array], B: int, H_idx: int, L: int, total_len: int
    ):
        valid = MiniMaxAttention._normalize_attention_mask(mask, B, L, total_len)
        if valid is None or isinstance(valid, str):
            return None
        if not _is_bool_mask(valid):
            valid = valid.astype(mx.float32) > mx.array(-1e20, dtype=mx.float32)
        if valid.shape[1] != 1 and valid.shape[1] != H_idx:
            valid = mx.any(valid, axis=1, keepdims=True)
        heads = H_idx if valid.shape[1] == H_idx else 1
        return mx.broadcast_to(valid, (B, heads, L, total_len))

    @staticmethod
    def _merge_sparse_mask(sparse_mask: mx.array, mask: Optional[mx.array]):
        B, _, L, total_len = sparse_mask.shape
        mask = MiniMaxAttention._normalize_attention_mask(mask, B, L, total_len)
        if mask is None or isinstance(mask, str):
            return sparse_mask
        if _is_bool_mask(mask):
            return sparse_mask & mask
        sparse_bias = mx.where(
            sparse_mask,
            mx.array(0.0, dtype=mask.dtype),
            mx.array(-float("inf"), dtype=mask.dtype),
        )
        return sparse_bias + mask

    def _sparse_decode_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        block_indices: mx.array,
        mask: Optional[mx.array],
        q_positions: mx.array,
    ):
        """Decode fast path: gather only the selected blocks' keys/values.
        Applies for a single query position with no explicit array mask."""
        if queries.shape[2] != 1 or (mask is not None and not isinstance(mask, str)):
            return None

        B = queries.shape[0]
        key_length = keys.shape[2]
        selected_length = block_indices.shape[-1] * self.sparse_block_size
        # Dense SDPA is still faster before this measured crossover point.
        if selected_length >= key_length or key_length < selected_length * 64:
            return None

        block_indices = block_indices.astype(mx.int32)
        offsets = mx.arange(self.sparse_block_size, dtype=mx.int32)
        token_indices = block_indices[..., None] * self.sparse_block_size + offsets
        valid = (
            (block_indices[..., None] >= 0)
            & (token_indices < key_length)
            & (token_indices <= q_positions[..., None, None].astype(mx.int32))
        )

        token_indices = token_indices.reshape(B, 1, selected_length)
        valid = valid.reshape(B, 1, selected_length)
        safe_indices = mx.where(valid, token_indices, mx.zeros_like(token_indices))

        gather_indices = mx.broadcast_to(
            safe_indices[:, None, 0, :, None],
            (B, keys.shape[1], selected_length, keys.shape[3]),
        )
        compact_keys = mx.take_along_axis(keys, gather_indices, axis=2)
        compact_values = mx.take_along_axis(values, gather_indices, axis=2)

        return scaled_dot_product_attention(
            queries,
            compact_keys,
            compact_values,
            cache=None,
            scale=self.scale,
            mask=valid[:, None],
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        offset = cache.offset if cache is not None else 0
        use_sparse = self.has_sparse_index and (
            cache is None or hasattr(cache, "update_index_and_fetch")
        )
        # Read before the kv/index caches advance below.
        q_start = getattr(cache, "index_offset", offset) if cache is not None else 0

        queries = self.q_proj(x).reshape(B, L, self.num_attention_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)

        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if use_sparse:
            block_indices, q_positions = self.indexer(x, offset, q_start, cache, mask)
            if block_indices is not None:
                output = self._sparse_decode_attention(
                    queries, keys, values, block_indices, mask, q_positions
                )
                if output is None:
                    sparse_mask = self.indexer.build_block_mask(
                        block_indices, mask, keys.shape[2], q_positions
                    )
                    output = scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        cache=cache,
                        scale=self.scale,
                        mask=sparse_mask,
                    )
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MiniMaxM3Indexer:
    """MiniMax M3 sparse block selector.

    The projection modules live on MiniMaxAttention to preserve checkpoint
    keys; this helper only orchestrates: score index queries against the roped
    index-key cache, pick top-k causal key blocks (init/local blocks forced),
    and optionally expand the selection into a token mask."""

    def __init__(self, attention: MiniMaxAttention):
        self.attention = attention
        self.block_size = attention.sparse_block_size
        self.topk_blocks = attention.sparse_topk_blocks
        self.init_blocks = attention.sparse_init_blocks
        self.local_blocks = attention.sparse_local_blocks
        self.score_type = attention.sparse_score_type

    def __call__(
        self,
        x: mx.array,
        rope_offset: int,
        q_start: int,
        cache,
        mask: Optional[mx.array],
    ):
        attention = self.attention
        B, L, _ = x.shape
        idx_queries = attention.index_q_proj(x).reshape(
            B, L, attention.index_heads, attention.index_dim
        )
        idx_keys = attention.index_k_proj(x).reshape(B, L, 1, attention.index_dim)
        idx_queries = attention.index_q_norm(idx_queries).transpose(0, 2, 1, 3)
        idx_keys = attention.index_k_norm(idx_keys).transpose(0, 2, 1, 3)
        idx_queries = attention.rope(idx_queries, offset=rope_offset)
        idx_keys = attention.rope(idx_keys, offset=rope_offset)

        if cache is not None:
            idx_keys = cache.update_index_and_fetch(idx_keys)

        positions = mx.arange(q_start, q_start + L)
        q_positions = mx.broadcast_to(positions[None, :], (B, L))
        total_len = idx_keys.shape[2]
        if total_len <= self.block_size * self.topk_blocks:
            # Every causal block would be selected anyway: dense == sparse.
            return None, q_positions

        block_indices = self.select_blocks(idx_queries, idx_keys, q_positions, mask)
        return block_indices, q_positions

    def select_blocks(
        self,
        idx_queries: mx.array,
        idx_keys: mx.array,
        q_positions: mx.array,
        mask: Optional[mx.array] = None,
    ):
        attention = self.attention
        B, H_idx, L, _ = idx_queries.shape
        total_len = idx_keys.shape[2]
        num_blocks = (total_len + self.block_size - 1) // self.block_size
        if (
            (mask is None or isinstance(mask, str))
            and self.score_type == "max"
            and num_blocks >= self.topk_blocks
        ):
            if attention.num_attention_heads % H_idx != 0:
                raise ValueError(
                    "MiniMax M3 sparse index heads must divide attention heads: "
                    f"{H_idx} index heads, {attention.num_attention_heads} "
                    "attention heads."
                )
            return _select_sparse_block_indices_compiled(
                idx_queries,
                idx_keys,
                q_positions,
                attention.scale,
                self.block_size,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
            )

        neg = mx.array(-float("inf"), dtype=mx.float32)
        scores = mx.matmul(
            idx_queries.astype(mx.float32),
            idx_keys.astype(mx.float32).swapaxes(-1, -2),
        )
        scores = scores * attention.scale

        kpos = mx.arange(total_len)
        causal = kpos[None, None, :] <= q_positions[:, :, None]
        scores = mx.where(causal[:, None], scores, neg)
        valid = attention._selection_valid_mask(mask, B, H_idx, L, total_len)
        if valid is not None:
            scores = mx.where(valid, scores, neg)

        pad = num_blocks * self.block_size - total_len
        if pad:
            pad_values = mx.full(
                (*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype
            )
            scores = mx.concatenate([scores, pad_values], axis=-1)
            if valid is not None:
                valid_pad = mx.zeros((*valid.shape[:-1], pad), dtype=mx.bool_)
                valid = mx.concatenate([valid, valid_pad], axis=-1)

        blocks = mx.arange(num_blocks)
        cur_block = q_positions // self.block_size
        causal_block = blocks[None, None, :] <= cur_block[:, :, None]
        scores = scores.reshape(B, H_idx, L, num_blocks, self.block_size)
        if valid is not None:
            valid_blocks = mx.any(
                valid.reshape(B, valid.shape[1], L, num_blocks, self.block_size),
                axis=-1,
            )
        else:
            valid_blocks = causal_block[:, None]

        if self.score_type == "lse":
            block_scores = mx.logsumexp(scores, axis=-1)
        else:
            block_scores = mx.max(scores, axis=-1)
        block_scores = mx.max(block_scores, axis=1)
        block_scores = mx.where(block_scores == block_scores, block_scores, neg)
        if valid_blocks.shape[1] == 1:
            valid_blocks = valid_blocks[:, 0]
        else:
            valid_blocks = mx.any(valid_blocks, axis=1)
        valid_blocks = valid_blocks & causal_block

        selected_scores = mx.where(valid_blocks, block_scores, neg)
        if self.init_blocks > 0:
            init_blocks = (blocks[None, None, :] < self.init_blocks) & valid_blocks
            selected_scores = mx.where(
                init_blocks,
                mx.array(1e30, dtype=selected_scores.dtype),
                selected_scores,
            )
        if self.local_blocks > 0:
            local_start = mx.maximum(cur_block - self.local_blocks + 1, 0)
            local_blocks = (blocks[None, None, :] >= local_start[:, :, None]) & (
                blocks[None, None, :] <= cur_block[:, :, None]
            )
            selected_scores = mx.where(
                local_blocks & valid_blocks,
                mx.array(1e29, dtype=selected_scores.dtype),
                selected_scores,
            )

        topk = min(self.topk_blocks, num_blocks)
        topk_idx = mx.argpartition(-selected_scores, kth=topk - 1, axis=-1)[..., :topk]
        topk_valid = mx.take_along_axis(valid_blocks, topk_idx, axis=-1)
        invalid = mx.full(topk_idx.shape, num_blocks, dtype=topk_idx.dtype)
        block_indices = mx.where(topk_valid, topk_idx, invalid)
        order = mx.argsort(block_indices, axis=-1)
        block_indices = mx.take_along_axis(block_indices, order, axis=-1)
        return mx.where(block_indices == num_blocks, mx.array(-1), block_indices)

    def build_block_mask(
        self,
        block_indices: mx.array,
        mask: Optional[mx.array],
        key_length: int,
        q_positions: mx.array,
    ):
        B, L, _ = block_indices.shape
        num_blocks = (key_length + self.block_size - 1) // self.block_size
        blocks = mx.arange(num_blocks, dtype=block_indices.dtype)
        block_keep = mx.any(block_indices[..., None] == blocks, axis=-2)

        kpos = mx.arange(key_length)
        key_blocks = (kpos // self.block_size).astype(block_indices.dtype)
        key_blocks = mx.broadcast_to(key_blocks[None, None, :], (B, L, key_length))
        key_keep = mx.take_along_axis(block_keep, key_blocks, axis=-1)
        causal = kpos[None, None, :] <= q_positions[:, :, None]
        sparse_mask = (key_keep & causal)[:, None]
        return self.attention._merge_sparse_mask(sparse_mask, mask)


class MiniMaxSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor
        self.scoring_func = args.scoring_func
        self.shared_expert_index = args.num_local_experts
        self.pack_shared_expert = (
            args.n_shared_experts == 1
            and args.shared_intermediate_size == args.intermediate_size
        )

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        activation = MiniMaxSwiGLUOAI(
            args.swiglu_alpha, args.swiglu_limit, args.swiglu_beta
        )
        if self.pack_shared_expert:
            self.switch_mlp = MiniMaxPackedSwitchGLU(
                args.hidden_size,
                args.intermediate_size,
                args.num_local_experts + 1,
                activation=activation,
            )
        else:
            self.switch_mlp = SwitchGLU(
                args.hidden_size,
                args.intermediate_size,
                args.num_local_experts,
                activation=activation,
            )
        self.shared_experts = (
            MiniMaxMLP(
                args.hidden_size,
                args.shared_intermediate_size,
                args.swiglu_alpha,
                args.swiglu_limit,
                args.swiglu_beta,
            )
            if args.n_shared_experts and not self.pack_shared_expert
            else None
        )
        self.e_score_correction_bias = (
            mx.zeros((args.num_local_experts,)) if args.use_routing_bias else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x.astype(mx.float32))
        if self.e_score_correction_bias is not None:
            inds, scores = _minimax_moe_select(
                gates,
                self.e_score_correction_bias,
                self.num_experts_per_tok,
                self.routed_scaling_factor,
                self.scoring_func,
            )
            scores = scores.astype(x.dtype)
        else:
            if self.scoring_func == "sigmoid":
                scores = mx.sigmoid(gates)
            else:
                scores = mx.softmax(gates, axis=-1, precise=True)
            k = self.num_experts_per_tok
            inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
            scores = mx.take_along_axis(scores, inds, axis=-1)
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
            scores = (scores * self.routed_scaling_factor).astype(x.dtype)

        if self.pack_shared_expert:
            shared_inds = mx.full(
                (*inds.shape[:-1], 1), self.shared_expert_index, dtype=inds.dtype
            )
            shared_scores = mx.ones((*scores.shape[:-1], 1), dtype=scores.dtype)
            inds = mx.concatenate([inds, shared_inds], axis=-1)
            scores = mx.concatenate([scores, shared_scores], axis=-1)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class MiniMaxDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = MiniMaxAttention(args, layer_idx)
        self.input_layernorm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )
        self.post_attention_layernorm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )
        self.is_moe_layer = args.is_moe_layer(layer_idx)
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxSparseMoeBlock(args)
        else:
            self.mlp = MiniMaxMLP(
                args.hidden_size,
                args.dense_intermediate_size,
                args.swiglu_alpha,
                args.swiglu_limit,
                args.swiglu_beta,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        mlp = self.block_sparse_moe if self.is_moe_layer else self.mlp
        return h + mlp(self.post_attention_layernorm(h))


class MiniMaxM3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MiniMaxDecoderLayer(args=args, layer_idx=layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )

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

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxM3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        """Load text-only, mlx-vlm-converted, and original VL checkpoints:
        strip the vision side and remap language_model.* to this layout, then
        stack any per-expert HF weights into switch_mlp tensors."""
        vision_prefixes = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")
        new_weights = {}
        for k, v in weights.items():
            if k.startswith(vision_prefixes) or k.startswith(
                tuple(f"model.{p}" for p in vision_prefixes)
            ):
                continue
            if k.startswith("model.language_model."):
                # New-style HF VL: the bare text model under model.language_model.*
                k = k.replace("model.language_model.", "model.", 1)
            elif k.startswith("language_model."):
                # mlx-vlm conversions: language_model.{model,lm_head}.*
                k = k.replace("language_model.", "", 1)
            new_weights[k] = v
        weights = new_weights

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        pack_shared = (
            self.args.n_shared_experts == 1
            and self.args.shared_intermediate_size == self.args.intermediate_size
        )
        num_experts = self.args.num_local_experts

        # mlx quants made from the UNFUSED module tree (separate switch_mlp
        # gate_proj/up_proj + a standalone shared expert — e.g. pipenetwork's
        # mixed-bit M3) already ship stacked tensors matching the unpacked
        # branch. Rebuild the MoE blocks unpacked so the module tree matches the
        # checkpoint — plain SwitchGLU is also the layout the expert-streaming
        # delegate knows how to stream.
        # Checkpoints exported WITHOUT the MSA indexer weights (e.g. pipenetwork's
        # mixed-bit quant ships no self_attn.index_*): fall back to DENSE attention
        # — the exact form the sparse indexer approximates — by clearing the
        # per-instance sparse flag (the forward and make_cache both read it) and
        # dropping the index modules so strict load_weights doesn't demand them.
        if not any(k.endswith("self_attn.index_q_proj.weight") for k in weights):
            for layer in self.model.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is None or not getattr(attn, "has_sparse_index", False):
                    continue
                attn.has_sparse_index = False
                attn.indexer = None
                for name in (
                    "index_q_proj",
                    "index_k_proj",
                    "index_q_norm",
                    "index_k_norm",
                ):
                    if hasattr(attn, name):
                        delattr(attn, name)

        unfused = any(
            k.endswith("block_sparse_moe.switch_mlp.gate_proj.weight") for k in weights
        )
        if unfused and pack_shared:
            pack_shared = False
            activation = MiniMaxSwiGLUOAI(
                self.args.swiglu_alpha, self.args.swiglu_limit, self.args.swiglu_beta
            )
            for layer in self.model.layers:
                moe = getattr(layer, "block_sparse_moe", None)
                if moe is None or not getattr(moe, "pack_shared_expert", False):
                    continue
                moe.pack_shared_expert = False
                moe.switch_mlp = SwitchGLU(
                    self.args.hidden_size,
                    self.args.intermediate_size,
                    num_experts,
                    activation=activation,
                )
                moe.shared_experts = MiniMaxMLP(
                    self.args.hidden_size,
                    self.args.shared_intermediate_size,
                    self.args.swiglu_alpha,
                    self.args.swiglu_limit,
                    self.args.swiglu_beta,
                )

        def expert_keys(prefix, name, suffix):
            return [
                f"{prefix}.experts.{expert}.{name}.{suffix}"
                for expert in range(num_experts)
            ]

        def pop_stack(keys):
            return mx.stack([weights.pop(key) for key in keys])

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.block_sparse_moe"
            for suffix in ("weight", "scales", "biases", "bias"):
                if pack_shared:
                    gate_keys = expert_keys(prefix, "w1", suffix)
                    up_keys = expert_keys(prefix, "w3", suffix)
                    shared_gate_key = f"{prefix}.shared_experts.gate_proj.{suffix}"
                    shared_up_key = f"{prefix}.shared_experts.up_proj.{suffix}"
                    if all(
                        key in weights
                        for key in [*gate_keys, *up_keys, shared_gate_key, shared_up_key]
                    ):
                        gate_up = mx.concatenate(
                            [pop_stack(gate_keys), pop_stack(up_keys)], axis=1
                        )
                        shared_gate_up = mx.concatenate(
                            [weights.pop(shared_gate_key), weights.pop(shared_up_key)],
                            axis=0,
                        )[None]
                        weights[f"{prefix}.switch_mlp.gate_up_proj.{suffix}"] = (
                            mx.concatenate([gate_up, shared_gate_up], axis=0)
                        )

                    down_keys = expert_keys(prefix, "w2", suffix)
                    shared_down_key = f"{prefix}.shared_experts.down_proj.{suffix}"
                    if all(key in weights for key in [*down_keys, shared_down_key]):
                        weights[f"{prefix}.switch_mlp.down_proj.{suffix}"] = (
                            mx.concatenate(
                                [
                                    pop_stack(down_keys),
                                    weights.pop(shared_down_key)[None],
                                ],
                                axis=0,
                            )
                        )
                    continue

                for hf_name, mlx_name in (
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ):
                    keys = expert_keys(prefix, hf_name, suffix)
                    if all(key in weights for key in keys):
                        weights[f"{prefix}.switch_mlp.{mlx_name}.{suffix}"] = pop_stack(
                            keys
                        )

        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            MiniMaxM3KVCache() if layer.self_attn.has_sparse_index else KVCache()
            for layer in self.layers
        ]

    @property
    def cast_predicate(self):
        def predicate(k):
            keep_fp32 = "e_score_correction_bias" in k or k.endswith(
                "block_sparse_moe.gate.weight"
            )
            return not keep_fp32

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("block_sparse_moe.gate"):
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return True

        return predicate
