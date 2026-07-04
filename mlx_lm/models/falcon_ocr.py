# Copyright © 2026 Apple Inc.
#
# Falcon-OCR (TII, 300M): a bespoke early-fusion body — fused QKV, weightless
# RMSNorms baked into the attention/MLP blocks, attention sinks, relu²-gated MLP,
# and a split rotary embedding whose first half is plain 1D RoPE over (image-
# collapsed) positions and whose second half is a learned per-head 2D "golden"
# rotary over image (h, w) coordinates. Ported from Blaizzy/mlx-vlm
# models/falcon_ocr (MIT © Blaizzy/mlx-vlm contributors), text side only.
#
# Vision (mlx-unified): the bridge injects merged embeddings via
# `input_embeddings` and installs the arch's forward-changing state through
# set_visual_state/reset_visual_state on the inner model: collapsed position_ids
# + rope_deltas (all tokens of an image block share one 1D position; decode
# continues at cache_offset + delta), pos_hw golden coordinates, and the
# block-diagonal mask that lets image tokens attend bidirectionally within
# their block. All of it comes straight from mlx-vlm's get_input_embeddings.

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "falcon_ocr"
    hidden_size: int = 768
    num_hidden_layers: int = 22
    num_attention_heads: int = 16
    head_dim: int = 64
    num_key_value_heads: int = 8
    vocab_size: int = 65536
    intermediate_size: int = 2304
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params):
        # The TII checkpoint config uses bespoke flat names.
        aliases = {
            "dim": "hidden_size",
            "n_layers": "num_hidden_layers",
            "n_heads": "num_attention_heads",
            "n_kv_heads": "num_key_value_heads",
            "ffn_dim": "intermediate_size",
            "norm_eps": "rms_norm_eps",
            "max_seq_len": "max_position_embeddings",
        }
        return super().from_dict({aliases.get(k, k): v for k, v in params.items()})


def precompute_freqs_1d(
    dim: int, end: int, theta: float
) -> Tuple[mx.array, mx.array]:
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32)[: dim // 2] / dim))
    t = mx.arange(end).astype(mx.float32)
    freqs = t[:, None] * freqs[None, :]
    return mx.cos(freqs), mx.sin(freqs)


def apply_rotary_emb_1d(
    xq: mx.array, xk: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    dtype = xq.dtype
    *shape_q, d = xq.shape
    *shape_k, _ = xk.shape
    xq_r = xq.astype(mx.float32).reshape(*shape_q, d // 2, 2)
    xk_r = xk.astype(mx.float32).reshape(*shape_k, d // 2, 2)
    xq_0, xq_1 = xq_r[..., 0], xq_r[..., 1]
    xk_0, xk_1 = xk_r[..., 0], xk_r[..., 1]
    if cos.ndim == 2:
        c = cos.reshape(1, 1, -1, cos.shape[-1])
        s = sin.reshape(1, 1, -1, sin.shape[-1])
    else:
        c = cos.reshape(cos.shape[0], 1, -1, cos.shape[-1])
        s = sin.reshape(sin.shape[0], 1, -1, sin.shape[-1])
    oq = mx.stack([xq_0 * c - xq_1 * s, xq_0 * s + xq_1 * c], axis=-1)
    ok = mx.stack([xk_0 * c - xk_1 * s, xk_0 * s + xk_1 * c], axis=-1)
    return oq.reshape(*shape_q, d).astype(dtype), ok.reshape(*shape_k, d).astype(dtype)


def compute_golden_freqs(
    freqs_golden: mx.array, pos_hw: mx.array
) -> Tuple[mx.array, mx.array]:
    theta = mx.einsum(
        "bsp,hfp->bshf", pos_hw.astype(mx.float32), freqs_golden.astype(mx.float32)
    )
    return mx.cos(theta), mx.sin(theta)


def apply_golden_rotary_emb(x: mx.array, cos_2d: mx.array, sin_2d: mx.array) -> mx.array:
    dtype = x.dtype
    cos = cos_2d.transpose(0, 2, 1, 3)
    sin = sin_2d.transpose(0, 2, 1, 3)
    x_f = x.astype(mx.float32)
    x_even, x_odd = x_f[..., 0::2], x_f[..., 1::2]
    o_even = x_even * cos - x_odd * sin
    o_odd = x_even * sin + x_odd * cos
    return mx.stack([o_even, o_odd], axis=-1).reshape(x.shape).astype(dtype)


def apply_3d_rotary_emb(
    xq: mx.array,
    xk: mx.array,
    cos_1d: mx.array,
    sin_1d: mx.array,
    cos_2d: Optional[mx.array],
    sin_2d: Optional[mx.array],
) -> Tuple[mx.array, mx.array]:
    # First half of head_dim: 1D temporal RoPE; second half: 2D golden rotary
    # (identity for text tokens, whose pos_hw is zero — so text-only forwards
    # skip it entirely).
    half = xq.shape[-1] // 2
    xq_t, xq_hw = xq[..., :half], xq[..., half:]
    xk_t, xk_hw = xk[..., :half], xk[..., half:]
    xq_t, xk_t = apply_rotary_emb_1d(xq_t, xk_t, cos_1d, sin_1d)
    if cos_2d is not None:
        xq_hw = apply_golden_rotary_emb(xq_hw, cos_2d, sin_2d)
        xk_hw = apply_golden_rotary_emb(xk_hw, cos_2d, sin_2d)
    return (
        mx.concatenate([xq_t, xq_hw], axis=-1).astype(xq.dtype),
        mx.concatenate([xk_t, xk_hw], axis=-1).astype(xk.dtype),
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.eps = args.rms_norm_eps
        self.q_size = self.n_heads * self.head_dim
        self.kv_size = self.n_kv_heads * self.head_dim
        self.wqkv = nn.Linear(args.hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.wo = nn.Linear(self.q_size, args.hidden_size, bias=False)
        self.sinks = mx.zeros((self.n_heads,))
        # The checkpoint ships no block-norm weights (weightless RMSNorm) —
        # underscore attrs so MLX never registers them as parameters.
        self._norm_w_in = mx.ones((args.hidden_size,))
        self._norm_w_qk = mx.ones((self.head_dim,))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        cos_1d: mx.array,
        sin_1d: mx.array,
        cos_2d: Optional[mx.array],
        sin_2d: Optional[mx.array],
    ) -> mx.array:
        B, L, _ = x.shape
        x_norm = mx.fast.rms_norm(x, self._norm_w_in, eps=self.eps)

        qkv = self.wqkv(x_norm)
        q = qkv[..., : self.q_size]
        k = qkv[..., self.q_size : self.q_size + self.kv_size]
        v = qkv[..., self.q_size + self.kv_size :]

        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = mx.fast.rms_norm(q, self._norm_w_qk, eps=self.eps)
        k = mx.fast.rms_norm(k, self._norm_w_qk, eps=self.eps)

        # KV must be expanded to n_heads BEFORE rotary (and thus before the
        # cache): the golden rotary has per-head frequencies, so each repeated
        # copy receives a different spatial rotation.
        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=1)
            v = mx.repeat(v, self.n_rep, axis=1)

        q, k = apply_3d_rotary_emb(q, k, cos_1d, sin_1d, cos_2d, sin_2d)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask, sinks=self.sinks
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_dim = args.intermediate_size
        self.eps = args.rms_norm_eps
        self.w13 = nn.Linear(args.hidden_size, 2 * args.intermediate_size, bias=False)
        self.w2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self._norm_w = mx.ones((args.hidden_size,))

    def __call__(self, x: mx.array) -> mx.array:
        x_norm = mx.fast.rms_norm(x, self._norm_w, eps=self.eps)
        w13_out = self.w13(x_norm)
        gate = w13_out[..., : self.hidden_dim]
        up = w13_out[..., self.hidden_dim :]
        return self.w2(nn.relu(gate) ** 2 * up)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)

    def __call__(self, x, mask, cache, cos_1d, sin_1d, cos_2d, sin_2d):
        x = x + self.self_attn(x, mask, cache, cos_1d, sin_1d, cos_2d, sin_2d)
        return x + self.mlp(x)


class FalconOCRModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        rope_dim = args.head_dim // 2
        # 1D tables are derived, never loaded (sanitize drops checkpoint copies);
        # the golden frequencies are real checkpoint data — [n_heads, rope_dim//2, 2].
        self._cos_1d, self._sin_1d = precompute_freqs_1d(
            rope_dim, args.max_position_embeddings, args.rope_theta
        )
        self.freqs_cis_golden = mx.zeros((args.num_attention_heads, rope_dim // 2, 2))

        # Visual side state (mlx-unified), installed by the server before a
        # vision generation and cleared afterwards. Underscore attrs so MLX's
        # module walker never registers them as parameters.
        self._vis_position_ids = None
        self._vis_pos_hw = None
        self._vis_rope_delta = None
        self._vis_mask = None

    def set_visual_state(
        self,
        *,
        position_ids: mx.array,
        pos_hw: mx.array,
        rope_deltas: mx.array,
        attention_mask_4d: mx.array,
    ) -> None:
        """Install falcon_ocr vision state for a single (B=1) prompt of length S:
        collapsed 1D positions (image blocks share one position), (1, S, 2) golden
        h/w coordinates, the decode position delta, and the (1, 1, S, S) causal-or-
        same-image-block mask. All sliced by cache offset per forward, so KV-cache
        reuse of a prompt prefix needs no re-alignment here."""
        self._vis_position_ids = position_ids.reshape(-1)
        self._vis_pos_hw = pos_hw
        self._vis_rope_delta = int(rope_deltas.reshape(-1)[0].item())
        self._vis_mask = attention_mask_4d

    def reset_visual_state(self) -> None:
        self._vis_position_ids = None
        self._vis_pos_hw = None
        self._vis_rope_delta = None
        self._vis_mask = None

    def _positions(self, offset, L: int):
        """(pos_t, pos_hw) for the span [offset, offset+L).

        Text-only: sequential positions from the cache offset (per-row for
        batched array offsets), no golden coordinates. With visual state:
        stored prompt positions/coordinates sliced by offset; past the prompt,
        decode positions continue at offset + rope_delta with zero (identity)
        golden coordinates — stitching both if a multi-token decode step
        (draft models) straddles the prompt boundary.
        """
        if self._vis_position_ids is None:
            if isinstance(offset, mx.array) and offset.size > 1:
                return mx.maximum(offset, 0)[:, None] + mx.arange(L), None
            off = int(offset.item()) if isinstance(offset, mx.array) else offset
            return mx.arange(off, off + L), None

        offset = int(offset)
        S = self._vis_position_ids.shape[0]
        end = min(offset + L, S)
        if offset >= S:
            return mx.arange(offset, offset + L) + self._vis_rope_delta, None
        pos_t = self._vis_position_ids[offset:end]
        pos_hw = self._vis_pos_hw[:, offset:end]
        if end < offset + L:
            tail = mx.arange(end, offset + L) + self._vis_rope_delta
            pos_t = mx.concatenate([pos_t, tail.astype(pos_t.dtype)])
            pos_hw = mx.pad(pos_hw, ((0, 0), (0, offset + L - end), (0, 0)))
        return pos_t, pos_hw

    def _mask(self, h: mx.array, offset, L: int, c0):
        if self._vis_mask is not None and L > 1:
            S = self._vis_mask.shape[2]
            if int(offset) + L <= S:
                off = int(offset)
                return self._vis_mask[:, :, off : off + L, : off + L]
        # Decode steps (and anything past the stored prompt) are text tokens:
        # plain causal attention over everything, image KV included.
        return create_attention_mask(h, c0)

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

        L = h.shape[1]
        offset = cache[0].offset if cache[0] is not None else 0

        pos_t, pos_hw = self._positions(offset, L)
        cos_1d = self._cos_1d[pos_t]
        sin_1d = self._sin_1d[pos_t]
        cos_2d = sin_2d = None
        if pos_hw is not None:
            cos_2d, sin_2d = compute_golden_freqs(self.freqs_cis_golden, pos_hw)

        mask = self._mask(h, offset, L, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, cos_1d, sin_1d, cos_2d, sin_2d)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FalconOCRModel(args)
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
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("language_model."):
                # mlx-vlm conversion: language_model.{model,lm_head}.*, w13
                # already de-interleaved. Drop the derived 1D tables and the
                # patch projector (vision-input side; the bridge owns it).
                k = k[len("language_model.") :]
                if k in ("model.cos_1d", "model.sin_1d", "model.img_projector.weight"):
                    continue
            elif k.startswith("img_projector."):
                continue
            elif k.startswith("tok_embeddings."):
                k = "model.embed_tokens." + k[len("tok_embeddings.") :]
            elif k.startswith(("norm.", "layers.")) or k == "freqs_cis_golden":
                if k.startswith("layers."):
                    k = k.replace(".attention.", ".self_attn.", 1)
                    k = k.replace(".feed_forward.", ".mlp.", 1)
                    if ".w13." in k:
                        # Raw checkpoints interleave gate/up rows; the runtime
                        # layout is [gate; up].
                        v = mx.concatenate([v[0::2], v[1::2]], axis=0)
                k = "model." + k
            elif k.startswith("output."):
                k = "lm_head." + k[len("output.") :]
            new_weights[k] = v
        return new_weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(path: str):
            # Rotary frequencies lose too much precision in low-bit floats.
            return not path.endswith("freqs_cis_golden")

        return predicate
