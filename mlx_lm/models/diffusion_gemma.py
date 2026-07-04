# Copyright © 2026 Apple Inc.
#
# DiffusionGemma: block-diffusion text generation over a gemma4-family body.
# Ported from Blaizzy/mlx-vlm's models/diffusion_gemma (MIT, © Blaizzy / mlx-vlm
# contributors), vision tower omitted — mlx-unified model files stay text-only.
#
# One set of transformer weights runs in two modes:
#   encoder — a causal/sliding-window pass that appends the prompt (and, later,
#             each accepted block) to the KV cache; it differs from the decoder
#             only by per-layer `layer_scalar`s of its own.
#   decoder — a bidirectional pass over a "canvas" of random token ids that
#             reads the cached context but writes nothing back; the denoising
#             loop re-runs it until every canvas position is accepted.
# Plain `Model.__call__` runs the encoder and returns its logits, satisfying the
# standard mlx-lm forward contract; generation must go through the diffusion_*
# methods, driven by mlx_lm.diffusion_generate.

import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .gemma4_text import Experts, RMSNormNoScale, Router, geglu
from .rope_utils import initialize_rope


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "diffusion_gemma_text"
    vocab_size: int = 262144
    hidden_size: int = 2816
    intermediate_size: int = 2112
    moe_intermediate_size: int = 704
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: Optional[int] = 2
    head_dim: int = 256
    global_head_dim: int = 512
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144
    attention_bias: bool = False
    sliding_window: int = 1024
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[Dict[str, Dict[str, Any]]] = None
    final_logit_softcapping: float = 30.0
    num_experts: int = 128
    top_k_experts: int = 8

    def __post_init__(self):
        if self.layer_types is None:
            pattern = ["sliding_attention"] * 5 + ["full_attention"]
            self.layer_types = (pattern * (self.num_hidden_layers // len(pattern) + 1))[
                : self.num_hidden_layers
            ]
            if self.layer_types[-1] != "full_attention":
                self.layer_types[-1] = "full_attention"
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                },
            }


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "diffusion_gemma"
    text_config: Optional[dict] = None
    canvas_length: int = 256
    generation_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.text_config is None:
            self.text_config = {}


def make_compiled_softcap(softcap: float):
    """Fused fp32 upcast + tanh softcap (one pass over the vocab logits)."""

    def _softcap(x):
        return mx.tanh(x.astype(mx.float32) / softcap) * softcap

    return mx.compile(_softcap, shapeless=True)


def _cache_offset(cache) -> int:
    if cache is None or getattr(cache, "keys", None) is None:
        return 0
    return int(cache.offset)


def _cache_state(cache):
    """Cached (keys, values) in temporal order, or None while empty."""
    if cache is None or getattr(cache, "keys", None) is None:
        return None
    if hasattr(cache, "_temporal_order"):
        return cache._temporal_order(cache.keys), cache._temporal_order(cache.values)
    return cache.state


class MLP(nn.Module):
    def __init__(self, config: TextArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x):
        return self.down_proj(geglu(self.gate_proj(x), self.up_proj(x)))


class Attention(nn.Module):
    def __init__(self, config: TextArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"

        self.head_dim = (
            config.global_head_dim
            if not self.is_sliding and config.global_head_dim
            else config.head_dim
        )
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = (
            config.num_global_key_value_heads
            if not self.is_sliding and config.num_global_key_value_heads is not None
            else config.num_key_value_heads
        )
        self.scale = 1.0

        dim = config.hidden_size
        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias
        )
        # Full-attention layers share K as V; only sliding layers project V.
        self.v_proj = (
            nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
            if self.is_sliding
            else None
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=config.attention_bias
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)

        rope_params = config.rope_parameters.get(self.layer_type, {})
        self.rope = initialize_rope(
            dims=self.head_dim,
            traditional=False,
            base=rope_params.get("rope_theta", 10000.0),
            scaling_config=rope_params,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        *,
        decoder: bool = False,
        offset: Optional[int] = None,
    ):
        B, L, _ = x.shape
        if offset is None:
            offset = _cache_offset(cache)

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        values = (
            self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            if self.v_proj is not None
            else keys
        )
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        keys = self.rope(keys, offset=offset)
        values = self.v_norm(values).transpose(0, 2, 1, 3)

        if decoder:
            # Canvas pass: read the cached context, write nothing back.
            state = _cache_state(cache)
            if state is not None:
                context_keys, context_values = state
                if self.is_sliding:
                    # The canvas only attends to the last `sliding_window - 1`
                    # context positions (the mask already zeroes the rest), so
                    # drop the out-of-window keys/values before SDPA instead of
                    # computing scores for thousands of masked positions.
                    window = max(self.config.sliding_window - 1, 0)
                    if window and context_keys.shape[2] > window:
                        context_keys = context_keys[:, :, -window:, :]
                        context_values = context_values[:, :, -window:, :]
                        if mask is not None and not isinstance(mask, str):
                            mask = mask[..., -(window + L) :]
                keys = mx.concatenate([context_keys, keys], axis=2)
                values = mx.concatenate([context_values, values], axis=2)
            attn_cache = None
        else:
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
            attn_cache = cache

        output = scaled_dot_product_attention(
            queries, keys, values, cache=attn_cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, config: TextArgs, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Every layer runs a dense MLP and a sparse MoE branch in parallel.
        self.router = Router(config)
        self.experts = Experts(config)
        self.post_feedforward_layernorm_1 = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm_2 = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm_2 = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        *,
        decoder: bool = False,
        offset: Optional[int] = None,
        layer_scalar: Optional[mx.array] = None,
    ):
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache, decoder=decoder, offset=offset)
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h
        h1 = self.pre_feedforward_layernorm(h)
        h1 = self.mlp(h1)
        h1 = self.post_feedforward_layernorm_1(h1)

        top_k_indices, top_k_weights = self.router(h)
        h2 = self.pre_feedforward_layernorm_2(h)
        h2 = self.experts(h2, top_k_indices, top_k_weights)
        h2 = self.post_feedforward_layernorm_2(h2)

        h = self.post_feedforward_layernorm(h1 + h2)
        h = residual + h
        # The encoder pass substitutes its own per-layer scalar.
        return h * (self.layer_scalar if layer_scalar is None else layer_scalar)


class SelfConditioning(nn.Module):
    """Mixes the previous denoising step's soft prediction into the canvas embedding."""

    def __init__(self, config: TextArgs):
        super().__init__()
        self.pre_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_norm = RMSNormNoScale(config.hidden_size, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, inputs_embeds, self_conditioning_signal):
        normed = self.pre_norm(self_conditioning_signal)
        signal = self.down_proj(geglu(self.gate_proj(normed), self.up_proj(normed)))
        return self.post_norm(inputs_embeds + signal)


class DecoderModel(nn.Module):
    def __init__(self, config: TextArgs):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        self.layers = [
            DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_conditioning = SelfConditioning(config)

    @property
    def prefers_logits_self_conditioning(self) -> bool:
        return isinstance(self.embed_tokens, nn.QuantizedEmbedding)

    def diffusion_prepare_self_conditioning(self) -> Optional[mx.array]:
        if self.prefers_logits_self_conditioning:
            return None
        return self.embed_tokens.weight

    def diffusion_self_conditioning(
        self,
        processed_logits: mx.array,
        embedding_weight: Optional[mx.array],
    ) -> mx.array:
        if self.prefers_logits_self_conditioning:
            return processed_logits
        # Match the HF generation path: self-conditioning signals are stored in
        # the embedding dtype before the next decoder softmax.
        probs = mx.softmax(
            processed_logits.astype(embedding_weight.dtype),
            axis=-1,
            precise=True,
        )
        return (probs.astype(embedding_weight.dtype) @ embedding_weight).astype(
            embedding_weight.dtype
        ) * self.embed_scale

    def _embed_canvas(
        self,
        canvas_ids,
        self_conditioning_logits=None,
        self_conditioning_embeddings=None,
    ):
        inputs_embeds = self.embed_tokens(canvas_ids) * self.embed_scale
        if (
            self_conditioning_logits is not None
            and self_conditioning_embeddings is not None
        ):
            raise ValueError(
                "Only one of self_conditioning_logits or "
                "self_conditioning_embeddings can be set."
            )
        if self_conditioning_embeddings is not None:
            soft_embeddings = self_conditioning_embeddings.astype(inputs_embeds.dtype)
        elif self_conditioning_logits is None:
            soft_embeddings = mx.zeros_like(inputs_embeds)
        else:
            probs = mx.softmax(self_conditioning_logits, axis=-1, precise=True)
            if isinstance(self.embed_tokens, nn.QuantizedEmbedding):
                soft_embeddings = mx.quantized_matmul(
                    probs.astype(inputs_embeds.dtype),
                    self.embed_tokens.weight,
                    self.embed_tokens.scales,
                    self.embed_tokens.biases,
                    transpose=False,
                    group_size=self.embed_tokens.group_size,
                    bits=self.embed_tokens.bits,
                    mode=getattr(self.embed_tokens, "mode", "affine"),
                )
            else:
                soft_embeddings = probs @ self.embed_tokens.weight
            soft_embeddings = (
                soft_embeddings.astype(inputs_embeds.dtype) * self.embed_scale
            )
        return self.self_conditioning(inputs_embeds, soft_embeddings)

    def _make_masks(self, batch_size, canvas_length, caches):
        """One boolean mask per layer type: the canvas attends bidirectionally to
        itself and to the cached context — all of it on full-attention layers
        (mask None), only the trailing `sliding_window - 1` positions on sliding
        layers. mlx-lm's dynamic caches hold no invalid trailing slots, so every
        cached position is valid (the reference's static-cache masking is gone)."""
        masks = {}
        for layer_type in set(self.config.layer_types):
            if layer_type == "full_attention":
                masks[layer_type] = None
                continue
            cache = next(
                (
                    c
                    for c, layer in zip(caches or [], self.layers)
                    if layer.layer_type == layer_type
                ),
                None,
            )
            state = _cache_state(cache)
            context_len = state[0].shape[2] if state is not None else 0
            window_prefix = max(self.config.sliding_window - 1, 0)
            if context_len <= window_prefix:
                masks[layer_type] = None
                continue
            row = mx.concatenate(
                [
                    mx.arange(context_len) >= context_len - window_prefix,
                    mx.ones((canvas_length,), dtype=mx.bool_),
                ]
            )
            masks[layer_type] = mx.broadcast_to(
                row[None, None, None, :],
                (batch_size, 1, canvas_length, context_len + canvas_length),
            )
        return masks

    def __call__(
        self,
        canvas_ids: mx.array,
        cache=None,
        self_conditioning_logits: Optional[mx.array] = None,
        self_conditioning_embeddings: Optional[mx.array] = None,
        masks=None,
    ):
        h = self._embed_canvas(
            canvas_ids,
            self_conditioning_logits,
            self_conditioning_embeddings,
        )
        cache = cache or [None] * len(self.layers)
        if masks is None:
            masks = self._make_masks(h.shape[0], h.shape[1], cache)
        offset = _cache_offset(cache[0]) if cache else 0

        for layer, c in zip(self.layers, cache):
            h = layer(h, masks.get(layer.layer_type), c, decoder=True, offset=offset)
        return self.norm(h)


class EncoderLayerScalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_scalar = mx.ones((1,))


class EncoderLanguageModel(nn.Module):
    """Owns only the encoder's per-layer scalars; the transformer weights are the
    decoder's own (tied), reached through a weakref so the module tree doesn't
    walk — and double-count — the decoder."""

    def __init__(self, decoder: "DecoderModel"):
        super().__init__()
        self._decoder_ref = weakref.ref(decoder)
        self.layers = [EncoderLayerScalar() for _ in decoder.layers]


class EncoderModel(nn.Module):
    def __init__(self, config: TextArgs, decoder: DecoderModel):
        super().__init__()
        self.config = config
        self.language_model = EncoderLanguageModel(decoder)
        # weakref.ref, not proxy: see EncoderLanguageModel.
        self._decoder_ref = weakref.ref(decoder)

    @property
    def decoder(self):
        return self._decoder_ref()

    def make_cache(self):
        return [
            (
                KVCache()
                if layer_type == "full_attention"
                else RotatingKVCache(max_size=self.config.sliding_window)
            )
            for layer_type in self.config.layer_types
        ]

    def __call__(
        self,
        input_ids: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is None:
            input_embeddings = self.decoder.embed_tokens(input_ids)
        h = input_embeddings * self.decoder.embed_scale
        if cache is None:
            cache = [None] * len(self.decoder.layers)

        for i, (layer, c) in enumerate(zip(self.decoder.layers, cache)):
            mask = create_attention_mask(
                h,
                c,
                window_size=(
                    self.config.sliding_window
                    if layer.layer_type == "sliding_attention"
                    else None
                ),
            )
            h = layer(
                h,
                mask,
                c,
                layer_scalar=self.language_model.layers[i].layer_scalar,
            )
        return self.decoder.norm(h)


class DiffusionGemmaModel(nn.Module):
    def __init__(self, config: TextArgs):
        super().__init__()
        self.decoder = DecoderModel(config)
        self.encoder = EncoderModel(config, self.decoder)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.text_args = TextArgs.from_dict(args.text_config)
        self.vocab_size = self.text_args.vocab_size
        self.canvas_length = args.canvas_length
        self.generation_config = args.generation_config or {}
        self.model = DiffusionGemmaModel(self.text_args)
        self._softcap = make_compiled_softcap(
            float(self.text_args.final_logit_softcapping)
        )

    def _logits(self, h):
        return self._softcap(self.model.decoder.embed_tokens.as_linear(h))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        h = self.model.encoder(inputs, cache=cache, input_embeddings=input_embeddings)
        return self._logits(h)

    # --- block-diffusion protocol (driven by mlx_lm.diffusion_generate) ------

    @property
    def prefers_logits_self_conditioning(self) -> bool:
        return self.model.decoder.prefers_logits_self_conditioning

    def diffusion_prepare_self_conditioning(self) -> Optional[mx.array]:
        return self.model.decoder.diffusion_prepare_self_conditioning()

    def diffusion_self_conditioning(
        self,
        processed_logits: mx.array,
        embedding_weight: Optional[mx.array],
    ) -> mx.array:
        return self.model.decoder.diffusion_self_conditioning(
            processed_logits,
            embedding_weight,
        )

    def diffusion_extend_cache(self, input_ids: mx.array, *, cache):
        """Encoder pass: append a prompt chunk or an accepted block to the cache."""
        self.model.encoder(input_ids, cache=cache)
        return cache

    def diffusion_decoder_masks(self, canvas_ids: mx.array, cache):
        return self.model.decoder._make_masks(
            canvas_ids.shape[0], canvas_ids.shape[1], cache
        )

    def diffusion_decoder_logits(
        self,
        canvas_ids: mx.array,
        cache=None,
        self_conditioning: Optional[mx.array] = None,
        masks=None,
    ):
        kwargs = (
            {"self_conditioning_logits": self_conditioning}
            if self.prefers_logits_self_conditioning
            else {"self_conditioning_embeddings": self_conditioning}
        )
        h = self.model.decoder(canvas_ids, cache=cache, masks=masks, **kwargs)
        return self._logits(h)

    # --------------------------------------------------------------------------

    @property
    def layers(self):
        return self.model.decoder.layers

    def make_cache(self):
        return self.model.encoder.make_cache()

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "rotary_emb" in k or k == "lm_head.weight":
                continue
            # mlx-unified is text-only: drop the vision tower and its embedder.
            if k.startswith(
                ("model.encoder.vision_tower.", "model.encoder.embed_vision.")
            ):
                continue
            # Encoder transformer weights are tied to the decoder's; the
            # checkpoint only carries the encoder's own per-layer scalars.
            if k.startswith("model.encoder.language_model."):
                if k.endswith(".layer_scalar"):
                    sanitized[k] = v
                continue
            # Fused experts → SwitchGLU. Handles raw HF keys (no suffix),
            # mlx-vlm conversions (.weight), and quantized ones (.scales/.biases
            # split row-wise like the weights).
            if ".experts.gate_up_proj" in k:
                base, _, param = k.partition(".experts.gate_up_proj")
                param = param.lstrip(".") or "weight"
                gate, up = map(mx.contiguous, mx.split(v, 2, axis=-2))
                sanitized[f"{base}.experts.switch_glu.gate_proj.{param}"] = gate
                sanitized[f"{base}.experts.switch_glu.up_proj.{param}"] = up
                continue
            if ".experts.down_proj" in k:
                base, _, param = k.partition(".experts.down_proj")
                param = param.lstrip(".") or "weight"
                sanitized[f"{base}.experts.switch_glu.down_proj.{param}"] = v
                continue
            sanitized[k] = v
        return sanitized

    @property
    def quant_predicate(self):
        def predicate(path, m):
            if not hasattr(m, "to_quantized"):
                return False
            if (
                path.endswith(
                    ("embed_tokens", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
                )
                or ".self_attn." in path
                or "router" in path
            ):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
