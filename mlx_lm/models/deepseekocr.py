# Copyright © 2026 Apple Inc.
#
# DeepSeek-OCR text body: DeepSeek-V2-style MoE (routed + shared experts) driven
# through plain Llama-style GQA attention — these checkpoints set
# qk_nope_head_dim == 0, which rules out MLA entirely. Ported from mlx-vlm's
# models/deepseekocr/language.py (MIT © Blaizzy / mlx-vlm contributors).
# deepseekocr_2 reuses this language model verbatim (MODEL_REMAPPING).

from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU

# Non-language components of the composite OCR checkpoints, in both the raw HF
# layout (model.<name>.*) and the mlx-vlm-converted layout (<name>.*).
# "view_seperator" is the raw checkpoints' typo for "view_separator".
_VISION_KEYS = (
    "vision_model",
    "sam_model",
    "projector",
    "image_newline",
    "view_separator",
    "view_seperator",
)


@dataclass
class ModelArgs(BaseModelArgs):
    # OCR configs flatten every language_config field at the top level, so the
    # checkpoint config parses directly.
    model_type: str = "deepseekocr"
    vocab_size: int = 129280
    hidden_size: int = 1280
    intermediate_size: int = 6848
    moe_intermediate_size: int = 896
    num_hidden_layers: int = 12
    num_attention_heads: int = 10
    num_key_value_heads: Optional[int] = None
    n_shared_experts: Optional[int] = 2
    n_routed_experts: Optional[int] = 64
    routed_scaling_factor: float = 1.0
    num_experts_per_tok: int = 6
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    scoring_func: str = "softmax"
    topk_method: str = "greedy"
    qk_nope_head_dim: int = 0

    def __post_init__(self):
        if self.qk_nope_head_dim != 0:
            raise ValueError(
                "deepseekocr implements only qk_nope_head_dim == 0 "
                "(Llama-style attention); MLA configs belong to deepseek_v2"
            )
        if self.topk_method != "greedy":
            raise ValueError(f"Unsupported topk method: {self.topk_method}")
        if self.scoring_func != "softmax":
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        attention_bias = args.attention_bias
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = initialize_rope(
            head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

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
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = intermediate_size or args.intermediate_size

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))

    def __call__(self, x):
        scores = mx.softmax(x @ self.weight.T, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(scores, inds, axis=-1)
        return inds, scores * self.routed_scaling_factor


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_shared_experts = args.n_shared_experts
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts
        )
        self.gate = MoEGate(args)
        if args.n_shared_experts is not None:
            self.shared_experts = MLP(
                args, args.moe_intermediate_size * args.n_shared_experts
            )

    def __call__(self, x) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = (
            MoE(args)
            if (
                args.n_routed_experts is not None
                and layer_idx >= args.first_k_dense_replace
                and layer_idx % args.moe_layer_freq == 0
            )
            else MLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class DeepseekOCRModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

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
        self.model = DeepseekOCRModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        return self.lm_head(out)

    def sanitize(self, weights):
        new_weights = {}
        for k, v in weights.items():
            # mlx-vlm conversions prefix the text model with language_model.
            k = k.removeprefix("language_model.")
            parts = k.split(".")
            root = parts[1] if parts[0] == "model" and len(parts) > 1 else parts[0]
            if root in _VISION_KEYS or "rotary_emb.inv_freq" in k:
                continue
            new_weights[k] = v

        # Raw HF checkpoints store routed experts individually; stack for SwitchGLU.
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for m in ("gate_proj", "down_proj", "up_proj"):
                for q in ("weight", "scales", "biases"):
                    if f"{prefix}.mlp.experts.0.{m}.{q}" in new_weights:
                        to_join = [
                            new_weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{q}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        new_weights[f"{prefix}.mlp.switch_mlp.{m}.{q}"] = mx.stack(
                            to_join
                        )
        return new_weights

    @property
    def layers(self):
        return self.model.layers
