# Copyright © 2026 Apple Inc.
#
# Qwen3-VL-MoE text model: qwen3_vl with sparse-MoE MLPs (selected via the
# shared TextModelArgs MoE fields) plus checkpoint-specific expert-weight
# sanitizing. Multimodal side state (interleaved mrope + deepstack) is
# inherited from qwen3_vl.

from dataclasses import dataclass

from .base import BaseModelArgs
from .qwen3_vl import Model as Qwen3VLModel


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(Qwen3VLModel):

    def sanitize(self, weights):
        weights = super().sanitize(weights)

        # Original HF checkpoints fuse each layer's experts into a single
        # (num_experts, hidden, 2*moe_intermediate) gate_up tensor; split and
        # transpose into SwitchGLU's (num_experts, moe_intermediate, hidden).
        # mlx-vlm conversions already store switch_mlp.* and pass through.
        for l in range(self.language_model.args.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in weights:
                gate_up = weights.pop(gate_up_key)
                mid = gate_up.shape[-1] // 2
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :mid
                ].swapaxes(-2, -1)
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                    ..., mid:
                ].swapaxes(-2, -1)
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.experts.down_proj"
                ).swapaxes(-2, -1)

        return weights
