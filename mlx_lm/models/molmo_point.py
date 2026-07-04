# Copyright © 2026 Apple Inc.
#
# AllenAI MolmoPoint text decoder + in-model point-token generation, ported from
# mlx-vlm's mlx_vlm/models/molmo_point (MIT © Blaizzy/mlx-vlm contributors).
#
# The text backbone is molmo2 (fused att_proj, per-head Q/K RMSNorm, split vocab
# table) with two departures: the lm_head is split into base + extension tables
# (checkpoint keys lm_head.{output_embeddings,new_output_embeddings}, covering the
# full 151936+128 vocab), and the small point-predictor / build_vit_embedding
# weights are KEPT so the extended point vocabulary is computed in-model:
#
#   V_ext = V_total + n_patches (+1 no-more-points) + n_subpatches + 9 locations
#
# Point logits are dot-products between point-predictor projections of the
# CURRENT step's pre-final-layernorm hidden state and keys cached at prefill:
# patch_k from the prefill's pre-ln hidden states at image-token positions
# (captured inside this model's own forward — chunk-safe, accumulated by absolute
# cache offset) and subpatch_k from gathered ViT features. Everything the vision
# side must provide arrives via set_visual_state on the inner model (the fork's
# side-state convention, see mlx_lm/multimodal.py); mlx-vlm's molmo_point
# get_input_embeddings stashes exactly these tensors on its Model._image_cache.
#
# During decode, mlx-lm feeds raw sampled ids back: ids >= V_total are mapped
# in-model to their embedding composition (patch -> wte(patch_token_id) + the
# patch's connector image feature; subpatch -> build_vit_embedding of the chosen
# ViT sub-patch feature, replacing the embedding; location -> plain
# wte(location_token_id)) and the reference's MolmoPointLogitProcessor ordering
# mask is applied to the last position in-model, so a plain mlx-lm sampler stays
# correct. Text-only (no side state) is byte-identical to molmo2 semantics.

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs, create_attention_mask
from .molmo2 import Molmo2Block, Molmo2Embedding


@dataclass
class ModelArgs(BaseModelArgs):
    # Checkpoints nest the decoder config under text_config (model_type
    # "molmo2_text") and carry vit_config/adapter_config plus the point-prediction
    # fields at the top level.
    model_type: str = "molmo_point"
    text_config: Optional[dict] = None
    vit_config: Optional[dict] = None
    vision_config: Optional[dict] = None
    adapter_config: Optional[dict] = None
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    additional_vocab_size: int = 128
    layer_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None
    max_position_embeddings: int = 37376
    qkv_bias: bool = False
    # Vision-derived dims: the point predictor consumes ViT features of width
    # vit_hidden_size * len(vit_layers) (defaults from MolmoPoint-8B).
    vit_hidden_size: int = 1152
    vit_layers: List[int] = field(default_factory=lambda: [-3, -9])
    # Point-prediction config (top-level in config.json).
    patch_embed_dim: int = 512
    patch_location: Optional[str] = "3x3"
    no_more_points_class: bool = True
    layer_norm_x: bool = True
    norm_logits: bool = True
    mask_patches: Optional[str] = "always"
    mask_subpatches: Optional[str] = "inference"
    mask_repeats: Optional[str] = "inference"
    token_prediction_rotary: Optional[str] = "one_d"
    token_prediction_rotary_theta: Optional[float] = 50000.0
    # Special token ids (in the wte extension table).
    patch_token_id: int = 151947
    subpatch_token_id: int = 151948
    location_token_id: int = 151949

    def __post_init__(self):
        if self.text_config:
            for key, value in self.text_config.items():
                if key != "model_type" and key in self.__dataclass_fields__:
                    setattr(self, key, value)
        vit = self.vit_config or self.vision_config
        if vit and "hidden_size" in vit:
            self.vit_hidden_size = vit["hidden_size"]
        if self.adapter_config and "vit_layers" in self.adapter_config:
            self.vit_layers = list(self.adapter_config["vit_layers"])

    @property
    def vit_feature_dim(self):
        return self.vit_hidden_size * len(self.vit_layers)

    @property
    def total_vocab_size(self):
        return self.vocab_size + self.additional_vocab_size


class MolmoPointPatchRope(nn.Module):
    """Dedicated 1D rotary over image-patch positions for patch q/k vectors."""

    def __init__(self, theta: float, dims: int):
        super().__init__()
        self._inv_freq = 1.0 / (
            theta ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        )

    def rotate_half(self, x: mx.array) -> mx.array:
        B, hs = x.shape
        x = x.reshape(B, 2, hs // 2)
        return mx.concatenate([-x[:, 1, :], x[:, 0, :]], axis=-1)

    def __call__(self, x: mx.array, position_ids: mx.array) -> mx.array:
        position_ids = position_ids.astype(mx.float32)
        x_float = x.astype(mx.float32)
        freqs = position_ids[:, None] * self._inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        out = (x_float * mx.cos(emb)) + (self.rotate_half(x_float) * mx.sin(emb))
        return out.astype(x.dtype)


class MolmoPointPadWithLearnedVector(nn.Module):
    """Appends the learned no-more-points class embedding as an extra patch row."""

    def __init__(self, dims: int):
        super().__init__()
        self.vector = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        vector = mx.broadcast_to(
            self.vector[None, None, :], (B, 1, self.vector.shape[0])
        )
        return mx.concatenate([x, vector.astype(x.dtype)], axis=1)


class PointPredictor(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        llm_dim = args.hidden_size
        patch_embed_dim = args.patch_embed_dim
        vit_dim = args.vit_feature_dim

        if args.layer_norm_x:
            self.x_norm = nn.RMSNorm(llm_dim, eps=args.layer_norm_eps)
        else:
            self.x_norm = None

        if args.token_prediction_rotary == "one_d":
            theta = args.token_prediction_rotary_theta or args.rope_theta
            self.patch_rotary = MolmoPointPatchRope(theta, patch_embed_dim)
        else:
            self.patch_rotary = None

        self.patch_q = nn.Linear(llm_dim, patch_embed_dim)
        self.patch_k = nn.Linear(llm_dim, patch_embed_dim)
        self.subpatch_q = nn.Linear(llm_dim, patch_embed_dim)
        self.subpatch_k = nn.Linear(vit_dim, patch_embed_dim)
        self.add_no_point_class_embed = MolmoPointPadWithLearnedVector(patch_embed_dim)

        if args.patch_location == "3x3":
            self.subpatch_loc_k = nn.Linear(llm_dim, 9)
        else:
            self.subpatch_loc_k = None


class GeneratedTokenBounds:
    """Extended-vocab id ranges: [patch | no-more-points | subpatch | location]."""

    def __init__(
        self, vocab_size, n_patches, n_subpatches, n_locations, no_more_points_class
    ):
        self.n_locations = n_locations
        self.n_patches = n_patches
        self.n_subpatches = n_subpatches
        self.vocab_size = vocab_size
        if no_more_points_class:
            self.no_more_points_token_id = vocab_size + n_patches
        else:
            self.no_more_points_token_id = -1
        self.patch_start = vocab_size
        self.patch_end_without_no_more_points = vocab_size + n_patches
        self.patch_end = vocab_size + n_patches + int(no_more_points_class)
        self.subpatch_start = self.patch_end
        self.subpatch_end = self.subpatch_start + n_subpatches
        self.location_start = self.subpatch_end
        self.location_end = self.subpatch_end + n_locations


class MolmoPointLogitProcessor:
    """Enforce valid point token generation order (patch -> subpatch -> location).

    Applied in-model to the last position's logits so any plain mlx-lm sampler
    stays correct. Mask is built host-side over python ints (no device sync)."""

    def __init__(
        self,
        bounds: GeneratedTokenBounds,
        prevent_repeats,
        force_patch_sorted,
        force_subpatch_sorted,
    ):
        self.bounds = bounds
        self.prevent_repeats = prevent_repeats
        self.force_patch_sorted = force_patch_sorted
        self.force_subpatch_sorted = force_subpatch_sorted

    def __call__(self, generated_ids, last_token, vocab_size):
        b = self.bounds
        NEG_INF = np.float32(-1e9)
        mask = np.zeros(vocab_size, dtype=np.float32)
        ids = generated_ids

        skip = 2 if b.n_locations else 1
        last_patch = None
        last_subpatch = None
        # Check ALL tokens for no_more_points (not subject to skip)
        no_more_points = any(tok == b.no_more_points_token_id for tok in ids)
        # Only scan history up to skip for patch/subpatch tracking
        for i in range(len(ids) - skip):
            tok = ids[i]
            if b.patch_start <= tok < b.patch_end:
                last_patch = tok
            elif b.subpatch_start <= tok < b.subpatch_end:
                last_subpatch = tok

        if no_more_points:
            mask[b.patch_start : b.location_end] = NEG_INF
        elif last_token < b.patch_start or last_token >= b.subpatch_end:
            # Can generate text or a patch, but not subpatch/location
            mask[b.subpatch_start : b.location_end] = NEG_INF
            if self.force_patch_sorted and last_patch is not None:
                # Patches must be in sorted order
                mask[b.patch_start : last_patch] = NEG_INF
            if (
                self.prevent_repeats
                and self.force_subpatch_sorted
                and last_subpatch is not None
                and last_subpatch == (b.subpatch_end - 1)
            ):
                # Last subpatch was at max — selecting same patch would force
                # a repeat since sorted order has no room for a new subpatch
                if last_patch is not None:
                    mask[last_patch] = NEG_INF
        elif b.patch_start <= last_token < b.patch_end:
            # After a patch, must select a subpatch
            mask[: b.subpatch_start] = NEG_INF
            mask[b.subpatch_end :] = NEG_INF
            if (
                self.force_subpatch_sorted
                and last_patch == last_token
                and last_subpatch is not None
            ):
                if self.prevent_repeats:
                    mask[b.subpatch_start : last_subpatch + 1] = NEG_INF
                else:
                    mask[b.subpatch_start : last_subpatch] = NEG_INF
        elif b.n_locations and b.subpatch_start <= last_token < b.subpatch_end:
            # After a subpatch, must select a location
            mask[: b.location_start] = NEG_INF
            mask[b.location_end :] = NEG_INF

        return mx.array(mask)


class MolmoPointTransformer(nn.Module):
    """molmo2's decoder body plus the visual side-state hooks and a pre-ln tap
    (patch keys/queries are projections of the PRE-final-layernorm hidden state)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.wte = Molmo2Embedding(
            args.vocab_size, args.additional_vocab_size, args.hidden_size
        )
        self.blocks = [Molmo2Block(args) for _ in range(args.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_eps)
        # Underscore attr so MLX's module walker never registers it (no weights).
        self._point_state = None

    def set_visual_state(
        self,
        *,
        token_pooling: mx.array,  # (B, P, S) int32 ViT-patch indices, -1 = padding
        vit_features: mx.array,  # (B, P, S, vit_dim) gathered ViT features
        image_features: mx.array,  # (n_image_tokens, llm_dim) connector outputs
        image_token_offsets: mx.array,  # (B,) per-example offsets into image_features
        is_image_token: mx.array,  # (B, L) bool over the FULL prompt
        is_indexable_image_token: mx.array,  # (B, L) bool
    ) -> None:
        """Install the vision side state for a point-capable prompt. All six
        tensors come from mlx-vlm's molmo_point Model._image_cache (stashed by
        its get_input_embeddings); the prompt embeddings themselves flow through
        the ordinary input_embeddings injection."""
        self._point_state = {
            "token_pooling": token_pooling.astype(mx.int32),
            "vit_features": vit_features,
            "image_features": image_features.reshape(-1, image_features.shape[-1]),
            "image_token_offsets": image_token_offsets,
            "is_image_token": is_image_token,
            "is_indexable_image_token": is_indexable_image_token,
            "generated_ids": [],  # plain ints for the ordering mask (no sync)
            "last_patch_id": None,
            "patch_k": None,
            "patch_k_mask": None,
            "patch_k_buf": None,
        }

    def reset_visual_state(self) -> None:
        self._point_state = None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        return_pre_ln: bool = False,
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

        out = self.ln_f(h)
        if return_pre_ln:
            return out, h
        return out


class ExtendedLmHead(nn.Module):
    """lm_head split into base + extension tables, concatenated at matmul time —
    matches the checkpoint's lm_head.{output_embeddings,new_output_embeddings}
    split (same pattern as Molmo2Embedding for wte)."""

    def __init__(self, num_embeddings: int, num_new_embeddings: int, dims: int):
        super().__init__()
        self.output_embeddings = mx.zeros((num_embeddings, dims))
        self.new_output_embeddings = mx.zeros((num_new_embeddings, dims))

    def __call__(self, x: mx.array) -> mx.array:
        w = mx.concatenate(
            [self.output_embeddings, self.new_output_embeddings], axis=0
        )
        return x @ w.T


class MolmoPointLanguageModel(nn.Module):
    """Namespace container matching checkpoint keys lm.model.* / lm.lm_head.*."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = MolmoPointTransformer(args)
        self.lm_head = ExtendedLmHead(
            args.vocab_size, args.additional_vocab_size, args.hidden_size
        )


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.lm = MolmoPointLanguageModel(args)
        self.point_predictor = PointPredictor(args)
        self.build_vit_embedding = nn.Linear(
            args.vit_feature_dim, args.hidden_size, bias=True
        )

    @property
    def model(self):
        # The fork addresses side-state hooks as model.model.set_visual_state /
        # reset_visual_state (see server.py); the inner module here is `lm.model`
        # to keep checkpoint keys loadable without remapping.
        return self.lm.model

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        state = self.lm.model._point_state
        if state is None:
            # Text-only: byte-identical to molmo2 semantics (extended-width head).
            if inputs is not None and input_embeddings is None:
                if mx.any(inputs >= self.args.total_vocab_size).item():
                    raise ValueError(
                        "molmo_point: token id >= the model vocabulary "
                        f"({self.args.total_vocab_size}) — extended point-token ids "
                        "are only meaningful while visual side state is installed "
                        "(model.model.set_visual_state)."
                    )
            h = self.lm.model(inputs, cache, input_embeddings)
            return self.lm.lm_head(h)
        if input_embeddings is not None:
            return self._prefill_forward(inputs, input_embeddings, cache, state)
        return self._generate_forward(inputs, cache, state)

    # --- prefill ----------------------------------------------------------

    def _prefill_forward(self, inputs, input_embeddings, cache, state):
        """Prompt forward with visual state: run the decoder, capture patch keys
        from the pre-ln hidden state at image-token positions (accumulated by
        absolute cache offset, so mlx-lm's chunked prefill — including its
        final-token step — composes exactly), and pad the logits with dummy
        extended entries so the width matches decode steps."""
        self._ensure_derived(state)
        offset = self._cache_offset(cache)
        h, pre_ln = self.lm.model(
            inputs, cache, input_embeddings, return_pre_ln=True
        )
        logits = self.lm.lm_head(h)

        self._capture_patch_keys(pre_ln, offset, state)

        bounds = state["bounds"]
        B, S, _ = logits.shape
        total_extra = bounds.location_end - bounds.patch_start
        dummy = mx.full((B, S, total_extra), -100000.0, dtype=logits.dtype)
        return mx.concatenate([logits, dummy], axis=-1)

    def _cache_offset(self, cache) -> int:
        if cache is None or cache[0] is None:
            return 0
        offset = cache[0].offset
        if isinstance(offset, int):
            return offset
        return offset.item() if offset.ndim == 0 else offset[0].item()

    def _ensure_derived(self, state):
        """Mask-only derivations, computed once per set_visual_state: index maps
        between prompt positions and pooled-patch rows, rotary positions,
        subpatch keys from ViT features, and the extended-vocab bounds."""
        if "bounds" in state:
            return
        args = self.args
        pp = self.point_predictor
        tp = state["token_pooling"]
        B, P, S = tp.shape

        vit_features_mask = tp >= 0  # (B, P, S)
        image_features_mask = mx.any(vit_features_mask, axis=-1)  # (B, P)

        is_img = np.array(state["is_image_token"]).astype(bool)  # (B, L)
        is_idx = np.array(state["is_indexable_image_token"]).astype(bool)
        L = is_img.shape[1]
        img_flat = np.where(is_img.reshape(-1))[0]  # flat (B*L) image positions
        valid_flat = np.where(np.array(image_features_mask).reshape(-1))[0]
        if len(img_flat) != len(valid_flat):
            raise ValueError(
                "molmo_point: image-token positions and valid pooled-patch rows "
                f"disagree ({len(img_flat)} vs {len(valid_flat)}) — the visual "
                "side state is inconsistent with the prompt."
            )

        # The k-th image token (flat prompt order) owns pooled row valid_flat[k];
        # its rotary position is the running count of INDEXABLE image tokens.
        is_idx_flat = is_idx.reshape(-1).astype(np.int64)
        idx_cum = np.cumsum(is_idx_flat) - 1
        img_rank = np.cumsum(is_img.reshape(-1).astype(np.int64)) - 1

        pos_ids_flat = np.zeros(B * P, dtype=np.int32)
        pos_ids_flat[valid_flat] = idx_cum[img_flat]
        patch_k_mask_flat = np.zeros(B * P, dtype=bool)
        patch_k_mask_flat[valid_flat] = is_idx_flat[img_flat].astype(bool)

        vit_dim = state["vit_features"].shape[-1]
        vit_sparse = state["vit_features"].reshape(-1, S, vit_dim)[
            mx.array(valid_flat.astype(np.int32))
        ]

        state.update(
            seq_len=L,
            vit_features_mask=vit_features_mask,
            image_pos_ids=mx.array(pos_ids_flat.reshape(B, P)),
            subpatch_k=pp.subpatch_k(state["vit_features"]),
            vit_sparse=vit_sparse,
            _np={
                "is_img": is_img,
                "img_rank": img_rank,
                "idx_cum": idx_cum,
                "valid_flat": valid_flat,
                "patch_k_mask": patch_k_mask_flat,
                "captured": np.zeros(len(img_flat), dtype=bool),
            },
            bounds=GeneratedTokenBounds(
                vocab_size=args.total_vocab_size,
                n_patches=P,
                n_subpatches=S,
                n_locations=9 if args.patch_location else 0,
                no_more_points_class=args.no_more_points_class,
            ),
        )

    def _capture_patch_keys(self, pre_ln, offset, state):
        """Accumulate patch keys for image-token positions inside this forward's
        window [offset, offset + S): project the pre-ln hidden rows through
        patch_k (after x_norm), rotate by the running indexable-image-token
        position, and scatter into the pooled-row buffer. Finalizes (adding the
        no-more-points class row) once every image position has been seen."""
        if state["patch_k"] is not None:
            return
        npd = state["_np"]
        pp = self.point_predictor
        B, S, D = pre_ln.shape
        L = state["seq_len"]
        if offset >= L:
            return
        window = npd["is_img"][:, offset : offset + S]
        bidx, pos = np.where(window)
        if len(bidx):
            abs_flat = bidx * L + (offset + pos)
            rank = npd["img_rank"][abs_flat]
            fresh = ~npd["captured"][rank]
            bidx, pos, abs_flat, rank = (
                bidx[fresh],
                pos[fresh],
                abs_flat[fresh],
                rank[fresh],
            )
        if len(bidx):
            local_flat = mx.array((bidx * S + pos).astype(np.int32))
            rows = pre_ln.reshape(-1, D)[local_flat]
            x_norm = (
                pp.x_norm(rows) if pp.x_norm is not None else rows / math.sqrt(D)
            )
            pk = pp.patch_k(x_norm)
            if pp.patch_rotary is not None:
                pk = pp.patch_rotary(
                    pk, mx.array(npd["idx_cum"][abs_flat].astype(np.int32))
                )
            buf = state["patch_k_buf"]
            if buf is None:
                P = state["token_pooling"].shape[1]
                Bs = state["token_pooling"].shape[0]
                buf = mx.zeros(
                    (Bs * P, self.args.patch_embed_dim), dtype=pre_ln.dtype
                )
            targets = mx.array(npd["valid_flat"][rank].astype(np.int32))
            state["patch_k_buf"] = buf.at[targets].add(pk.astype(buf.dtype))
            npd["captured"][rank] = True

        if npd["captured"].all():
            Bs, P = state["token_pooling"].shape[:2]
            patch_k = state["patch_k_buf"].reshape(Bs, P, -1)
            patch_k_mask = mx.array(npd["patch_k_mask"].reshape(Bs, P))
            if self.args.no_more_points_class:
                patch_k = pp.add_no_point_class_embed(patch_k)
                patch_k_mask = mx.concatenate(
                    [patch_k_mask, mx.ones((Bs, 1), dtype=mx.bool_)], axis=1
                )
            state["patch_k"] = patch_k
            state["patch_k_mask"] = patch_k_mask

    # --- decode -----------------------------------------------------------

    def _embed_generated(self, inputs, state):
        """Decode-step embeddings for raw sampled ids (extended ids included):
        map extended ids back to their special token for the wte lookup, ADD the
        patch's connector image feature for patch ids, and REPLACE the embedding
        with build_vit_embedding(ViT sub-patch feature) for subpatch ids."""
        bounds = state["bounds"]
        ids = inputs.astype(mx.int32)
        B = ids.shape[0]

        is_patch = (ids >= bounds.patch_start) & (
            ids < bounds.patch_end_without_no_more_points
        )
        is_no_more_points = ids == bounds.no_more_points_token_id
        is_subpatch = (ids >= bounds.subpatch_start) & (ids < bounds.subpatch_end)
        is_location = (ids >= bounds.location_start) & (ids < bounds.location_end)

        input_patch_ids = mx.where(is_patch, ids - bounds.patch_start, -1)
        input_subpatch_ids = mx.where(is_subpatch, ids - bounds.subpatch_start, -1)

        decoded_ids = mx.where(
            is_patch | is_no_more_points, self.args.patch_token_id, ids
        )
        decoded_ids = mx.where(is_subpatch, self.args.subpatch_token_id, decoded_ids)
        decoded_ids = mx.where(is_location, self.args.location_token_id, decoded_ids)

        x = self.lm.model.wte(decoded_ids)

        # Patch id: add the patch's connector image feature (per-example offset).
        any_patch = mx.any(is_patch).item()
        if any_patch:
            feats = state["image_features"]
            offsets = state["image_token_offsets"]
            for b in range(B):
                pid = int(input_patch_ids[b, 0].item())
                if (
                    0
                    <= pid
                    < bounds.patch_end_without_no_more_points - bounds.patch_start
                ):
                    flat_idx = pid + int(offsets[b].item())
                    x = x.at[b, 0].add(feats[flat_idx])

        # Subpatch id: replace the embedding with the chosen ViT sub-patch
        # feature (of the LAST selected patch) through build_vit_embedding.
        any_subpatch = mx.any(is_subpatch).item()
        if any_subpatch:
            vit_sparse = state["vit_sparse"]
            offsets = state["image_token_offsets"]
            for b in range(B):
                spid = int(input_subpatch_ids[b, 0].item())
                if spid >= 0 and state["last_patch_id"] is not None:
                    lpid = int(state["last_patch_id"][b].item())
                    flat_pid = lpid + int(offsets[b].item())
                    embedded = self.build_vit_embedding(
                        vit_sparse[flat_pid, spid : spid + 1]
                    )
                    x = x.at[b, 0:1].add(embedded - x[b, 0:1])

        return x, is_subpatch, input_patch_ids, any_patch

    def _generate_forward(self, inputs, cache, state):
        """Decode step with point prediction: extended logits are dot-products of
        point-predictor projections of THIS step's pre-ln hidden state against
        the prefill-cached patch keys / ViT-feature subpatch keys, with the
        reference's ordering mask applied to the last position."""
        self._ensure_derived(state)
        if state["patch_k"] is None:
            raise ValueError(
                "molmo_point: decode step before the visual prefill completed — "
                "the prompt (with input_embeddings) must run through the model "
                "so patch keys can be captured from its hidden states."
            )
        bounds = state["bounds"]
        pp = self.point_predictor
        args = self.args
        D = args.hidden_size
        B = inputs.shape[0]

        # Track sampled ids as plain ints for the ordering mask.
        for i in range(inputs.shape[1]):
            state["generated_ids"].append(int(inputs[0, i].item()))

        x, is_subpatch, input_patch_ids, any_patch = self._embed_generated(
            inputs, state
        )

        h, pre_ln = self.lm.model(None, cache, x, return_pre_ln=True)
        logits = self.lm.lm_head(h)

        x_norm = pp.x_norm(pre_ln) if pp.x_norm is not None else pre_ln / math.sqrt(D)

        # Patch logits: rotate the query by the last selected patch's position.
        image_q = pp.patch_q(x_norm)
        if pp.patch_rotary is not None and state["last_patch_id"] is not None:
            pos_ids = state["image_pos_ids"]
            lpid = state["last_patch_id"]
            rotate_by = pos_ids[
                mx.arange(B), mx.clip(lpid.squeeze(-1), 0, pos_ids.shape[1] - 1)
            ]
            rotate_by = mx.where(lpid.squeeze(-1) >= 0, rotate_by, 0)
            image_q_flat = image_q.reshape(-1, image_q.shape[-1])
            image_q_flat = pp.patch_rotary(
                image_q_flat, mx.clip(rotate_by, a_min=0, a_max=None)
            )
            image_q = image_q_flat.reshape(B, -1, image_q.shape[-1])

        dots = image_q @ state["patch_k"].transpose(0, 2, 1)
        if args.norm_logits:
            dots = dots / math.sqrt(dots.shape[-1])
        patch_logits = mx.where(state["patch_k_mask"][:, None, :], dots, -100000.0)

        # Move the patch_token_id probability mass onto the argmax patch slot.
        B_, S_, _ = logits.shape
        patch_token_logits = logits[
            :, :, args.patch_token_id : args.patch_token_id + 1
        ]
        logits = logits.at[:, :, args.patch_token_id].add(
            -100000.0 - logits[:, :, args.patch_token_id]
        )
        n_patches = patch_logits.shape[-1]
        selected_patches = mx.argmax(patch_logits, axis=-1)  # (B, S)
        indices = mx.arange(n_patches)[None, None, :]
        is_selected = indices == selected_patches[:, :, None]
        argmax_patch_logits = mx.where(
            is_selected,
            patch_token_logits,
            mx.full((B_, S_, n_patches), -100000.0, dtype=logits.dtype),
        )

        # Subpatch logits (only live right after a patch id was consumed).
        n_subpatches = state["token_pooling"].shape[-1]
        subpatch_logits = mx.full(
            (B_, S_, n_subpatches), -100000.0, dtype=logits.dtype
        )
        if any_patch:
            subpatch_point_q = pp.subpatch_q(
                x_norm.squeeze(1) if S_ == 1 else x_norm[:, -1:].squeeze(1)
            )
            batch_idx = mx.arange(B_)
            spk = state["subpatch_k"][
                batch_idx,
                mx.clip(
                    input_patch_ids.squeeze(1), 0, state["subpatch_k"].shape[1] - 1
                ),
            ]
            sp_logits = mx.sum(subpatch_point_q[:, None, :] * spk, axis=-1)
            if args.norm_logits:
                sp_logits = sp_logits / math.sqrt(state["patch_k"].shape[-1])
            sp_mask = state["vit_features_mask"][
                batch_idx,
                mx.clip(
                    input_patch_ids.squeeze(1),
                    0,
                    state["vit_features_mask"].shape[1] - 1,
                ),
            ]
            sp_logits = mx.where(sp_mask, sp_logits, -100000.0)
            subpatch_logits = sp_logits[:, None, :].astype(logits.dtype)

        logits = logits.at[:, :, args.subpatch_token_id].add(
            -100000.0 - logits[:, :, args.subpatch_token_id]
        )

        # Location logits (3x3 refinement; width follows the bounds so prefill
        # and decode always agree).
        n_locations = bounds.location_end - bounds.subpatch_end
        location_logits = mx.full(
            (B_, S_, n_locations), -100000.0, dtype=logits.dtype
        )
        if n_locations and mx.any(is_subpatch).item():
            location_logits = pp.subpatch_loc_k(pre_ln).astype(logits.dtype)

        logits = logits.at[:, :, args.location_token_id].add(
            -100000.0 - logits[:, :, args.location_token_id]
        )

        logits = mx.concatenate(
            [logits, argmax_patch_logits, subpatch_logits, location_logits], axis=-1
        )

        # Ordering mask on the last position (host-side ints, no device sync).
        if state["generated_ids"]:
            processor = MolmoPointLogitProcessor(
                bounds=bounds,
                prevent_repeats=args.mask_repeats in ("all", "inference"),
                force_patch_sorted=args.mask_patches in ("always", "inference"),
                force_subpatch_sorted=args.mask_subpatches
                in ("always", "inference"),
            )
            lp_mask = processor(
                state["generated_ids"], state["generated_ids"][-1], logits.shape[-1]
            )
            last_logits = logits[:, -1, :] + lp_mask
            logits = mx.concatenate([logits[:, :-1, :], last_logits[:, None, :]], axis=1)

        if mx.any(input_patch_ids >= 0).item():
            previous = (
                state["last_patch_id"]
                if state["last_patch_id"] is not None
                else mx.full((B, 1), -1, dtype=mx.int32)
            )
            state["last_patch_id"] = mx.where(
                input_patch_ids == -1, previous, input_patch_ids
            )

        return logits

    def sanitize(self, weights):
        new_weights = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue
            # Raw HF checkpoints: model.transformer.* / model.point_predictor.* /
            # model.build_vit_embedding.* / model.{vit,connector}.* / top-level
            # lm_head.{output_embeddings,new_output_embeddings}.
            if k.startswith("model."):
                k = k[len("model.") :]
            if k.startswith("lm_head."):
                k = "lm." + k
            if k.startswith("transformer."):
                k = "lm.model." + k[len("transformer.") :]
            # mlx-vlm conversions arrive as lm.model.* / lm.lm_head.* /
            # point_predictor.* / build_vit_embedding.* already in place; the
            # vision tower and connector are the only weights stripped (the
            # point predictor is small and computes the extended logits here).
            if k.startswith(("vit.", "vision_model.", "vision_backbone.", "connector.")):
                continue
            new_weights[k] = v
        return new_weights

    @property
    def layers(self):
        return self.lm.model.blocks
