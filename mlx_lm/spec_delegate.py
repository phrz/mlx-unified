# Copyright © 2026 Apple Inc.
#
# mlx-unified: drafter-family speculative decoding via mlx-vlm. The drafter
# implementations (MTP assistant, dflash, eagle3) and their round-loop math live
# in the pinned mlx_vlm.speculative package; this module is the lazy bridge that
# loads a drafter and runs its rounds against an mlx-lm-loaded TARGET, whose
# model class implements the speculative hook contract (speculative_verify_hidden
# / speculative_logits_from_hidden / rollback_speculative_cache — see
# docs/PORTING-DRAFTERS.md). Text-only use of mlx_lm must never require mlx_vlm:
# every mlx_vlm import in this module is lazy.

from typing import Any, Optional, Tuple

_INSTALL_HINT = (
    "drafter-family speculative decoding (--draft-kind) needs the mlx-vlm "
    "package — install the unified fork's vision extra: `mlx-lm[vision]`."
)

# v1 scope: the MTP assistant round loop is target-hook-complete for gemma4.
# dflash/eagle3 need capture_layer_ids prefill hooks the mlx-lm side doesn't
# implement yet — reject them clearly rather than fail deep in a round.
SUPPORTED_KINDS = {"mtp", "eagle3"}


def _speculative():
    try:
        from mlx_vlm import speculative
    except ImportError as e:
        raise RuntimeError(_INSTALL_HINT) from e
    return speculative


def resolve_drafter_kind(path_or_repo: str, kind: Optional[str] = None) -> Optional[str]:
    """The drafter FAMILY for a checkpoint: explicit kind, else auto-detected
    from its HF model_type. None when the checkpoint is not a known drafter at
    all (i.e. it's a plain LM → classic same-tokenizer draft path)."""
    from mlx_vlm.speculative.drafters import (  # lazy; ImportError → not available
        DRAFTER_KIND_BY_MODEL_TYPE,
        _peek_drafter_model_type,
    )
    from mlx_vlm.utils import get_model_path

    if kind is not None:
        return kind
    model_type = _peek_drafter_model_type(get_model_path(path_or_repo))
    return DRAFTER_KIND_BY_MODEL_TYPE.get(model_type or "")


def is_drafter_checkpoint(path_or_repo: str) -> bool:
    """True when the checkpoint's model_type names a drafter family — used by the
    server to route --draft-model between the classic draft path and this one."""
    try:
        return resolve_drafter_kind(path_or_repo) is not None
    except ImportError:
        return False  # no mlx_vlm installed → only the classic path exists


def load_drafter(path_or_repo: str, kind: Optional[str] = None) -> Tuple[Any, str]:
    """Load a drafter checkpoint via mlx-vlm's registry → (drafter, kind).

    The kind gate runs BEFORE any weights load — rejecting an unsupported
    family costs a config peek, not a checkpoint load.
    """
    _speculative()  # surface the install hint before deeper imports
    from mlx_vlm.speculative.drafters import load_drafter as vlm_load_drafter

    resolved = resolve_drafter_kind(path_or_repo, kind)
    if resolved not in SUPPORTED_KINDS:
        raise ValueError(
            f"draft-kind {resolved!r} is not supported by the unified server yet "
            f"(supported: {sorted(SUPPORTED_KINDS)}); run it under mlx_vlm.server."
        )
    drafter, resolved = vlm_load_drafter(path_or_repo, resolved)
    return drafter, resolved


def mtp_rounds(
    model: Any,
    drafter: Any,
    prompt_cache: list,
    hidden: Any,
    shared_kv_states: dict,
    *,
    first_bonus: int,
    max_tokens: int,
    sampler,
    greedy_sampling: bool,
):
    """The single-request MTP round-loop generator, from mlx_vlm.speculative."""
    from mlx_vlm.speculative.mtp import _mtp_rounds

    return _mtp_rounds(
        model,
        drafter,
        prompt_cache,
        hidden,
        shared_kv_states,
        first_bonus=first_bonus,
        max_tokens=max_tokens,
        sampler=sampler,
        greedy_sampling=greedy_sampling,
    )


def eagle3_rounds(
    model: Any,
    drafter: Any,
    prompt_cache: list,
    hidden: Any,
    *,
    prompt_tokens=None,
    first_bonus: int,
    max_tokens: int,
    sampler,
    greedy_sampling: bool,
):
    """The single-request EAGLE3 round-loop generator, from mlx_vlm.speculative.
    `hidden` is the CONCATENATED aux hidden of the prompt (capture_layer_ids
    forward); the target must implement rollback_speculative_cache."""
    from mlx_vlm.speculative.eagle3 import _eagle3_rounds

    return _eagle3_rounds(
        model,
        drafter,
        prompt_cache,
        hidden,
        prompt_tokens=prompt_tokens,
        first_bonus=first_bonus,
        max_tokens=max_tokens,
        sampler=sampler,
        greedy_sampling=greedy_sampling,
    )


def eagle3_capture_layer_ids(drafter: Any) -> list:
    """The target layer ids this drafter wants captured."""
    from mlx_vlm.speculative.eagle3 import _eagle3_capture_layer_ids

    return _eagle3_capture_layer_ids(drafter)
