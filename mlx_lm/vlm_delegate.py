# Copyright © 2026 Apple Inc.
#
# mlx-unified: serving delegation to mlx-vlm. Model families whose GENERATION
# MODE mlx-lm cannot express are loaded and generated wholesale by mlx-vlm —
# its processor renders the chat template, its prepare_inputs handles images,
# and its own engine runs the denoising loop — so a delegated checkpoint served
# by mlx_lm.server behaves exactly like the pinned mlx_vlm.server (including
# the opt-in x_stream_draft_blocks extension). Text-only use of mlx_lm must
# never require mlx_vlm: every mlx_vlm import in this module is lazy.

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional

# model_type → served wholesale via mlx-vlm. These families either use a
# generation mode mlx-lm does not implement (diffusion_gemma) or have their
# authoritative language implementation only in mlx-vlm (GLM-5-Next and
# Qwen3.8-Flash-Next's experimental Qwen4 architecture).
DELEGATED_VLM_FAMILIES = {"diffusion_gemma", "glm5_next", "qwen4_exp"}

# Runway inspects this marker without importing MLX/Metal. Version 1 means the
# delegated lane forwards cache tenants, explicit token checkpoints, and
# sliding expiry into an mlx-vlm APC implementation with the same contract.
DELEGATED_APC_CONTRACT_VERSION = 1


def is_delegated_model_type(model_type: Optional[str]) -> bool:
    """True when checkpoints of this model_type are served via mlx-vlm."""
    return model_type in DELEGATED_VLM_FAMILIES


@dataclass
class DelegatedResponse:
    """One committed token (or one opt-in draft) from the delegated engine.

    Field-compatible with diffusion_generate.DiffusionResponse where the
    server consumes it, so _serve_delegated_diffusion reads like
    _serve_diffusion. Draft responses carry only `draft_blocks` (the
    not-yet-committed blocks' current text, "░" per masked position) and must
    be excluded from token/usage accounting downstream."""

    text: str
    token: int
    # mlx-vlm's diffusion engine does not compute logprobs — always 0.0.
    logprob: float
    block_complete: bool
    finish_reason: Optional[str]  # "stop" | "length" | None
    draft_blocks: Optional[List[str]] = None
    cached_tokens: int = 0


def _result_logprob(result) -> float:
    logprobs = getattr(result, "logprobs", None)
    token = getattr(result, "token", None)
    if logprobs is None or token is None:
        return 0.0
    try:
        return float(logprobs[token].item())
    except (IndexError, TypeError):
        return 0.0


def _adapt_results(results, *, diffusion: bool = True) -> Iterator[DelegatedResponse]:
    """mlx-vlm GenerationResults → DelegatedResponses.

    Mirrors mlx_vlm.server's _diffusion_block_chunks semantics — drafts are
    deduped (repeats whose decoded text did not change are dropped) and always
    precede their block's committed text; committed text is byte-identical
    with drafts on or off — but keeps mlx_lm's per-token accounting: one
    response per committed token, `block_complete` on each block's last token.
    mlx-vlm emits a SEPARATE block-boundary marker and a final finish result;
    both are folded into the preceding token (finalize leftovers included),
    the way mlx_lm.diffusion_generate emits them natively, so downstream
    len(tokens) accounting matches the non-delegated diffusion lane."""
    pending: Optional[DelegatedResponse] = None
    last_draft_blocks: Optional[List[str]] = None
    for result in results:
        if getattr(result, "is_draft", False):
            if not result.draft_blocks or result.draft_blocks == last_draft_blocks:
                continue
            if pending is not None:
                # A new canvas is denoising: the previous block's held-back
                # last token must flush first so drafts precede only their
                # OWN block's content.
                yield pending
                pending = None
            last_draft_blocks = result.draft_blocks
            yield DelegatedResponse(
                "", 0, 0.0, False, None, list(result.draft_blocks), 0
            )
            continue
        if result.finish_reason is not None:
            if pending is None:
                # Zero committed tokens (e.g. immediate eos) — one terminal
                # response so the server can still flush a finish_reason.
                pending = DelegatedResponse(
                    result.text,
                    int(result.token or 0),
                    _result_logprob(result),
                    True,
                    result.finish_reason,
                    cached_tokens=int(getattr(result, "cached_tokens", 0) or 0),
                )
            else:
                pending.text += result.text
                pending.block_complete = True
                pending.finish_reason = result.finish_reason
                pending.cached_tokens = int(
                    getattr(result, "cached_tokens", pending.cached_tokens) or 0
                )
            yield pending
            return
        if diffusion and result.diffusion_block_complete:
            if pending is not None:
                pending.block_complete = True
            continue
        if pending is not None:
            yield pending
        pending = DelegatedResponse(
            result.text,
            int(result.token),
            _result_logprob(result),
            not diffusion,
            None,
            cached_tokens=int(getattr(result, "cached_tokens", 0) or 0),
        )
    if pending is not None:  # engine exhausted without a finish result
        yield pending


class VlmDelegate:
    """A checkpoint loaded and generated by mlx-vlm, adapted for mlx_lm.server."""

    def __init__(
        self,
        model,
        processor,
        tokenizer,
        *,
        apc_manager=None,
        apc_directory: Optional[Path] = None,
    ):
        self.model = model
        self.processor = processor
        self.config = model.config
        self.tokenizer = tokenizer
        self.processor_tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.is_diffusion = getattr(self.config, "model_type", None) == "diffusion_gemma"
        self.apc_manager = apc_manager
        self.apc_directory = apc_directory

    @staticmethod
    def _is_explicit_breakpoint(value: Any) -> bool:
        return value is True or (
            isinstance(value, dict) and value.get("mode") == "explicit"
        )

    @classmethod
    def _messages_through_breakpoint(cls, messages: List[dict]):
        prefix = None
        for message_index, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, list):
                if cls._is_explicit_breakpoint(message.get("prompt_cache_breakpoint")):
                    prefix = [dict(m) for m in messages[: message_index + 1]]
                continue
            for part_index, part in enumerate(content):
                if not isinstance(part, dict) or not cls._is_explicit_breakpoint(
                    part.get("prompt_cache_breakpoint")
                ):
                    continue
                last = dict(message)
                last["content"] = [
                    dict(p) if isinstance(p, dict) else p
                    for p in content[: part_index + 1]
                ]
                prefix = [dict(m) for m in messages[:message_index]] + [last]
        return prefix

    def render_chat(self, messages: List[dict], tools=None, **template_kwargs):
        """OpenAI chat messages → (formatted prompt, image payloads), exactly
        as mlx_vlm.server's chat endpoint prepares them: image_url payloads
        pulled from the user content parts, text flattened, and the template
        applied through mlx-vlm's prompt_utils (which places the model
        family's image tokens)."""
        from mlx_vlm.prompt_utils import apply_chat_template, extract_text_from_content

        images = []
        processed = []
        for message in messages:
            msg = {k: v for k, v in message.items() if k != "content"}
            content = message.get("content")
            if isinstance(content, list):
                if message.get("role") == "user":
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        part_type = part.get("type")
                        if part_type == "input_image":
                            images.append(part["image_url"])
                        elif part_type == "image_url":
                            images.append(part["image_url"]["url"])
                msg["content"] = extract_text_from_content(content)
            else:
                msg["content"] = content
            processed.append(msg)

        add_generation_prompt = template_kwargs.pop("add_generation_prompt", True)
        prompt = apply_chat_template(
            self.processor,
            self.config,
            processed,
            add_generation_prompt=add_generation_prompt,
            num_images=len(images),
            tools=tools,
            **template_kwargs,
        )
        return prompt, images

    def explicit_checkpoint_len(
        self,
        messages: List[dict],
        full_inputs: dict,
        *,
        tools=None,
        **template_kwargs,
    ) -> Optional[int]:
        """Token boundary selected by the last explicit content marker.

        The prefix is rendered without an assistant-generation suffix and must
        match the already-rendered request byte-for-token. A template whose
        prefix depends on later messages therefore fails closed instead of
        warming a subtly different cache.
        """
        prefix_messages = self._messages_through_breakpoint(messages)
        if prefix_messages is None:
            return None
        prefix_prompt, prefix_images = self.render_chat(
            prefix_messages,
            tools=tools,
            add_generation_prompt=False,
            **template_kwargs,
        )
        prefix_inputs = self.prepare(prefix_prompt, prefix_images)
        prefix_ids = [int(t) for t in prefix_inputs["input_ids"].reshape(-1).tolist()]
        full_ids = [int(t) for t in full_inputs["input_ids"].reshape(-1).tolist()]
        if not prefix_ids or len(prefix_ids) >= len(full_ids):
            return None
        if full_ids[: len(prefix_ids)] != prefix_ids:
            return None
        return len(prefix_ids)

    def prepare(self, prompt: str, images: List[Any]) -> dict:
        """Tokenize + image-preprocess through mlx-vlm's prepare_inputs —
        mirrors mlx_vlm.server's _cpu_preprocess, including its
        add_special_tokens rule (image payloads may be urls, data URIs or raw
        base64; prepare_inputs decodes them itself)."""
        from mlx_vlm.utils import prepare_inputs

        add_special_tokens = (
            getattr(self.processor, "chat_template", None) is None
            if getattr(self.config, "model_type", None)
            in ("gemma3", "gemma3n", "gemma4", "gemma4_unified")
            else True
        )
        return prepare_inputs(
            self.processor,
            images=images,
            prompts=prompt,
            image_token_index=getattr(self.config, "image_token_index", None),
            add_special_tokens=add_special_tokens,
        )

    def stream(
        self,
        inputs: dict,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        presence_context_size: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        frequency_context_size: Optional[int] = None,
        seed: Optional[int] = None,
        stop_words: Optional[List[str]] = None,
        prefill_step_size: Optional[int] = None,
        draft_blocks: bool = False,
        apc_tenant: Optional[str] = None,
        apc_checkpoint_len: Optional[int] = None,
        apc_ttl_seconds: Optional[float] = None,
    ) -> Iterator[DelegatedResponse]:
        """Drive mlx-vlm's authoritative generator and adapt its results."""

        if apc_ttl_seconds is not None:
            if self.apc_manager is None:
                raise ValueError("prompt cache is disabled for this process")
            coordinator = self.apc_manager.coordinator(
                getattr(self.model, "language_model", self.model)
            )
            if not coordinator.is_checkpoint:
                raise ValueError(
                    "prompt-cache TTL is unsupported for this delegated architecture"
                )

        input_ids = inputs.get("input_ids")
        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids[None]
        tokenizer = self.processor_tokenizer
        if hasattr(tokenizer, "stopping_criteria"):
            tokenizer.stopping_criteria.reset(getattr(self.config, "eos_token_id", None))
        if self.is_diffusion:
            from mlx_vlm.generate.diffusion import stream_diffusion_generate

            skip_special_token_ids = set(
                getattr(tokenizer, "all_special_ids", None) or []
            )
            results = stream_diffusion_generate(
                self.model,
                self.processor,
                tokenizer,
                input_ids,
                inputs.get("pixel_values"),
                inputs.get("attention_mask"),
                max_tokens=max_tokens,
                temperature=temperature,
                skip_special_token_ids=skip_special_token_ids,
                diffusion_draft_blocks=draft_blocks,
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
            )
        else:
            from mlx_vlm.generate import stream_generate

            forwarded = {
                k: v
                for k, v in inputs.items()
                if k not in {"input_ids", "pixel_values", "attention_mask"}
            }
            results = stream_generate(
                self.model,
                self.processor,
                "",
                input_ids=input_ids,
                pixel_values=inputs.get("pixel_values"),
                mask=inputs.get("attention_mask"),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                presence_penalty=presence_penalty,
                presence_context_size=presence_context_size,
                frequency_penalty=frequency_penalty,
                frequency_context_size=frequency_context_size,
                seed=seed,
                eos_tokens=stop_words or None,
                prefill_step_size=prefill_step_size,
                apc_manager=self.apc_manager,
                apc_tenant=apc_tenant,
                apc_checkpoint_len=apc_checkpoint_len,
                apc_ttl_seconds=apc_ttl_seconds,
                **forwarded,
            )
        try:
            yield from _adapt_results(results, diffusion=self.is_diffusion)
        finally:
            results.close()

    def close(self) -> None:
        if self.apc_manager is not None:
            self.apc_manager.close()
            self.apc_manager = None
        if self.apc_directory is not None:
            shutil.rmtree(self.apc_directory, ignore_errors=True)
            self.apc_directory = None


def _repair_diffusion_gemma_processor():
    """Compat shim for the pinned mlx-vlm on transformers ≥5.12.

    transformers' ProcessorMixin discovers a processor's modality attributes
    by inspecting __init__'s NAMED parameters; DiffusionGemma4Processor masks
    its parent's (image_processor, tokenizer, ...) behind *args/**kwargs, so
    get_attributes() comes back empty, ProcessorMixin rejects
    `image_processor` at construction, and mlx-vlm's AutoProcessor patch
    silently falls back to the bare tokenizer — dropping image support.
    Restore an introspectable signature on the class (in-process only) until
    the fork carries the fix; a no-op once it does."""
    from mlx_vlm.models.diffusion_gemma import processing_diffusion_gemma as pdg

    cls = pdg.DiffusionGemma4Processor
    get_attributes = getattr(cls, "get_attributes", None)
    if get_attributes is None or get_attributes():
        return  # older transformers (no introspection) or fixed upstream
    original_init = cls.__init__

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        original_init(
            self,
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    cls.__init__ = __init__


def _repair_glm5_next_namespaces():
    """Map converted GLM-5 container and forget-gate tensor names.

    Keep strict checking for all other tensor names.
    """
    from mlx_vlm.models.glm5_next.glm5_next import Model

    if getattr(Model, "_mlx_unified_vision_namespace", False):
        return
    original_sanitize = Model.sanitize
    original_quantization_path_aliases = getattr(
        Model, "quantization_path_aliases", None
    )

    def sanitize(self, weights):
        remapped = {}
        for key, value in weights.items():
            if key.startswith("vision_tower."):
                key = "vision_model." + key[len("vision_tower.") :]
            for projection in ("f_a_proj", "f_b_proj"):
                source = f".self_attn.{projection}."
                if source in key:
                    key = key.replace(
                        source, f".self_attn.forget_gate.{projection}."
                    )
            remapped[key] = value
        return original_sanitize(self, remapped)

    def quantization_path_aliases(self, path):
        aliases = []
        if original_quantization_path_aliases is not None:
            aliases.extend(original_quantization_path_aliases(self, path))
        checkpoint_path = path
        for projection in ("f_a_proj", "f_b_proj"):
            checkpoint_path = checkpoint_path.replace(
                f".self_attn.forget_gate.{projection}",
                f".self_attn.{projection}",
            )
        if checkpoint_path != path:
            aliases.append(checkpoint_path)
            if checkpoint_path.startswith("language_model."):
                aliases.append(checkpoint_path[len("language_model.") :])
        return tuple(dict.fromkeys(aliases))

    Model.sanitize = sanitize
    Model.quantization_path_aliases = quantization_path_aliases
    Model._mlx_unified_vision_namespace = True


def _disable_incompatible_glm5_next_fusion(model):
    """Keep GLM's fused input projection only for identical quant layouts."""
    disabled = 0
    for layer in getattr(model, "layers", ()):
        attention = getattr(layer, "self_attn", None)
        forget_gate = getattr(attention, "forget_gate", None)
        if attention is None or forget_gate is None:
            continue
        modules = (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            forget_gate.f_a_proj,
            attention.g_a_proj,
            attention.b_proj,
        )
        signatures = {
            (
                type(module),
                tuple(module.weight.shape[1:]),
                getattr(module, "group_size", None),
                getattr(module, "bits", None),
                hasattr(module, "scales"),
            )
            for module in modules
        }
        if len(signatures) > 1:
            attention.fuse_in = False
            disabled += 1
    return disabled


def load_delegate(
    model_path,
    *,
    prompt_cache_bytes: Optional[int] = None,
    prompt_cache_disk_bytes: Optional[int] = None,
) -> VlmDelegate:
    """Load a delegated checkpoint with mlx_vlm.utils.load (model + processor)."""
    try:
        from mlx_vlm.utils import load as vlm_load
    except ImportError as e:
        raise ValueError(
            "This model family is served through mlx-vlm, which is not "
            f"installed (pip install 'mlx-lm[vision]'): {e}"
        )
    _repair_diffusion_gemma_processor()
    try:
        import json

        with open(Path(model_path) / "config.json") as config_file:
            model_type = json.load(config_file).get("model_type")
    except (OSError, ValueError):
        model_type = None
    if model_type == "glm5_next":
        _repair_glm5_next_namespaces()
        # GLM conversion checkpoints can retain one MTP-only mlp_layer_types
        # entry beyond num_hidden_layers. The MLX model consumes the base-layer
        # schedule correctly, but AutoProcessor -> AutoTokenizer needlessly
        # reparses and rejects the full model config. Load the strict MLX model
        # and its self-describing fast tokenizer independently.
        from mlx_vlm.utils import StoppingCriteria
        from mlx_vlm.utils import load_model as vlm_load_model
        from mlx_vlm.tokenizer_utils import load_tokenizer as load_vlm_tokenizer
        from transformers import PreTrainedTokenizerFast

        model = vlm_load_model(Path(model_path))
        _disable_incompatible_glm5_next_fusion(model)
        processor = PreTrainedTokenizerFast.from_pretrained(model_path)
        detokenizer_class = load_vlm_tokenizer(
            Path(model_path), return_tokenizer=False
        )
        processor.detokenizer = detokenizer_class(processor)
        eos_token_ids = getattr(model.config, "eos_token_id", None)
        if eos_token_ids is None:
            eos_token_ids = getattr(model.config.text_config, "eos_token_id", None)
        processor.stopping_criteria = StoppingCriteria(
            eos_token_ids,
            processor,
            additional_eos_token_ids=getattr(
                processor, "additional_eos_token_ids", ()
            ),
        )
    else:
        model, processor = vlm_load(str(model_path))
    from .utils import load_tokenizer

    text_config = getattr(model.config, "text_config", None)
    eos_token_ids = getattr(model.config, "eos_token_id", None)
    if eos_token_ids is None and text_config is not None:
        eos_token_ids = getattr(text_config, "eos_token_id", None)
    tokenizer = load_tokenizer(model_path, eos_token_ids=eos_token_ids)

    apc_manager = None
    apc_directory = None
    if prompt_cache_bytes is not None and prompt_cache_bytes > 1:
        from mlx_vlm.apc import apc_disk_namespace, from_env

        overrides = {
            "enabled": True,
            "max_resident_bytes": int(prompt_cache_bytes),
        }
        if prompt_cache_disk_bytes is not None and prompt_cache_disk_bytes > 0:
            apc_directory = Path(tempfile.mkdtemp(prefix="mlx_vlm_apc_"))
            overrides.update(
                disk_path=str(apc_directory),
                disk_max_gb=prompt_cache_disk_bytes / (1 << 30),
            )
        namespace = apc_disk_namespace(str(model_path))
        try:
            apc_manager = from_env(namespace, overrides=overrides)
        except Exception:
            if apc_directory is not None:
                shutil.rmtree(apc_directory, ignore_errors=True)
            raise
    return VlmDelegate(
        model,
        processor,
        tokenizer,
        apc_manager=apc_manager,
        apc_directory=apc_directory,
    )
