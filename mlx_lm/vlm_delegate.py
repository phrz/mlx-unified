# Copyright © 2026 Apple Inc.
#
# mlx-unified: serving delegation to mlx-vlm. Model families whose GENERATION
# MODE mlx-lm cannot express are loaded and generated wholesale by mlx-vlm —
# its processor renders the chat template, its prepare_inputs handles images,
# and its own engine runs the denoising loop — so a delegated checkpoint served
# by mlx_lm.server behaves exactly like the pinned mlx_vlm.server (including
# the opt-in x_stream_draft_blocks extension). Text-only use of mlx_lm must
# never require mlx_vlm: every mlx_vlm import in this module is lazy.

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

# model_type → served wholesale via mlx-vlm. These families either use a
# generation mode mlx-lm does not implement (diffusion_gemma) or have their
# authoritative language implementation only in mlx-vlm (GLM-5-Next and
# Qwen3.8-Flash-Next's experimental Qwen4 architecture).
DELEGATED_VLM_FAMILIES = {"diffusion_gemma", "glm5_next", "qwen4_exp"}


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
            yield DelegatedResponse("", 0, 0.0, False, None, list(result.draft_blocks))
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
                )
            else:
                pending.text += result.text
                pending.block_complete = True
                pending.finish_reason = result.finish_reason
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
        )
    if pending is not None:  # engine exhausted without a finish result
        yield pending


class VlmDelegate:
    """A checkpoint loaded and generated by mlx-vlm, adapted for mlx_lm.server."""

    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.config = model.config
        self.tokenizer = tokenizer
        self.processor_tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.is_diffusion = getattr(self.config, "model_type", None) == "diffusion_gemma"

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

        prompt = apply_chat_template(
            self.processor,
            self.config,
            processed,
            num_images=len(images),
            tools=tools,
            **template_kwargs,
        )
        return prompt, images

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
    ) -> Iterator[DelegatedResponse]:
        """Drive mlx-vlm's authoritative generator and adapt its results."""

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
                **forwarded,
            )
        try:
            yield from _adapt_results(results, diffusion=self.is_diffusion)
        finally:
            results.close()


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


def load_delegate(model_path) -> VlmDelegate:
    """Load a delegated checkpoint with mlx_vlm.utils.load (model + processor)."""
    try:
        from mlx_vlm.utils import load as vlm_load
    except ImportError as e:
        raise ValueError(
            "This model family is served through mlx-vlm, which is not "
            f"installed (pip install 'mlx-lm[vision]'): {e}"
        )
    _repair_diffusion_gemma_processor()
    model, processor = vlm_load(str(model_path))
    from .utils import load_tokenizer

    text_config = getattr(model.config, "text_config", None)
    eos_token_ids = getattr(model.config, "eos_token_id", None)
    if eos_token_ids is None and text_config is not None:
        eos_token_ids = getattr(text_config, "eos_token_id", None)
    tokenizer = load_tokenizer(model_path, eos_token_ids=eos_token_ids)
    return VlmDelegate(model, processor, tokenizer)
