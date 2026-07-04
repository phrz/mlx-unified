# Copyright © 2026 Apple Inc.
#
# mlx-unified: block-diffusion generation ("canvas denoising").
# Ported from Blaizzy/mlx-vlm's generate/diffusion.py (MIT, © Blaizzy / mlx-vlm
# contributors), trimmed to mlx_lm.server's needs: batch 1, dynamic KV caches,
# no terminal visualizer, no static cache.
#
# Block-diffusion models don't decode token-by-token. Each iteration the model
# (1) appends the prompt — later, each accepted block — to the KV cache with a
# causal encoder pass, then (2) fills a canvas of `canvas_length` random token
# ids and denoises it with bidirectional decoder passes until the sampler
# accepts every position. Text therefore arrives block-by-block.

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

DEFAULT_MIN_CANVAS_LENGTH = 64
DEFAULT_MAX_DENOISING_STEPS = 48
DEFAULT_CONFIDENCE_THRESHOLD = 0.9

# The protocol a model must implement (plus `canvas_length`, `vocab_size`, and
# `generation_config` attributes) to be driven by stream_diffusion_generate.
BLOCK_DIFFUSION_METHODS = (
    "diffusion_extend_cache",
    "diffusion_decoder_masks",
    "diffusion_decoder_logits",
    "diffusion_prepare_self_conditioning",
    "diffusion_self_conditioning",
)

# model_type → generation family. "block" models are driven by the shared loop
# below. Masked-diffusion families (llada2_moe, nemotron_labs_diffusion) own
# their generate loops in mlx-vlm and would need a family of their own here —
# their samplers infill a mask token in place rather than denoise a canvas.
SUPPORTS_DIFFUSION = {"diffusion_gemma": "block"}


def is_diffusion_model(model: nn.Module) -> bool:
    """True when `model` generates by block diffusion instead of autoregression."""
    return SUPPORTS_DIFFUSION.get(
        getattr(model, "model_type", None)
    ) == "block" and all(
        callable(getattr(model, method, None)) for method in BLOCK_DIFFUSION_METHODS
    )


@dataclass
class DiffusionResponse:
    """One accepted token from a denoised block.

    Tokens arrive in per-block bursts; `text` is the token's detokenized segment
    and `block_complete` marks a block's last token — consumers that want
    block-sized chunks (the server) aggregate `text` until it is set.
    """

    text: str
    token: int
    # Log probability of the token under the final (temperature-scheduled)
    # denoising step's logits — the distribution it was actually accepted from.
    logprob: float
    block_index: int
    block_complete: bool
    finish_reason: Optional[str]  # "stop" | "length" | None
    prompt_tokens: int
    generation_tokens: int
    denoising_steps: int


def _config_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _initialize_canvas(canvas_length: int, vocab_size: int, dtype) -> mx.array:
    return mx.random.randint(0, vocab_size, (1, canvas_length)).astype(dtype)


def _linear_temperature(
    cur_step: int,
    max_denoising_steps: int,
    schedule_config: Dict[str, Any],
) -> float:
    t_min = float(schedule_config.get("t_min", 0.4))
    t_max = float(schedule_config.get("t_max", 0.8))
    return t_min + ((t_max - t_min) * (cur_step / max_denoising_steps))


def _sample_canvas(processed_logits: mx.array, dtype, temperature: float) -> mx.array:
    logits = processed_logits.astype(mx.float32)
    if temperature <= 0:
        return mx.argmax(logits, axis=-1).astype(dtype)
    if temperature != 1.0:
        logits = logits / temperature
    return mx.random.categorical(logits).astype(dtype)


def _token_log_probability(
    processed_logits: mx.array,
    token_ids: mx.array,
) -> mx.array:
    logits = processed_logits.astype(mx.float32)
    token_logits = mx.take_along_axis(
        logits,
        token_ids[..., None],
        axis=-1,
    ).squeeze(-1)
    return token_logits - mx.logsumexp(logits, axis=-1)


def _token_entropy(processed_logits: mx.array) -> mx.array:
    logits = processed_logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    probs = mx.exp(log_probs)
    return -mx.sum(probs * log_probs, axis=-1)


def _confidence_transfer_mask(
    confidence: mx.array,
    unrevealed_mask: mx.array,
    threshold: float,
) -> mx.array:
    """Accept unrevealed positions whose confidence clears the threshold; if none
    does, force the single most confident one so every step makes progress."""
    transfer_mask = unrevealed_mask & (confidence >= threshold)
    has_unrevealed = mx.any(unrevealed_mask, axis=-1)
    has_transfer = mx.any(transfer_mask, axis=-1)
    needs_force = has_unrevealed & (~has_transfer)
    masked_confidence = mx.where(unrevealed_mask, confidence, -mx.inf)
    best_index = mx.argmax(masked_confidence, axis=-1)
    positions = mx.arange(confidence.shape[-1])[None, :]
    forced = (positions == best_index[:, None]) & needs_force[:, None]
    return transfer_mask | forced


def _entropy_transfer_mask(entropy: mx.array, entropy_bound: float) -> mx.array:
    """Accept the largest low-entropy subset whose cumulative entropy (beyond its
    own maximum) stays within the bound."""
    sorted_indices = mx.argsort(entropy, axis=-1)
    sorted_entropy = mx.take_along_axis(entropy, sorted_indices, axis=-1)
    cumulative_entropy = mx.cumsum(sorted_entropy, axis=-1)
    cumulative_maximum_entropy = mx.cummax(sorted_entropy, axis=-1)
    sorted_selection_mask = (
        cumulative_entropy - cumulative_maximum_entropy
    ) <= entropy_bound
    selection_mask = mx.zeros_like(sorted_selection_mask)
    return mx.put_along_axis(
        selection_mask,
        sorted_indices,
        sorted_selection_mask,
        axis=-1,
    )


def _stable_and_confident(
    accepted_canvas: mx.array,
    processed_logits: mx.array,
    history: List[mx.array],
    stopping_config: Optional[Dict[str, Any]],
) -> bool:
    """Early-exit when the canvas hasn't changed for `stability_threshold` steps
    and the model's mean token entropy is below `confidence_threshold`."""
    if stopping_config is None:
        return False

    stability_threshold = int(stopping_config.get("stability_threshold", 1))
    confidence_threshold = float(stopping_config.get("confidence_threshold", 0.005))

    if len(history) == stability_threshold:
        stable = all(
            bool(mx.all(accepted_canvas == canvas).item()) for canvas in history
        )
    else:
        stable = False

    history.append(accepted_canvas)
    if len(history) > stability_threshold:
        history.pop(0)

    if not stable:
        return False

    token_entropy = _token_entropy(processed_logits)
    return bool((mx.mean(token_entropy) < confidence_threshold).item())


def stream_diffusion_generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[List[int], mx.array],
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    sampler: Optional[str] = None,
    threshold: Optional[float] = None,
    max_denoising_steps: Optional[int] = None,
    min_canvas_length: Optional[int] = None,
    max_canvas_length: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: Optional[int] = 2048,
    prompt_progress_callback=None,
) -> Generator[DiffusionResponse, None, None]:
    """Denoise blocks of `canvas_length` tokens, yielding one DiffusionResponse
    per accepted token (in per-block bursts). The final response carries the
    finish_reason ("stop" on an eos token, "length" at max_tokens)."""
    input_ids = prompt if isinstance(prompt, mx.array) else mx.array(prompt)
    if input_ids.ndim == 1:
        input_ids = input_ids[None]
    if input_ids.shape[0] != 1:
        raise ValueError("Diffusion generation only supports batch size 1.")

    generation_config = _config_dict(getattr(model, "generation_config", None))
    model_canvas_length = int(model.canvas_length)
    vocab_size = int(model.vocab_size)
    dtype = input_ids.dtype

    max_new_tokens = int(max_tokens or generation_config.get("max_new_tokens", 256))
    if max_denoising_steps is None:
        max_denoising_steps = int(
            generation_config.get("max_denoising_steps")
            or DEFAULT_MAX_DENOISING_STEPS
        )
    if max_denoising_steps < 1:
        raise ValueError("max_denoising_steps must be a positive integer.")
    max_canvas = min(
        model_canvas_length, int(max_canvas_length or model_canvas_length)
    )
    min_canvas = min(max_canvas, int(min_canvas_length or DEFAULT_MIN_CANVAS_LENGTH))
    if max_canvas <= 0 or min_canvas <= 0:
        raise ValueError("Canvas lengths must be positive integers.")

    if sampler is None:
        sampler = "confidence-threshold"
    if sampler not in ("entropy-bound", "confidence-threshold"):
        raise ValueError(f"Unsupported diffusion sampler: {sampler!r}.")
    if threshold is None:
        threshold = DEFAULT_CONFIDENCE_THRESHOLD
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    sampler_config = _config_dict(generation_config.get("sampler_config"))
    sampler_name = sampler_config.get("_cls_name", "EntropyBoundSamplerConfig")
    if sampler_name != "EntropyBoundSamplerConfig":
        raise NotImplementedError(
            f"Diffusion sampler {sampler_name!r} is not supported yet."
        )
    entropy_bound = float(sampler_config.get("entropy_bound", 0.1))

    temperature_config = _config_dict(
        generation_config.get("linear_temperature_schedule_config")
    )
    if not temperature_config:
        temperature_config = {
            "t_min": generation_config.get("t_min", 0.4),
            "t_max": generation_config.get("t_max", 0.8),
        }

    stopping_config = _config_dict(generation_config.get("diffusion_stopping_config"))
    if not stopping_config:
        stopping_config = {
            key: generation_config[key]
            for key in ("confidence_threshold", "stability_threshold")
            if key in generation_config
        } or None

    eos_token_ids = set(getattr(tokenizer, "eos_token_ids", None) or ())
    config_eos = generation_config.get("eos_token_id")
    if config_eos is not None:
        eos_token_ids.update(
            (config_eos,) if isinstance(config_eos, int) else config_eos
        )

    cache = prompt_cache if prompt_cache is not None else model.make_cache()
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    # Chunked prefill, so long prompts keep progress callbacks (and the server's
    # HTTP keepalives) flowing.
    prompt_length = input_ids.shape[1]
    step = prefill_step_size or prompt_length
    for start in range(0, prompt_length, step):
        model.diffusion_extend_cache(input_ids[:, start : start + step], cache=cache)
        mx.eval([c.state for c in cache])
        if prompt_progress_callback is not None:
            prompt_progress_callback(min(start + step, prompt_length), prompt_length)
    mx.clear_cache()

    generated = 0
    block_index = 0
    total_steps = 0
    finish_reason = None
    last_token = None
    last_logprob = 0.0
    current_canvas = None
    self_conditioning_context = model.diffusion_prepare_self_conditioning()

    while generated < max_new_tokens and finish_reason is None:
        block_index += 1
        if block_index > 1:
            # The previous block was accepted: append it to the causal cache.
            model.diffusion_extend_cache(current_canvas, cache=cache)

        remaining = max_new_tokens - generated
        canvas_length = min(max_canvas, max(remaining, min_canvas))
        current_canvas = _initialize_canvas(canvas_length, vocab_size, dtype)
        draft_reveal_mask = mx.zeros(current_canvas.shape, dtype=mx.bool_)
        draft_canvas = current_canvas
        self_conditioning = None
        masks = model.diffusion_decoder_masks(current_canvas, cache)
        history: List[mx.array] = []

        for cur_step in reversed(range(1, max_denoising_steps + 1)):
            total_steps += 1
            logits = model.diffusion_decoder_logits(
                current_canvas,
                cache=cache,
                self_conditioning=self_conditioning,
                masks=masks,
            )
            logits = logits / _linear_temperature(
                cur_step, max_denoising_steps, temperature_config
            )
            argmax_canvas = mx.argmax(logits, axis=-1).astype(dtype)
            if cur_step == 1:
                break

            denoiser_canvas = (
                argmax_canvas
                if temperature <= 0
                else _sample_canvas(logits, dtype, temperature)
            )

            if sampler == "entropy-bound":
                next_self_conditioning = model.diffusion_self_conditioning(
                    logits, self_conditioning_context
                )
                acceptance_mask = _entropy_transfer_mask(
                    _token_entropy(logits), entropy_bound
                )
                accepted_canvas = mx.where(
                    acceptance_mask, denoiser_canvas, current_canvas
                )
                current_canvas = mx.where(
                    acceptance_mask,
                    accepted_canvas,
                    _initialize_canvas(canvas_length, vocab_size, dtype),
                )
                draft_reveal_mask = acceptance_mask
                draft_canvas = argmax_canvas
            else:
                next_self_conditioning = None
                confidence = mx.exp(_token_log_probability(logits, denoiser_canvas))
                acceptance_mask = _confidence_transfer_mask(
                    confidence, ~draft_reveal_mask, threshold
                )
                accepted_canvas = mx.where(
                    acceptance_mask, denoiser_canvas, draft_canvas
                )
                current_canvas = mx.where(
                    draft_reveal_mask | acceptance_mask,
                    accepted_canvas,
                    _initialize_canvas(canvas_length, vocab_size, dtype),
                )
                draft_reveal_mask = draft_reveal_mask | acceptance_mask
                draft_canvas = mx.where(acceptance_mask, accepted_canvas, draft_canvas)

            if sampler == "confidence-threshold" and bool(
                mx.all(draft_reveal_mask).item()
            ):
                break

            if _stable_and_confident(argmax_canvas, logits, history, stopping_config):
                break

            if next_self_conditioning is None:
                next_self_conditioning = model.diffusion_self_conditioning(
                    logits, self_conditioning_context
                )
            self_conditioning = next_self_conditioning

        # The final step's argmax is the emitted block (mirrors the reference:
        # intermediate acceptance only steers the denoising trajectory).
        current_canvas = argmax_canvas
        logprobs = _token_log_probability(logits, current_canvas)
        mx.eval(current_canvas)
        block_tokens = current_canvas[0].tolist()
        block_logprobs = logprobs[0].tolist()

        for i, token in enumerate(block_tokens):
            last_token = token
            last_logprob = block_logprobs[i]
            if token in eos_token_ids:
                finish_reason = "stop"
                break
            detokenizer.add_token(token)
            generated += 1
            if generated >= max_new_tokens:
                finish_reason = "length"
                break
            yield DiffusionResponse(
                text=detokenizer.last_segment,
                token=token,
                logprob=last_logprob,
                block_index=block_index,
                block_complete=(i == len(block_tokens) - 1),
                finish_reason=None,
                prompt_tokens=prompt_length,
                generation_tokens=generated,
                denoising_steps=total_steps,
            )
        mx.clear_cache()

    detokenizer.finalize()
    yield DiffusionResponse(
        text=detokenizer.last_segment,
        token=last_token,
        logprob=last_logprob,
        block_index=block_index,
        block_complete=True,
        finish_reason=finish_reason or "length",
        prompt_tokens=prompt_length,
        generation_tokens=generated,
        denoising_steps=total_steps,
    )


def diffusion_generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[List[int], mx.array],
    max_tokens: int = 256,
    **kwargs,
) -> str:
    """Generate a complete response by block diffusion and return its text."""
    return "".join(
        response.text
        for response in stream_diffusion_generate(
            model, tokenizer, prompt, max_tokens=max_tokens, **kwargs
        )
    )
