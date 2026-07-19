# Copyright © 2026 Apple Inc.
#
# mlx-unified: greedy self-speculative decoding from a baked MTP (nextn) head
# (deepseek_v32 / glm_moe_dsa — GLM-5.2). The head drafts K tokens sequentially
# (cheap: ONE extra decoder layer per token); the main model verifies all K+1 in
# a single batched forward — which is what makes this compose so well with
# expert streaming: the batch's expert loads are shared across every position
# (batch-union), so a round costs barely more disk I/O than a single token.
# Validated recurrence + acceptance against Colibrì (JustVugg/colibri).

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import mlx.core as mx

from .models.base import create_attention_mask
from .models.cache import make_prompt_cache, trim_prompt_cache


@dataclass
class MtpStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    tokens: int = 0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


def _mtp_absorb(model, tokens: mx.array, hiddens: mx.array, mtp_cache) -> None:
    """Append the VERIFIED pairs (token@p+1, true-hidden@p) to the MTP head's KV
    in one batched forward — draft-time entries were chained off the head's own
    hiddens (approximate), so after each verify they're trimmed and replaced with
    exact pairs. tokens (1, n) / hiddens (1, n, D), n >= 1."""
    mtp = model.mtp
    x = mtp.enorm(model.model.embed_tokens(tokens))
    h = mtp.hnorm(hiddens)
    hx = mtp.eh_proj(mx.concatenate([x, h], axis=-1))
    mask = create_attention_mask(hx, mtp_cache[0], return_array=True)
    mtp(hx, mask, mtp_cache)


def mtp_speculative_generate(
    model,
    prompt_ids: List[int],
    *,
    max_tokens: int = 256,
    num_draft: int = 3,
    eos_ids: Optional[set] = None,
    on_tokens: Optional[Callable[[List[int]], None]] = None,
    after_forward: Optional[Callable[[], None]] = None,
) -> tuple[List[int], MtpStats]:
    """Greedy generation with baked-MTP self-speculation.

    Per round: chain `num_draft` head steps (each appends one MTP-KV entry),
    verify [next, d1..dK] in ONE main forward, accept the longest matching
    prefix (+ the verify batch's own token at the first mismatch — the "bonus"),
    then trim the main KV by the rejected suffix and the MTP KV by all K draft
    entries, re-absorbing the n ACCEPTED true pairs. The boundary pair (bonus
    token + its hidden) is deliberately NOT absorbed — the next round's first
    draft step appends it, keeping entry index == position with append-only KV.

    `after_forward` runs after every MAIN forward (prefill + each verify) — the
    expert-streaming hook (dynamic_cache_update). Greedy-only by design: the
    acceptance test is exact token equality (Colibrì-style), not stochastic
    speculative sampling.
    """
    from .generate import wired_limit

    stats = MtpStats()
    with wired_limit(model):
        return _generate(model, prompt_ids, stats, max_tokens, num_draft, eos_ids, on_tokens, after_forward)


def _generate(model, prompt_ids, stats, max_tokens, num_draft, eos_ids, on_tokens, after_forward):
    cache = make_prompt_cache(model)
    hid = model.model(mx.array([prompt_ids]), cache)[:, -1:, :]
    nxt = mx.argmax(model.lm_head(hid), axis=-1)  # (1, 1)
    mx.eval(nxt)
    if after_forward:
        after_forward()

    mtp_cache = model.make_mtp_cache()
    out: List[int] = [nxt.item()]
    if on_tokens:
        on_tokens(out[-1:])
    eos_ids = eos_ids or set()

    while len(out) < max_tokens and out[-1] not in eos_ids:
        K = min(num_draft, max_tokens - len(out))
        # --- draft K tokens by chaining the head off its own hiddens ---
        drafts: List[mx.array] = []
        h, t = hid, nxt
        for _ in range(K):
            dlogits, h = model.mtp_draft_step(t, h, mtp_cache)
            t = mx.argmax(dlogits, axis=-1)
            drafts.append(t)
        # --- verify all K+1 positions in ONE main forward (batch-union I/O) ---
        batch = mx.concatenate([nxt] + drafts, axis=1)  # (1, K+1)
        hids = model.model(batch, cache)
        actual = mx.argmax(model.lm_head(hids), axis=-1)  # (1, K+1)
        mx.eval(actual)
        if after_forward:
            after_forward()
        # --- accept the longest matching prefix ---
        n = 0
        while n < K and drafts[n].item() == actual[0, n].item():
            n += 1
        emitted = [int(actual[0, i].item()) for i in range(n + 1)]
        # Truncate at EOS inside the accepted run.
        for j, tok in enumerate(emitted):
            if tok in eos_ids:
                emitted = emitted[: j + 1]
                n = min(n, j)
                break
        out.extend(emitted)
        if on_tokens:
            on_tokens(emitted)
        stats.rounds += 1
        stats.proposed += K
        stats.accepted += n
        # --- rollback: main KV wrote K+1, keep n+1; MTP KV wrote K drafts ---
        trim_prompt_cache(cache, K - n)
        trim_prompt_cache([mtp_cache], K)
        if n > 0:
            _mtp_absorb(model, actual[:, :n], hids[:, :n, :], mtp_cache)
        hid = hids[:, n : n + 1, :]
        nxt = actual[:, n : n + 1]

    stats.tokens = len(out)
    return out, stats
