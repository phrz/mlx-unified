# Copyright © 2026 Apple Inc.
#
# mlx-unified: prompt-lookup (n-gram) speculative decoding — self-speculation
# with NO draft model and NO extra weights. Drafts are copied from wherever the
# current suffix last appeared in the context (prompt + generated so far); the
# main model verifies all K+1 positions in ONE batched forward. On long-prompt
# workloads (summarize / rewrite / extract / code-edit) the model frequently
# emits spans verbatim from its context, so acceptance is high exactly where
# big-MoE decode is most expensive: every accepted token amortizes the full
# expert-weight read across the verify batch. Same accept/trim bookkeeping as
# mtp_speculative (minus the head/absorb — there is no drafter state at all).

from dataclasses import dataclass
from typing import Callable, List, Optional

import mlx.core as mx

from .models.cache import make_prompt_cache, trim_prompt_cache


@dataclass
class NgramStats:
    rounds: int = 0
    drafted_rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    tokens: int = 0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


def ngram_propose(
    context: List[int],
    *,
    max_draft: int,
    max_ngram: int = 5,
    min_ngram: int = 3,
) -> List[int]:
    """Propose up to max_draft continuation tokens by finding the most recent
    earlier occurrence of the context's current suffix n-gram (longest n first)
    and copying what followed it. Empty when no n-gram recurs — the caller then
    just decodes normally, so a miss costs nothing."""
    L = len(context)
    for n in range(min(max_ngram, L - 1), min_ngram - 1, -1):
        suffix = context[L - n :]
        # Scan backwards for the most recent PRIOR occurrence.
        for start in range(L - n - 1, -1, -1):
            if context[start : start + n] == suffix:
                cont = context[start + n : start + n + max_draft]
                if cont:
                    return cont
                break  # matched at the very end — nothing follows; try shorter n
    return []


def ngram_speculative_generate(
    model,
    prompt_ids: List[int],
    *,
    max_tokens: int = 256,
    num_draft: int = 8,
    max_ngram: int = 4,
    eos_ids: Optional[set] = None,
    on_tokens: Optional[Callable[[List[int]], None]] = None,
    after_forward: Optional[Callable[[], None]] = None,
) -> tuple[List[int], NgramStats]:
    """Greedy generation with prompt-lookup speculation.

    Per round: propose draft tokens from the context's own n-gram recurrences
    (free), verify [next, d1..dK] in ONE main forward, accept the longest
    matching prefix + the bonus token, trim the KV by the rejected suffix.
    Rounds with no n-gram match decode a single token — identical cost to
    plain greedy, so the worst case is ~baseline speed."""
    from .generate import wired_limit

    stats = NgramStats()
    eos_ids = eos_ids or set()
    with wired_limit(model):
        return _generate(
            model, prompt_ids, stats, max_tokens, num_draft, max_ngram,
            eos_ids, on_tokens, after_forward,
        )


def _generate(model, prompt_ids, stats, max_tokens, num_draft, max_ngram, eos_ids, on_tokens, after_forward):
    cache = make_prompt_cache(model)
    hid = model.model(mx.array([prompt_ids]), cache)[:, -1:, :]
    nxt = mx.argmax(model.lm_head(hid), axis=-1)
    mx.eval(nxt)
    if after_forward:
        after_forward()

    context: List[int] = list(prompt_ids)
    out: List[int] = [nxt.item()]
    context.append(out[-1])
    if on_tokens:
        on_tokens(out[-1:])

    while len(out) < max_tokens and out[-1] not in eos_ids:
        budget = max_tokens - len(out)
        drafts = ngram_propose(context, max_draft=min(num_draft, budget), max_ngram=max_ngram)
        K = len(drafts)
        batch = mx.array([[int(nxt.item()), *drafts]] if K else [[int(nxt.item())]])
        hids = model.model(batch, cache)
        actual = mx.argmax(model.lm_head(hids), axis=-1)  # (1, K+1)
        mx.eval(actual)
        if after_forward:
            after_forward()
        n = 0
        while n < K and drafts[n] == actual[0, n].item():
            n += 1
        emitted = [int(actual[0, i].item()) for i in range(n + 1)]
        for j, t in enumerate(emitted):  # truncate at EOS inside the accepted run
            if t in eos_ids:
                emitted = emitted[: j + 1]
                n = min(n, j)
                break
        out.extend(emitted)
        context.extend(emitted)
        if on_tokens:
            on_tokens(emitted)
        stats.rounds += 1
        stats.drafted_rounds += 1 if K else 0
        stats.proposed += K
        stats.accepted += n
        trim_prompt_cache(cache, K - n)
        hid = hids[:, n : n + 1, :]
        nxt = actual[:, n : n + 1]

    stats.tokens = len(out)
    return out, stats
