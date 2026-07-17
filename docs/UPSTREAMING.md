# Pending upstream candidates

Changes we carry in our forks that would benefit the UPSTREAM projects
(ml-explore/mlx-lm, Blaizzy/mlx-vlm) — as opposed to mlx-unified-only work
(vision unification, delegation, prompt-cache tiers, server extensions), which
is ours and stays here.

**Policy: nothing is PR'd anywhere without Paul's explicit go-ahead, and only
once the work has proven itself locally.** This file is the queue, not a plan
of record. Status meanings: `pending` = candidate, not sent; `declined-for-now`
= considered and deliberately held; `superseded` = upstream fixed it their way.

## ml-explore/mlx-lm

| Change | Our commits | Status | Notes |
|---|---|---|---|
| KV-cache quantization on sliding-window models: `maybe_quantize_kv_cache` skips Rotating/BatchRotating caches (bounded at window_size — ~nothing to win) instead of raising `RotatingKVCache Quantization NYI`; unbounded full-attention layers still convert | `d055908` | pending | Upstream still hard-fails `--kv-bits` on every gemma-class model. Small, self-contained, easily their style. Includes tests. |
| Quantized-donor dispatch for KV-shared layers: gemma4 shared layers consume the donor's post-update K/V — when the donor cache is quantized those are packed triples, and SDPA must dispatch on the donor's cache type (threaded as `shared_kv_cache`), not the layer's own `None` | `d055908` (same commit) | pending | Only meaningful together with the row above. |
| Drafter-family speculative decoding at the `stream_generate`/server level (`--draft-kind`, drafter auto-detection by model_type, `drafter_generate_step`) | `0269c38`, `8573cc3`, `c78a7dd` | pending (design conversation, not a direct PR) | Our version bridges to mlx_vlm.speculative — upstream would want it self-contained. LM Studio's stated position is upstream-first (lmstudio-ai/mlx-engine#205: "would need the implementation to be ready in mlx-lm"). Overlaps stalled upstream PRs #1276 (gemma4_assistant class) and the #990 family (baked-MTP self-speculation) — reconcile if those land. |
| gemma4_text speculative hooks (`speculative_verify_hidden` / `speculative_logits_from_hidden` / `rollback_speculative_cache`, shared-KV sink) | `0269c38` | pending | Prerequisite of the row above; mirrors what mlx-vlm's gemma4 already exposes. |
| Server `x_speculative=false` per-request draft opt-out (A/B benchmarking without a reload) | `84608c8` | pending | Tiny; upstream may prefer a differently-named field. |

Already superseded upstream (nothing to send):

- transformers ≥5.13 `AutoTokenizer.register` breakage — upstream fixed via the
  config-class form (#1465); we adopted their form in `c2dd4e3`.

## Blaizzy/mlx-vlm

| Change | Our commits (phrz/mlx-vlm@draft-blocks) | Status | Notes |
|---|---|---|---|
| `MaskedEmbedder` quantized-embedding fix: the gemma4_assistant sparse LM head indexed the tied embedding's raw `weight`, which is bit-packed on 4-bit checkpoints → `[reshape] Cannot reshape array of size … ` crash on the drafter's second token. Fix dequantizes only the gathered top-k rows; raw arrays still accepted; regression test included | `3749636` (rebased: `662b3fd`) | pending | Verified upstream main still has the bug (raw indexing at the same site, 2026-07-16). Every user of the mlx-community `*-assistant-4bit` drafters hits this. Strongest candidate in this file. |
| Opt-in draft-block streaming for block-diffusion models (`x_stream_draft_blocks` → `delta.x_draft_blocks` SSE) | `e40cf82` | pending | Upstream added `draft_blocks` to `StreamingToken` since, so the concept has landed halfway — our wire format may be welcome. |
| `diffusion_gemma` introspectable processor `__init__` (transformers 5.12 introspection silently dropped images) | `bee0256` | declined-for-now | Paul previously declined upstreaming this one. |

## How to maintain this file

When a fork commit fixes something upstream also has, add a row (change,
commits, status, why upstream wants it). When upstream fixes it independently,
move it to "superseded" with their PR number. Before any actual PR: ask Paul.
