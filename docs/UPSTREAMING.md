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
| EAGLE3 as a drafter-family kind in the `--draft-kind` path: `spec_delegate` eagle3 bridge + `generate._eagle3_generate_step` (chunked capture-prefill → mlx-vlm `_eagle3_rounds`); plus an optional `draft_block_size` override (`_EAGLE3_BLOCK_OVERRIDE` module var / `RUNWAY_EAGLE3_BLOCK` env) that threads past mlx-vlm's hard min-2 clamp | `10e136b`, `2cf2d7c` | pending (rides with the drafter-family row above) | The generic generate/spec_delegate wiring is upstream-shaped; the actual capture/rollback target hooks live in our `minimax_m3` model (ours — upstream has no minimax_m3). Block override defaults to None (=clamp to 2), which is measured-optimal for sparse-MoE targets (see the mlx-vlm block-size row) — kept only for a future better-trained drafter. Needs the mlx-vlm eagle3 loader row below to be useful. |
| Server `x_speculative=false` per-request draft opt-out (A/B benchmarking without a reload) | `84608c8` | pending | Tiny; upstream may prefer a differently-named field. |
| GLM-5.2 cross-layer DSA indexer sharing (`config.indexer_types`): "shared" layers carry no indexer weights and reuse the previous full layer's top-k, threaded through the decoder loop; `make_cache` skips the indexer sub-cache on shared layers | (uncommitted) | pending | Upstream `glm_moe_dsa`/`deepseek_v32` can't load GLM-5.2 checkpoints at all (285 "missing" indexer params). Matches transformers' `modeling_glm_moe_dsa` semantics. Back-compatible: no `indexer_types` → per-layer indexers as before. |

Already superseded upstream (nothing to send):

- transformers ≥5.13 `AutoTokenizer.register` breakage — upstream fixed via the
  config-class form (#1465); we adopted their form in `c2dd4e3`.

## ml-explore/mlx

| Change | Our commits (local clone ~/Repos/mlx, branch qmv-poc off v0.32.0) | Status | Notes |
|---|---|---|---|
| 3-bit qmv fast path: three aligned uint16 loads instead of six byte loads in the qdot inner loop (same mask-only + pre-scaled-x trick, wider words; v5/v10 word-straddles fold back with the *65536 term). vpt=8 kernels and safe tails untouched | `a72a9e85` | declined-for-now (shelved PoC) | Proven e2e on MiniMax-M3 3_6bit: 27.5 → 28.0 tok/s (+2%, non-overlapping 4-rep ranges); short-chain latency −15%/4-layer gather chain; numerics 3× tighter (fewer pre-scale divisions). Paul's call 2026-07-20: the gain is real but **de minimis against the build effort** (source-build mlx + Metal toolchain, re-sync on every tool reinstall), so we do NOT deploy it locally — it exists as evidence. Would only be worth surfacing upstream where it ships in the official wheel and everyone gets it for free. Negative result worth carrying too: a shift-composite variant regressed 17% — serialized extraction loses the masked-FMA ILP. GitHub fork not created yet (gh token lacks scope); local-only. |

Also measured, no change to carry: batch=1 quantized gemv marginal BW is ~630-720 GB/s
(77-88% of the M3 Ultra's 819 GB/s) across bits — the "3-bit is slow" appearance in naive
per-call benchmarks is a ~200µs per-eval sync floor, not kernel inefficiency. The dependent-chain
decode regime runs ~500 GB/s (latency exposure between chained kernels), which is where the
remaining headroom lives — a scheduling/overlap question, not an unpack one.

## Blaizzy/mlx-vlm

| Change | Our commits (phrz/mlx-vlm@draft-blocks) | Status | Notes |
|---|---|---|---|
| `MaskedEmbedder` quantized-embedding fix: the gemma4_assistant sparse LM head indexed the tied embedding's raw `weight`, which is bit-packed on 4-bit checkpoints → `[reshape] Cannot reshape array of size … ` crash on the drafter's second token. Fix dequantizes only the gathered top-k rows; raw arrays still accepted; regression test included | `3749636` (rebased: `662b3fd`) | pending | Verified upstream main still has the bug (raw indexing at the same site, 2026-07-16). Every user of the mlx-community `*-assistant-4bit` drafters hits this. Strongest candidate in this file. |
| Opt-in draft-block streaming for block-diffusion models (`x_stream_draft_blocks` → `delta.x_draft_blocks` SSE) | `e40cf82` | pending | Upstream added `draft_blocks` to `StreamingToken` since, so the concept has landed halfway — our wire format may be welcome. |
| `diffusion_gemma` introspectable processor `__init__` (transformers 5.12 introspection silently dropped images) | `bee0256` | declined-for-now | Paul previously declined upstreaming this one. |
| EAGLE3 drafter loading for flat `LlamaForCausalLMEagle3` checkpoints: `Eagle3Config.from_dict` lifts top-level llama fields into `transformer_layer_config` when there's no nested config (fixes a phantom d2t map / wrong vocab), plus per-chunk `fc_norm` RMSNorms applied before the aux-hidden concat→fc | `6f650a5` | pending | Upstream eagle3 assumes a nested transformer config and no fc_norm, so the widely-used Inferact EAGLE3 drafters (flat schema, `fc_norm`, ~24k downloads) can't load. Verified end-to-end on MiniMax-M3. Strong candidate alongside the `MaskedEmbedder` fix. |
| Observation (no fork change — do NOT "fix" the clamp): `_eagle3_rounds` hard-clamps block size to 2 when `draft_block_size is None`, and the adaptive-tier machinery (`_eagle3_next_block_size`/`_eagle3_block_tiers`) is unreachable (passing a value disables adaptive; passing None hits the clamp) | — | declined-for-now | Measured (MiniMax-M3 + Inferact EAGLE3, greedy): block 2 = 32 tok/s (commit/round 1.77, full-accept 0.77), block 4 = 24 (2.27, 0.16), block 6 = 17 (2.23, 0.01). Raising the block HURTS on a sparse-MoE target — each wider verify forward pages a larger union of experts while acceptance cliffs past 1-2. The clamp is *correct* here; only the dead adaptive code is worth pruning, and only after confirming it doesn't help dense targets. |

## How to maintain this file

When a fork commit fixes something upstream also has, add a row (change,
commits, status, why upstream wants it). When upstream fixes it independently,
move it to "superseded" with their PR number. Before any actual PR: ask Paul.
