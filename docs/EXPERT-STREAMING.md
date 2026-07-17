# Expert-aware SSD streaming (design)

Goal: serve MoE models far larger than the RAM we're willing to give them
(GLM-5.2 @ ~238GB on ~100GB residency, other models staying loaded) WITHOUT
naive mmap paging — the OS page cache is expert-blind, so random 8-of-256
routing across 75 layers thrashes RAM and disk. The cache must be
expert-granular and router-driven.

## Sources studied (2026-07-17)

- **FlashMoE (arXiv:2601.17063)** — cache REPLACEMENT is where the wins are:
  a lightweight adaptive policy blending recency + frequency beats LRU/LFU by
  up to 51% hit rate, 2.6× end-to-end. Policy layer, portable anywhere.
- **mu-hashmi/mlx-moe (MIT)** — the proven MLX recipe, and the closest to our
  stack: swap every `SwitchGLU` expert module for a lazy-loading version that
  materializes only router-selected experts into GPU-resident stacked tensors;
  capacity-bounded per-layer cache (`--capacity`, auto from free RAM);
  precomputed **expert profiles** pin "universal" experts permanently (their
  finding: without pinning, quality degrades after ~300 tokens as hot experts
  churn); zero-eval dispatch keeps MLX's async pipeline intact. 46GB model on
  32GB Macs at 6–23 tok/s. Works for "any MLX model using SwitchGLU".
- **SharpAI/SwiftLM `--stream-experts` (MIT)** — the throughput ceiling story:
  cross-projection batching (collapse ~1400 per-expert calls → ~48/token),
  concurrent NVMe prefetch at queue depth 24, SPECULATIVE prefetch of the next
  token's experts during current-token compute (~70% hit), persistent Metal
  buffers. 0.58 → 5.9 tok/s on Qwen3.5-122B/64GB. Needs custom mlx forks for
  out-of-core Metal; GLM-5.x not supported.

## Why mlx-unified is the vehicle (not llama.cpp, not a new provider)

- llama.cpp has NO expert-aware cache (mainline confirmed; the one fork is
  Vulkan/Linux); building one is a months-scale C++ project.
- mlx-moe's technique is SwitchGLU-generic, and our fork's GLM-5.2 arch
  (`glm_moe_dsa` → deepseek_v32 MoE) IS SwitchGLU-based — as are qwen3.5,
  glm4_moe, deepseek, gemma4 MoE variants. One implementation covers them all.
- In-fork means coexistence with everything else we've built: prompt cache,
  drafters, the OpenAI server, Runway's canary/estimator.
- MLX checkpoints exist for GLM-5.2 down to 2.56bpw (~238GB on SSD).

## Design (phase 1 — the mlx-moe recipe, adapted)

New module `mlx_lm/expert_stream.py` + server flag `--stream-experts`:

1. **Load transform**: after model load with lazy weights, replace each
   SwitchGLU's expert weight stack with an `ExpertStreamStore`: safetensors
   stay on disk; per-(layer, projection) GPU-resident ring of `capacity`
   expert slots + an index map expert_id → slot.
2. **Dispatch**: forward hook reads router top-k, ensures selected experts are
   resident (batch-loading misses from disk via MLX lazy slicing), then runs
   the existing grouped SwitchGLU math over the stacked slots. No per-expert
   Python calls on the hot path (mlx-moe's zero-eval dispatch).
3. **Capacity**: `--expert-capacity N | auto` (auto = derive from free RAM at
   load, leaving headroom for KV + other wired models).
4. **Pinning**: `--expert-profile FILE` — precomputed universal-expert sets
   (we can generate profiles with a small calibration run, mlx-moe style);
   pinned experts never evict. Without a profile, first-N-touched heuristic.
5. **Eviction**: start LRU (phase 1), upgrade to FlashMoE-style
   recency+frequency scoring (phase 2) — the score is a couple of counters
   per expert, cheap.
6. **Estimator/Runway**: resident bytes = non-expert weights + capacity ×
   expert-slot bytes (exact from safetensors index); the streamed remainder is
   NEITHER wired nor RAM-resident — a third class the estimator reports as
   "streamed from SSD".

Phase 2 (throughput): speculative next-token prefetch overlapped with compute
(SwiftLM's ~70%-hit trick), concurrent prefetch queue, and the FlashMoE
replacement policy. Phase 3 (maybe): cross-layer co-activation profiles.

Expectations, set honestly: single-request decode on GLM-5.2-class models at
a ~100GB budget should land in the low-to-mid single digits tok/s (mlx-moe:
6–23 on a much smaller model; SwiftLM: ~6 on 122B) — SSD latency is the
floor, and DSA/MLA attention still runs dense. This is a "have the huge model
available alongside everything else" mode, not a speed mode.


## Status + open bug (2026-07-17, post-integration)

Mechanics land (7.5GB wired for 16.6GB Qwen3-30B-A3B at cap 52/128; swaps,
drains, refinement all run). CONTROL: mlx-moe's own server on the identical
model+capacity produces flawless output — the ceiling is reachable.

Our server's output quality is UNSTABLE run-to-run (isolated wrong tokens →
repetition collapse; one fully-clean cold request observed after adding a
pipeline drain, not reproducible). Attempted fixes so far: mx.synchronize()
before slot mutation (default stream, then generation_stream too), disabling
cross-request prompt-cache reuse/insert under streaming (principled — decode
KV is computed through the approximate skip-fallback path — keep regardless).
Neither stabilized it; signal is noisy, likely a race.

Next-session plan (deterministic, no more shotgun):
1. Same converged prepacked cache state in BOTH stacks, temp 0, fixed prompt.
2. Debug hook capturing per-token logits (or argmax id) in our serve loop and
   in mlx-moe's Server._stream; find the FIRST divergent token.
3. Candidate deltas to check, in order: our fork's generate_step pipeline
   depth + thread-local generation_stream vs the server's worker thread
   (dynamic_cache_update mutating slots the in-flight graph reads); our
   chunked prefill (prefill_step_size) vs their single-shot prefill through
   lazy expert loading; detokenizer/think-parse (cosmetic only).
4. If the race is pipeline-depth: consider running dynamic_cache_update on
   generation_stream itself (ordering instead of draining).

GLM-5.2 Alis 2.56bpw (~225GB) is downloaded and waiting in the runway store.
Runway-side plumbing (spec/args/estimator) deliberately deferred until this
stabilizes.
