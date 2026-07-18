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


## Status (2026-07-17, delegation architecture)

The hand-rolled expert machinery was replaced by DELEGATION: `--stream-experts`
now runs mlx-moe's own `Server` session (`_MoeSession`) on a dedicated
single-thread executor; `_serve_streamed_experts` tokenizes with our chat
template, keys the mlx-moe prompt cache per conversation (sha256 of the first
user message), and adapts `session._stream(...)` responses back into our SSE
loop. Everything else in the fork (prompt cache, drafters, diffusion, vision)
is untouched on the non-streamed path.

Post-mortem on the earlier "unstable output" bug hunt: the dominant cause was
the TEST HARNESS, not the server — zsh arrays are 1-indexed, so the
`declare -a`-based multi-question batteries sent an EMPTY first prompt and
shifted every subsequent one. With a Python harness the delegate answers
correctly. The residual truth: at aggressive capacity (52/128 experts on
Qwen3-30B under topic churn) quality degrades EVEN ON mlx-moe's own server —
an engine-inherent limit of capacity-bounded expert caching, not an
integration bug. Use temp ~0.7 (greedy loops on approximate experts), and
size capacity generously.

GLM-5.2 (glm_moe_dsa) needed model-side work in the fork: the Alis 2.56bpw
checkpoint uses GLM-5.2's cross-layer DSA indexer sharing
(`config.indexer_types`: "full" every 4th layer, "shared" between), which
upstream mlx-lm doesn't support — shared layers carry no indexer weights and
reuse the previous full layer's top-k selection, threaded through the decoder
loop (matches transformers' `modeling_glm_moe_dsa`). GLM's activation ratio
(8/256 routed) gives expert streaming a far better cushion than the 30B proxy.

Runway-side plumbing (spec/args/estimator) lands after GLM-5.2 verifies
end-to-end.
