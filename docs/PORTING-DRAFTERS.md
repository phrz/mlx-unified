# Porting drafter-family speculative decoding into mlx-unified

Goal: `mlx_lm.server --draft-model <drafter> --draft-kind {dflash,eagle3,mtp}` (kind
auto-detected from the drafter's HF `model_type` when omitted), so assistant/MTP
drafters run under the UNIFIED server — same process as the prompt-cache layer —
instead of requiring a standalone `mlx_vlm.server`.

## Design: import, not vendor

mlx-vlm (`phrz/mlx-vlm@draft-blocks`, rebased onto upstream 2026-07-16) owns the
drafter implementations and round-loop math in `mlx_vlm.speculative`. The fork's
`[vision]` extra already pins it, and the unified architecture's rule is "mlx-lm
backbone; mlx-vlm contributes components" (vision embeddings via bridge, diffusion
via `vlm_delegate`). Speculative follows the same pattern: **lazy-import
`mlx_vlm.speculative` at drafter load time** (`mlx_lm/spec_delegate.py`, mirroring
`vlm_delegate.py`). No copied drafter code; upstream mlx-vlm improvements (ddtree
etc.) arrive through the pin. Without `[vision]` installed, `--draft-kind` fails
with a clear "install mlx-lm[vision]" error; plain `--draft-model` (classic
same-tokenizer draft) is untouched.

## Duplication check (2026-07-16)

- upstream mlx-lm: nothing merged; PR #1276 (gemma4_assistant class only) and
  #990-family (qwen3.5 baked-MTP self-speculation) stalled. Reconcile if they land.
- mlx-engine (LM Studio): no speculative code of its own — passes draft_model into
  mlx-lm's stream_generate; waiting on mlx-lm-level support (their issue #205).
- mlx-vlm: the only full implementation (this port's source).

## Target-model hook contract (what the round loop needs from gemma4)

Discovered from `mlx_vlm/models/gemma4/language.py` (the vlm-side patched model):

1. **Prefill**: kwargs `return_hidden=True` / `return_shared_kv=True` → forward
   returns pre-norm hidden states and fills a `shared_kv_sink: dict` keyed by
   `layer.layer_type` with the (keys, values) of the KV-shared top layers
   (`first_kv_shared_layer_idx`). The assistant drafter cross-attends (Q-only)
   into these.
2. **Decode-step layers**: accept `shared_kv=(keys, values)` to reuse target KV.
3. **Helpers on the model**: `logits_from_hidden(h)` (embed_tokens.as_linear + caps),
   `speculative_logits_from_hidden(h)` (= logits_from_hidden(norm(h))),
   `speculative_draft_hidden(h)` (= norm(h)).
4. Drafter side (`Gemma4AssistantDraftModel.bind(target)`): tolerant attribute
   walk — finds `embed_tokens` via `target`, `target.model`, or
   `target.language_model.model`; mlx_lm's gemma4 `Model.model` property already
   matches this shape.

Port location: `mlx_lm/models/gemma4_text.py` (backbone — structurally close:
already has `first_kv_shared_layer_idx` / kv-shared layers) + thin passthroughs on
`gemma4.py` and `gemma4_unified.py` wrappers.

## Round loop integration

`mlx_lm/generate.py` gains `drafter_generate_step(...)` alongside the existing
`speculative_generate_step` (classic draft). It dispatches per resolved kind and
reuses `mlx_vlm.speculative`'s verify walks (`_mtp_verify_target`,
`_speculative_walk_deferred_greedy`, dflash/eagle3 equivalents) — those operate on
"an lm with the hook contract" + drafter + caches, so they run against mlx_lm
models once the hooks exist. Batch-size 1 only in v1 (the server's speculative
path is sequential; mlx-vlm's own kv-bits path sets the same precedent).

`stream_generate` chooses: `drafter_generate_step` when a drafter-family model is
the draft (detected via `resolve_drafter_kind` at load), else the classic step.

## Server wiring

- `--draft-kind {dflash,eagle3,mtp}` (default None = auto from drafter model_type).
- Drafter loading via `mlx_vlm.speculative.drafters.load_drafter(path, kind)`
  through spec_delegate; classic draft models keep loading via mlx_lm `load()`.
- Prompt-cache coexistence: same rule as `--kv-bits` — drafter speculation runs on
  the sequential decode path; the LRU prompt cache stores/reuses TARGET KV as
  usual (shared_kv_sink is derived per request at prefill, never cached in v1).

## Step sequence

1. `gemma4_text.py` hooks (+ wrapper passthroughs) — parity-tested against
   mlx-vlm's language.py outputs on a tiny random config.
2. `spec_delegate.py` (lazy import, load_drafter/resolve_kind wrappers, clear
   install-hint errors).
3. `drafter_generate_step` in generate.py + stream_generate dispatch.
4. server: `--draft-kind` flag + drafter load path.
5. tests: tests/test_unified_drafters.py (tiny-config: hook parity, kind
   resolution, one greedy mtp round-trip vs unspeculated output equality).
6. Runway (separate repo): caps probe `--draft-kind` on mlx-lm path, ParamsDialog
   applies the existing select to unified, live verify gemma-4-12B + assistant
   (canary READY, ~1.4x, prompt cache active), consider paulup-thinking switch.

## Verification targets

- Greedy MTP output must EQUAL the unspeculated greedy output (speculation is
  lossless under greedy acceptance) — that's the core correctness test.
- Live: 12B + assistant drafter ≥ 65 tok/s (vlm-server baseline 76.5 after
  rebase); prompt-cache hit TTFT still ~90ms on repeat prompts.
