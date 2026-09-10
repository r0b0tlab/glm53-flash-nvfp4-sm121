---
library_name: vllm
base_model: nvidia/GLM-5.3-Flash-NVFP4
tags:
- sm121
- gb10
- dgx-spark
- vllm
- dflash2
- speculative-decoding
- tensor-parallel
---

# GLM-5.3-Flash-NVFP4 — SM121 (2× GB10) serving recipe

Serving recipe + reproduction package for the official
[`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
checkpoint on **two NVIDIA DGX Spark / GB10 nodes** (SM121, 121 GiB unified each)
at **TP=2**, with the
[`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
block-diffusion drafter and thinking at `reasoning_effort=max`.

**This repo contains no weights** — it is the recipe (patches, serve profiles,
evidence). The checkpoint is NVIDIA's; we do not re-quantize it.

## What's here

- `patches/` — four anchored, unit-tested SM121 patches applied in-build on top
  of the pinned NVIDIA image
  (`vllm/vllm-openai@sha256:b0501f99…`, vLLM 0.28.1rc1.dev580, FlashInfer 0.6.18):
  1. **NoPE sparse MLA on SM121** — enable the SM90 sparse-MLA backend for
     capability 12 (the stock SM120 path requires the packed `fp8_ds_mla`
     layout GLM's NoPE layers cannot use), FA2 off-Hopper.
  2. **DSA indexer `persistent_topk` gate** — GB10's 48 SMs take
     `top_k_per_row_decode`; the fast path's fallback needs 128 KiB
     smem/block and the engine dies past ~24K decode tokens without this.
  3. **DFlash2 aux-capture glue** — `SupportsEagle3` + mHC-contracted hidden
     states at taps 5/14/24/33/42 (runner ids 6/15/25/34/43).
  4. **DFlash2 drafter KV group** — the GLM KV fast path learns the drafter's
     sliding-window layers (exact-fit slot-share of the MLA tensors; never
     `page_size_padded`).
- `container/` — `build.sh` + `Dockerfile.overlay` (pull pinned base, apply
  patches with in-build verification, distribute to both ranks with image-ID
  parity).
- `serve/` — TP=2 launcher + frozen profiles (`boot_dflash2.sh`,
  `boot_and_q200.sh`, `ar_cycle.sh`), fabric/NCCL config.
- `scripts/` — the gate harness (corruption canary, acceptance matrix, NIAH,
  Q200v2/BFCL wrappers, benches).
- `evidence/` — sanitized results bundle with `MANIFEST.sha256`.

## Verified results (full detail in `evidence/`)

| Gate | Result |
|---|---|
| Corruption canary (ModelOpt U+FFFD / locks) | PASS 18/18 |
| Text | 5/5 (incl. real tool_calls) |
| Vision | 8/8 at max thinking |
| NIAH (served window 126,720) | **PASS 5/5** — 25/50/90 % + multi-key 33/66 at ~120K-token prompts |
| Deep decode (>24K ctx) | PASS |
| Q200v2 text180 | see `evidence/q200v2/` |
| DFlash2 speedups (C1, same prompts) | code 15.1 → **34.0 tok/s (2.25×)**; prose 14.9 → 25.0 (+68 %); math-reasoning 46.3 tok/s |
| Spec acceptance | 39 % pooled, first-slot 78 % (== reference), mean accepted length 2.75 |
| Spec losslessness | INDETERMINATE — same-serve repeats differ (temp-0 nondeterminism), not drafter decay |

## Boundaries (measured, disclosed)

- **KV: bfloat16.** fp8 requests canonicalize to a packed `uint8` layout the
  SM90 sparse-MLA backend rejects; bf16 is the validated maximum.
- **Window 126,720** at `gpu_memory_utilization=0.86` + drafter — the KV pool is
  the binding constraint (`kv_cache_max_concurrency` ≈ 1.36×).
- **Thinking cannot be disabled** (template always opens `<think>`); all gates
  ran at max effort with ≥8k budgets.
- Reasoning is returned in the **`reasoning`** field.
- Multi-image batches of identical-size frames can order-misread
  nondeterministically (single/pair reads exact).

## Attribution

NVIDIA (checkpoint + base image), vLLM, FlashInfer; public SM121 GLM work
verified and credited: `tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`
(SM121 NoPE route, DFlash2 glue, persistent_topk forensics), LibertAI
(`glm53-flash-vllm-gb10`), vLLM PR #53969 (NoPE zero-pad). Independent
deployment by r0b0tlab — not affiliated with or endorsed by NVIDIA or ZAI.
