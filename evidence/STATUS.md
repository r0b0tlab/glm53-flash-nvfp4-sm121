# Campaign status checkpoint

Last updated: 2026-09-10 ~02:20 UTC (mid bring-up).

## Green (verified, evidence in tree)

| Gate | Result | Evidence |
|---|---|---|
| g0 checkpoint static audit | **PASS** — 36,297 NVFP4 modules with full companions; MTP layer BF16; 347 vision tensors unquantized; 147,661 index keys; 0 dangling | `evidence/gates/g0_static.json` |
| g1 boot signatures | SM90 sparse-MLA backend on SM121; FLASHINFER_CUTLASS NVFP4 MoE (native); DeepGEMM PDL+E8M0; KV groups [2304,4,…] | `evidence/gates/g1_boot_grep.txt` |
| g3 corruption canary (HARD) | **PASS** — 18/18 cases, 0 U+FFFD, no repetition locks (NVIDIA ModelOpt checkpoint is clean) | `evidence/gates/g3_canary.json` |
| g7 text | **5/5** (391, fib 55, JSON, code, real tool_calls) | `evidence/g7_text.json` |
| g8 vision | **8/8** (4 counting, OCR 4271, bar chart, frame order, frame 3) at reasoning_effort=max + 8192 tokens | `evidence/g8_vision.json` |
| deep-decode (>24K, P2 gate) | **PASS** — 24,207 prompt tokens, 127 completion, engine healthy after | stdout |
| sm121 overlay | P1–P4 applied in image `glm53-flash-vllm-sm121:overlay-v2`, `OVERLAY_VERIFY_PASS`, parity OK on both ranks | `evidence/runtime/` |

## Standing user rules applied

- Thinking is always on (template cannot disable it); **reasoning_effort=max everywhere**.
- **Ample token budgets** (probes 8k; quality lanes 8k).

## In flight

- **DFlash2 boot**: 4 cycles to a green profile — fixes and their root causes:
  1. `FULL_AND_PIECEWISE` unavailable (model not torch-compiled) → pin
     `FULL_DECODE_ONLY`.
  2. `fp8`/`fp8_e4m3` KV → packed `uint8` spec the SM90 sparse-MLA backend
     rejects → **bfloat16 KV** (validated path).
  3. KV admission: 262K needed 4.19 GiB at gmu 0.85 (had 2.09) → gmu 0.86;
     131K short by 0.04 GiB → **window 126,720**.
  4. FlashInfer autotune's distributed profiling hung once during spec warmup
     (rank0 spinning, rank1 idle) → skip the tintune sweep (configs are cached;
     perf re-measured against the autotuned baseline).
- **Acceptance analysis (37.7% pooled answered)**: first-slot 77% == reference;
  decay is workload-bound (mixed canary workload, thinking=max). End-to-end on
  the same code prompt: **AR 15.1 → DFlash2 33.6 tok/s (2.2×)** — matches the
  reference's 2.15×.
- **Reasoning field is `reasoning`** (not `reasoning_content`) — all scripts
  updated; this was the cause of the false "empty content" canary flags.

## Pending decisions / follow-ups

- fp8 KV on this backend is NOT supported (packed uint8 canonicalization);
  bf16 KV is the qualified path. Revisit only with a dedicated fix.
- Window 126,720 on 2×GB10 (KV-memory bound with the drafter); 262K needs
  ~4.2 GiB KV more than the free-memory cap allows at TP=2.
- Video probe: native `video_url` not yet exercised (image+frame probes pass).
- Publication gated until all lanes close.

## Measured so far (base serve, bf16 KV, no draft)

- single-stream decode ~25 tok/s (generation throughput at c=1).
- 262K KV pool sizing: 5.01 GiB at gmu 0.85 (AR); est. max length 66,816 at
  2.09 GiB (with drafter at 0.85).
