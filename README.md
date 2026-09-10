# GLM-5.3-Flash-NVFP4 on 2× GB10 (SM121): SM121-optimized vLLM + DFlash2 serving package

Maximum-performance, fully-qualified serving of
[`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
(the official NVIDIA ModelOpt NVFP4 checkpoint; **we do not quantize**) on two NVIDIA
DGX Spark / GB10 nodes at tensor-parallel 2, with the
[`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
block-diffusion drafter, thinking at `reasoning_effort=max` throughout.

- **Runtime image:** `ghcr.io/r0b0tlab/glm53-flash-nvfp4-sm121:overlay-v2` — layered over
  `vllm/vllm-openai@sha256:b0501f99…` (vLLM 0.28.1rc1.dev580, FlashInfer 0.6.18,
  transformers 5.16.1, torch 2.13.0+cu130, arm64) with four anchored SM121/GLM patches.
  (Pushed digest `sha256:b9edc8e0…`; **package visibility flip to public is a one-click
  UI action** at the package settings page — the repository CI's anonymous-pull gate
  goes green once flipped.)
- **Patches** (`patches/`, each idempotent, fail-closed on anchor drift):
  1. `0001-sm90-nope-mla-sm121.py` — enable the SM90 NoPE sparse-MLA backend on
     capability 12 (the stock SM120 path requires the packed `fp8_ds_mla` layout the
     NoPE layers cannot use), FA2 off-Hopper.
  2. `0002-indexer-persistent-topk-sm121.py` — gate the DSA indexer's
     `persistent_topk` on device SM count ≥ 78; GB10's 48 SMs take
     `top_k_per_row_decode` (the fast path's fallback needs 128 KiB smem/block;
     SM121 has 99 KiB and the engine dies past ~24K decode tokens).
  3. `0003-glm-dflash2-aux-capture.py` — SupportsEagle3 + mHC-contracted aux
     hidden-state capture for the DFlash2 drafter (taps 5/14/24/33/42 → runner
     ids 6/15/25/34/43).
  4. `0004-glm-dflash2-drafter-group.py` — teach the GLM KV-group fast path the
     drafter's SlidingWindowSpec layers (exact-fit slot-share of the MLA tensors;
     never `page_size_padded`).

## Verified gates (evidence in `evidence/`)

| Gate | Result |
|---|---|
| Checkpoint static audit | PASS — 36,297 NVFP4 modules fully scaled; MTP layer BF16; 347 vision tensors unquantized |
| Corruption canary (ModelOpt U+FFFD / repetition locks) | PASS — 18/18 clean (the NVIDIA checkpoint is unaffected by the community-reported ModelOpt corruption) |
| Text | 5/5 (391, fib 55, JSON, code, tool call) |
| Vision | 8/8 (counting ×4, OCR 4271, bar chart, frame order, frame 3) at max thinking |
| Deep decode (>24K ctx) | PASS (24,207-token prompt; engine healthy) |
| DFlash2 spec decode | active — taps verified (`Eagle3 auxiliary layers (6, 15, 25, 34, 43)`); pooled acceptance 37.7%, mean accepted length 2.64 on a mixed workload; **C1 code prompt: AR 15.1 → DFlash2 33.6 tok/s (2.2×)**; prose 14.9 → 17.9 (+20%). Acceptance is workload-dependent: first-slot acceptance (77%) matches the reference, deeper positions decay faster on reasoning-heavy (thinking=max) output. See `evidence/spec/` |
| Quality lanes | **NIAH PASS 5/5** (window 126,720; 25/50/90 % + multi-key 33/66 at ~120K-token prompts, ~2.5 min/case); **Q200v2 170/178 transported (95.5 %)**: gsm8k 78/80, humaneval 39/40, ifeval 35/40, hard-reasoning 18/20 (2 transport-excluded at the 8192 ceiling; review also confirmed the model gets (13,8), which the dataset reference itself omits); **BFCL-hard20 14/20 (70.0 %, 0 errors)**; **vision benchmark 1,013/1,237 = 81.9 %** on deterministic first-300-per-suite bounds of r0b0bench-vision v1.0.0 (cvbench-Count 66.5, MMVP 87.3, RealWorldQA 80.3, OCRBench 95.3; ling full-suite reference 80.8 %). See `evidence/` |
| Perf | see `evidence/perf/` |

## Operating notes (measured, not assumed)

- **KV cache dtype: `bfloat16`.** `fp8`/`fp8_e4m3` requests are canonicalized through a
  packed `uint8` layout this backend cannot consume; the SM90 sparse-MLA path's
  validated cache is plain bf16 (its FA2 wrapper was probed on SM121 with GLM's
  exact shape).
- **Graphs:** `{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}` — the model is not
  torch-compiled, so the default `FULL_AND_PIECEWISE` is unavailable.
- **Thinking:** the chat template always opens `<think>`; `enable_thinking:false` is
  ignored. All gates and lanes run at `reasoning_effort=max` with generous budgets
  (≥8k for reasoning-laden tasks).
- **Reasoning parser / field:** reasoning is returned in the **`reasoning`** field
  (not `reasoning_content`) — read that key or truncated outputs look empty. The
  server runs `deepseek_r1`; `glm45` is only kept as a historical note.
- **Memory:** TP=2 at `gpu_memory_utilization=0.86` (free-memory admission drifts by
  ~0.5 GiB across boots on the 121 GiB unified pool; 0.87+ can be rejected).
- **Swap:** `vm.swappiness=0` with swap enabled on both ranks (UVM stability).
- **Multi-image batches:** identical-size multi-image ordering can be misread
  nondeterministically; single-image and 2-image requests are exact. Documented as a
  boundary, not hidden.

## Reproduce

```
bash container/build.sh            # pull pinned base, apply patches, verify, distribute
bash serve/boot_dflash2.sh         # TP=2 + DFlash2 K=7 (see config.env for the fabric)
```

Attribution: NVIDIA (model + base image), vLLM, FlashInfer, and the public SM121 GLM
work this campaign verified and credits — `tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`
(NoPE-on-SM121 route, DFlash2 glue, persistent_topk forensics), LibertAI
(`glm53-flash-vllm-gb10`, MoE-scale root cause), and vLLM PR #53969 (NoPE zero-pad).
Independent deployment by r0b0tlab; not affiliated with or endorsed by NVIDIA or ZAI.
