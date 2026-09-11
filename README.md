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
- **Window 126,720; a full-context 1M NIAH does NOT fit this pair.** Live budget:
  3.64 GiB/rank of KV available at gmu 0.86 with the drafter → pool 148,808 tokens
  (1.17× concurrency at the window), effective cost ~22.2 KiB/token (bf16). A
  1,048,576-token window needs **~22.2 GiB/rank of KV — ~6× more than available**
  (even with the drafter dropped and gmu maxed it stays ~4× short). TP=3 is ruled
  out by arithmetic (64 attention heads don't divide by 3); a 1M ladder needs
  TP=4-class hardware. The full served-window ladder (126,720) is qualified 5/5.
- **Swap:** `vm.swappiness=0` with swap enabled on both ranks (UVM stability).
- **Multi-image batches:** identical-size multi-image ordering can be misread
  nondeterministically; single-image and 2-image requests are exact. Documented as a
  boundary, not hidden.

## Reproduce

```
cp config.env.example config.env   # optional — only needed for multi-node serve/distribute
bash container/build.sh            # pinned base + anchored patches + in-image verify;
                                   # distributes to both ranks only when RANK0/RANK1 are set
bash serve/boot_dflash2.sh         # TP=2 + DFlash2 K=7 (host/fabric values come from config.env)
```

`config.env` is intentionally not committed (node and fabric specifics live there);
`config.env.example` documents every variable, and a single-node image build works
without it.

The Q200v2 quality lane (`scripts/q200_only.sh`, `scripts/quality_campaign.sh`) runs the
frozen kit published in [`r0b0tlab/r0b0bench`](https://github.com/r0b0tlab/r0b0bench/tree/main/subsets/q200v2)
— dataset `quality-text-180-v2.jsonl` sha256
`74623ab9b075120cd6f7a93059cc16d8817a6039dd20118b8f0350279f8b1ed6` — plus the BFCL v4
`multi_turn_base` structural-hard20 selection. Point `Q200V2_RUNNER` at your local checkout of
that kit.

## Run the published image

The overlay image is public on GHCR and CI pulls it anonymously on every run:

```
docker pull ghcr.io/r0b0tlab/glm53-flash-nvfp4-sm121:overlay-v2
```

To bring up the two-rank serve, fill in `config.env` (see `config.env.example`), make
sure the image is present on both ranks (`docker pull`, or `bash container/build.sh`
builds and distributes it with an image-ID parity check), then:

```
bash serve/boot_dflash2.sh        # DFlash2 K=7 profile; TP=2, window 126,720
SG_DRYRUN=1 bash serve/launch-glm53-tp2.sh   # print the exact docker run commands first
```

The launcher mounts the model tree read-only at `/model`, the drafter at `/drafter` for the
DFlash2 profile, each rank's JIT caches (`~/.cache/{triton,vllm,flashinfer}`), and passes the
per-rank `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` values from `config.env`. Rank 1 starts first
(headless); rank 0 binds the API on `API_PORT`.

## What's in this repo

| Path | Contents |
|---|---|
| `container/` | `Dockerfile.overlay` + `build.sh` — pinned base by digest, anchored patches applied in-build, in-image verify, optional rank distribution with ID parity |
| `patches/` | the SM121 patches (SM90 NoPE sparse-MLA on SM121, indexer `persistent_topk` SM-count gate, DFlash2 aux-capture glue, drafter KV group, SM120 NoPE pad) + `verify_overlay.py` |
| `serve/` | TP=2 launchers and profiles: `boot_dflash2.sh` (primary), `boot_v2.sh`, `ar_cycle.sh`, `boot_when_ready.sh`, `boot_and_q200.sh`, `launch-glm53-tp2.sh` |
| `scripts/` | gate harness — canary, acceptance matrix, lossless record/compare, NIAH, Q200v2/BFCL lanes, evidence packager, `test_no_config_env.sh` |
| `evidence/` | sanitized results with `MANIFEST.sha256`, verified by CI |
| `model-card/` | the model/recipe card source |

Contributions are welcome as issues; the CI job (`validate`) runs the evidence manifest
check, script syntax + ShellCheck, the no-`config.env` regression guard, the private-residue
scan and the anonymous image pull.

Attribution: NVIDIA (model + base image), vLLM, FlashInfer, and the public SM121 GLM
work this campaign verified and credits — `tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`
(NoPE-on-SM121 route, DFlash2 glue, persistent_topk forensics), LibertAI
(`glm53-flash-vllm-gb10`, MoE-scale root cause), and vLLM PR #53969 (NoPE zero-pad).
Independent deployment by r0b0tlab; not affiliated with or endorsed by NVIDIA or ZAI.
