# GLM-5.3-Flash-NVFP4 SM121 — execution log (campaign evidence)

Record of the bring-up, verbatim root causes and fixes. Hostnames/IPs scrubbed;
see the private campaign tree for the raw logs.

## 1. Base image (pinned)

- `vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5`
  (`glm53-flash-arm64-cu130`)
- vLLM `0.28.1rc1.dev580+g385dce36b`, FlashInfer 0.6.18, transformers 5.16.1,
  torch 2.13.0+cu130, arm64.

## 2. Checkpoint (static audit, g0)

- `nvidia/GLM-5.3-Flash-NVFP4`, sha `423acf375837…`, 33 shards, 190.4 GiB.
- NVFP4 (modelopt) covers routed experts + dense MLP of layers 0–44 only:
  36,288 expert + 9 dense projections, each with `weight_scale`,
  `weight_scale_2`, `input_scale` (vLLM #54189 does NOT apply here).
- MTP layer 45 is BF16 (no scale companions) — expected.
- 347 vision tensors present, none quantized. Index 147,661 keys, 0 dangling.
- Verdict: PASS (`evidence/gates/g0_static.json`).

## 3. SM121 overlay (see `patches/`)

- **P1** — `FLASHINFER_MLA_SPARSE_SM90` enabled for capability 12 (the stock
  capability-12 list only offers `FLASHINFER_MLA_SPARSE_SM120`, whose packed
  `fp8_ds_mla` layout cannot serve NoPE sparse MLA), with the FA2 path off
  Hopper. Mechanism informed by public SM121 GLM deployments; anchors
  re-derived for this base.
- **P2** — DSA indexer `persistent_topk` gated on device SM count ≥ 78. GB10 has
  48 SMs; past ~24K decode tokens the kernel's fallback needs 128 KiB
  smem/block vs SM121's 99 KiB and the engine dies. Fallback path
  (`top_k_per_row_decode`) is correct everywhere.
- **P3** — DFlash2/EAGLE-3 aux-capture glue on `glm5next` (SupportsEagle3 +
  mHC-contracted taps). Inert without a drafter.
- **P4** — GLM KV-group support for the DFlash2 drafter's SlidingWindowSpec
  layers (exact-fit slot-share / standalone), ported to this base's packed
  per-layer tensor layout. Inert without a drafter.
- **P5** (prepared, not shipped) — SM120 NoPE zero-pad route, derived from
  vllm PR #53969. Kept as the fallback if P1 ever regresses.

## 4. Boot forensics (3 failures → fixes)

1. **`piecewise CUDA graphs (FULL_AND_PIECEWISE) unavailable, model is not
   torch-compiled and breakable CUDA graph is off`** — the model does not
   support torch.compile; the default graph mode is unusable.
   Fix: `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`.
2. **`MLA kv_data_type torch.uint8 is not supported. Supported dtypes: [fp16,
   bf16, fp8_e4m3fn]`** with `--kv-cache-dtype fp8` — for MLA models the engine
   canonicalizes fp8 requests through a packed uint8 path the SM90 sparse MLA
   backend cannot consume.
   Fix: **`--kv-cache-dtype bfloat16`** (the SM90 backend's validated base
   path; its FA2 `BatchMLAPagedAttentionWrapper` was probed on SM121 with
   GLM's exact shape). fp8 KV on this backend needs further work and is NOT
   claimed.
3. (Same as 2 for `fp8_e4m3`; the canonicalization is not dtype-string-specific
   for this backend.)

## 5. Successful boot signatures (bf16 KV profile)

- `Using FLASHINFER_MLA_SPARSE_SM90 attention backend out of potential
  backends: ['FLASHINFER_MLA_SPARSE_SM90', 'FLASHINFER_MLA_SPARSE_SM120']`
- `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend` (native W4A4, no Marlin)
- DeepGEMM PDL + E8M0 enabled
- `DSA indexer decode path: use_flattening=False supports_varlen=False
  (next_n=1, use_fp4_cache=False)`
- KV layout: indexer block 64; attention block derived to keep the attention
  page ≥ mamba page (mamba page padded 2.64%)
- TileLang JIT compiles mHC kernels on first boot
  (`mhc_pre_big_fuse_with_norm_tilelang`, `mhc_post_tilelang`).
