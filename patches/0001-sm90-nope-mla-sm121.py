#!/usr/bin/env python3
"""P1: enable the SM90 NoPE sparse-MLA backend on SM121 (GLM-5.3-Flash).

GLM-5.3-Flash sparse MLA layers are NoPE (qk_rope_head_dim=0). On capability-12
parts the stock candidate list only offers FLASHINFER_MLA_SPARSE_SM120, whose
packed fp8_ds_mla cache layout hard-requires DeepSeek's pe_dim=64 and dies in
warmup ("pe_dim must be 64 for fp8_ds_mla"). The SM90 sparse-MLA backend already
accepts qk_rope_head_dim in (0, 64) in supports_combination and drives
FlashInfer's BatchMLAPagedAttentionWrapper (FA2 path on SM121); it is gated to
capability major 9 and hardcodes backend="fa3" (Hopper-only).

Mechanism informed by the public r0b0tlab-style SM121 GLM deployments
(tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark docker v1); anchors and
verification re-derived for this base image (vLLM 0.28.1rc1.dev580+g385dce36b).

Idempotent (marker GLM53-SM121-NOPE-MLA); fails loudly if any anchor drifts.
Usage: python3 0001-sm90-nope-mla-sm121.py [--root /usr/local/lib/python3.12/dist-packages] [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "GLM53-SM121-NOPE-MLA"
DEFAULT_ROOT = "/usr/local/lib/python3.12/dist-packages"

EDITS = [
    (
        "platforms/cuda.py: capability-12 MLA candidates gain FLASHINFER_MLA_SPARSE_SM90",
        """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]
""",
        """        elif device_capability.major == 12:
            # GLM53-SM121-NOPE-MLA: the SM120 backend's fp8_ds_mla layout
            # cannot serve NoPE (pe_dim=0) sparse MLA; the SM90 backend can.
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]
""",
    ),
    (
        "flashinfer_mla_sparse_sm90.py: extend capability gate to major 12",
        """    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 9
""",
        """    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # GLM53-SM121-NOPE-MLA: FA2 path of the FlashInfer MLA wrapper runs on
        # capability 12 (GB10/SM121); the kernel covers ckv=512, kpe in {0,64}.
        return capability.major in (9, 12)
""",
    ),
    (
        "flashinfer_mla_sparse_sm90.py: pick FA2 off Hopper",
        """            backend="fa3",
""",
        """            backend=(
                "fa3" if torch.cuda.get_device_capability()[0] == 9 else "fa2"
            ),  # GLM53-SM121-NOPE-MLA
""",
    ),
]


def patch(root: Path, dry_run: bool) -> int:
    for name, anchor, replacement in EDITS:
        # files touched by this edit are encoded in the name
        pass
    targets = {
        "platforms/cuda.py": [EDITS[0]],
        "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py": EDITS[1:],
    }
    for rel, edits in targets.items():
        path = root / "vllm" / rel
        text = path.read_text(encoding="utf-8")
        if MARKER in text:
            print(f"[P1] {rel}: already patched; no-op")
            continue
        for name, anchor, replacement in edits:
            n = text.count(anchor)
            assert n == 1, (
                f"[P1] ANCHOR FAILED [{name}] in {rel}: expected 1 occurrence, "
                f"found {n}.\n--- anchor ---\n{anchor}\n--------------"
            )
            text = text.replace(anchor, replacement, 1)
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as e:
            raise AssertionError(f"[P1] post-edit ast.parse failed for {rel}: {e}") from e
        if dry_run:
            print(f"[P1] DRY RUN {rel}: {len(edits)} edits validated")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"[P1] {rel}: {len(edits)} edits applied, ast.parse OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    return patch(Path(a.root), a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
