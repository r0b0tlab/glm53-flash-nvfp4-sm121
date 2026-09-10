#!/usr/bin/env python3
"""Verify the GLM53 SM121 overlay inside the built image (fail-closed).

Checks every patch marker, that the patched files still parse, and that the
behavioral hooks exist. Exits non-zero on any miss.
Usage: python3 verify_overlay.py [--root /usr/local/lib/python3.12/dist-packages]
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

DEFAULT_ROOT = "/usr/local/lib/python3.12/dist-packages"

CHECKS = [
    ("vllm/platforms/cuda.py", "GLM53-SM121-NOPE-MLA"),
    ("vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py", "GLM53-SM121-NOPE-MLA"),
    ("vllm/model_executor/layers/sparse_attn_indexer_kpool.py", "GLM53-SM121-TOPK-GATE"),
    ("vllm/models/glm5next/nvidia/model.py", "GLM53-DFLASH2-AUX-CAPTURE"),
    ("vllm/v1/core/kv_cache_utils.py", "GLM53-DFLASH2-DRAFTER-GROUP"),
]

BEHAVIOR = [
    ("vllm/platforms/cuda.py", "FLASHINFER_MLA_SPARSE_SM90"),
    ("vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py", "capability.major in (9, 12)"),
    ("vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py", 'else "fa2"'),
    ("vllm/model_executor/layers/sparse_attn_indexer_kpool.py", "_persistent_topk_fits_device()"),
    ("vllm/models/glm5next/nvidia/model.py", "EagleModelMixin"),
    ("vllm/models/glm5next/nvidia/model.py", "SupportsEagle3"),
    ("vllm/models/glm5next/nvidia/model.py", "aux_hidden_states"),
    ("vllm/v1/core/kv_cache_utils.py", "draft_group"),
    ("vllm/v1/core/kv_cache_utils.py", "GLM53-DFLASH2-DRAFTER-GROUP"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    a = ap.parse_args()
    root = Path(a.root)
    bad = 0
    for rel, needle in CHECKS:
        p = root / rel
        if not p.exists():
            print(f"FAIL missing {rel}")
            bad += 1
            continue
        text = p.read_text(encoding="utf-8")
        try:
            ast.parse(text, filename=str(p))
        except SyntaxError as e:
            print(f"FAIL ast.parse {rel}: {e}")
            bad += 1
            continue
        if needle not in text:
            print(f"FAIL marker {needle} absent in {rel}")
            bad += 1
    for rel, needle in BEHAVIOR:
        p = root / rel
        if needle not in p.read_text(encoding="utf-8"):
            print(f"FAIL behavior hook {needle!r} absent in {rel}")
            bad += 1
    if bad:
        print(f"OVERLAY_VERIFY_FAIL ({bad} checks failed)")
        return 1
    print("OVERLAY_VERIFY_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
