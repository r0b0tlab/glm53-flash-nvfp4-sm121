#!/usr/bin/env python3
"""P2: gate the DSA indexer's persistent_topk off small-SM parts (SM121 GB10).

persistent_topk's CTA grid grows with context; past ~24K decode tokens on a
48-SM GB10 it oversubscribes the SM slots and the FilteredTopK fallback needs
128 KiB smem/block while SM121 has 99 KiB -> hard RuntimeError -> EngineDead.
The fallback path (top_k_per_row_decode) is correct everywhere.

Mechanism informed by the public SM121 GLM deployment forensics
(tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark, sparse_attn_indexer_kpool
SM121 gate); re-derived and verified for this base image.

Idempotent (marker GLM53-SM121-TOPK-GATE); fails loudly on anchor drift.
Usage: python3 0002-indexer-persistent-topk-sm121.py [--root ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "GLM53-SM121-TOPK-GATE"
DEFAULT_ROOT = "/usr/local/lib/python3.12/dist-packages"
TARGET = "vllm/model_executor/layers/sparse_attn_indexer_kpool.py"

HELPER_ANCHOR = """RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
"""

HELPER_NEW = """RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def _persistent_topk_fits_device() -> bool:
    \"\"\"GLM53-SM121-TOPK-GATE: persistent_topk's CTA grid grows with context.

    Past ~24K decode tokens on a 48-SM GB10 the FilteredTopK fallback needs
    128 KiB smem/block (SM121 has 99 KiB) and the engine dies with a hard
    RuntimeError. Keep the fast path only on >=78-SM parts; small-SM parts use
    top_k_per_row_decode, which is correct everywhere.
    \"\"\"
    from vllm.utils.platform_utils import num_compute_units

    return num_compute_units() >= 78
"""

COND_ANCHOR = """        if current_platform.is_cuda() and select_k in (512, 1024, 2048):
"""

COND_NEW = """        if (
            current_platform.is_cuda()
            and select_k in (512, 1024, 2048)
            and _persistent_topk_fits_device()
        ):
"""


def patch(root: Path, dry_run: bool) -> int:
    path = root / TARGET
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[P2] {TARGET}: already patched; no-op")
        return 0
    for name, anchor, replacement in (
        ("insert SM-count gate helper", HELPER_ANCHOR, HELPER_NEW),
        ("gate the kpool persistent_topk dispatch", COND_ANCHOR, COND_NEW),
    ):
        n = text.count(anchor)
        assert n == 1, (
            f"[P2] ANCHOR FAILED [{name}] in {TARGET}: expected 1, found {n}."
            f"\n--- anchor ---\n{anchor}\n--------------"
        )
        text = text.replace(anchor, replacement, 1)
    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as e:
        raise AssertionError(f"[P2] post-edit ast.parse failed: {e}") from e
    if dry_run:
        print(f"[P2] DRY RUN {TARGET}: 2 edits validated")
    else:
        path.write_text(text, encoding="utf-8")
        print(f"[P2] {TARGET}: 2 edits applied, ast.parse OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    return patch(Path(a.root), a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
