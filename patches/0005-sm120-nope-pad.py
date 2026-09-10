#!/usr/bin/env python3
"""P5 (fallback route): NoPE sparse MLA on the SM120 backend via zero-padding.

Use ONLY if the SM90-on-SM121 route (P1) is unusable. The SM120 backend is the
upstream-intended sparse-MLA path for capability 12, but it requires the packed
fp8_ds_mla cache layout and the compiled `concat_and_cache_mla` kernel asserts
pe_dim == 64; GLM-5.3's sparse layers are NoPE (qk_rope_head_dim == 0). This
port zero-pads the rope lane symmetrically on the KV-write and query side (a
zero rope vector contributes exactly 0 to q_pe . k_pe, bit-exact NoPE) and
passes the indexer's ACTUAL top-k buffer width (kpool widens it past
index_topk, e.g. 2048 -> 2176) to the kernel.

Derived from vllm-project/vllm PR #53969 (hamiltongaianimd, validated on 2x
DGX Spark GB10); the supports_combination effective-width change is
intentionally NOT ported: it rejects this checkpoint's 2048+3 -> 2176 buffer,
while the base image's raw `index_topk != 2048` check accepts it.

Idempotent (marker GLM53-SM120-NOPE-PAD); fails loudly on anchor drift.
Usage: python3 0005-sm120-nope-pad.py [--root ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "GLM53-SM120-NOPE-PAD"
DEFAULT_ROOT = "/usr/local/lib/python3.12/dist-packages"
TARGET = "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"

EDITS = [
    (
        "add do_kv_cache_update with NoPE zero-pad",
        """        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None
""",
        """        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None
        # GLM53-SM120-NOPE-PAD: NoPE models (qk_rope_head_dim == 0) fit the
        # fixed 656-byte fp8_ds_mla tile by zero-padding the 64-wide rope lane;
        # a zero rope vector contributes exactly 0 to q_pe . k_pe.
        self._nope_pad = self.qk_rope_head_dim == 0

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        if self._nope_pad and k_pe.size(-1) == 0:
            k_pe = k_pe.new_zeros((*k_pe.shape[:-1], 64))
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )
""",
    ),
    (
        "forward_mqa: pad q and pass the actual topk buffer width",
        """        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
""",
        """        num_actual_toks = q.shape[0]

        # GLM53-SM120-NOPE-PAD: mirror the KV-write zero-pad on the query side
        # and use the buffer's actual width (kpool tail widens it past the
        # configured index_topk).
        rope_dim = self.qk_rope_head_dim
        if self.qk_rope_head_dim == 0:
            q = torch.nn.functional.pad(q, (0, 64))
            rope_dim = 64

        assert self.topk_indices_buffer is not None
""",
    ),
    (
        "forward_mqa: sparse_mla_top_k = actual buffer width",
        """        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
            max_seq_len=attn_metadata.topk_tokens,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=attn_metadata.topk_tokens,
            kv_scale_format=self.kv_scale_format,
        )
""",
        """        eff_topk = topk_indices_physical.shape[-1]
        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=rope_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
            max_seq_len=eff_topk,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=eff_topk,
            kv_scale_format=self.kv_scale_format,
        )
""",
    ),
]


def patch(root: Path, dry_run: bool) -> int:
    path = root / TARGET
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[P5] {TARGET}: already patched; no-op")
        return 0
    for name, anchor, replacement in EDITS:
        n = text.count(anchor)
        assert n == 1, (
            f"[P5] ANCHOR FAILED [{name}]: expected 1 occurrence, found {n}."
            f"\n--- anchor ---\n{anchor}\n--------------"
        )
        text = text.replace(anchor, replacement, 1)
    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as e:
        raise AssertionError(f"[P5] post-edit ast.parse failed: {e}") from e
    if dry_run:
        print(f"[P5] DRY RUN {TARGET}: {len(EDITS)} edits validated")
    else:
        path.write_text(text, encoding="utf-8")
        print(f"[P5] {TARGET}: {len(EDITS)} edits applied, ast.parse OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    return patch(Path(a.root), a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
