#!/usr/bin/env python3
"""P4: teach the GLM-5-Next KV layout about the DFlash2 drafter's SWA layers.

Without this, `_get_kv_cache_groups_glm5_next` bails to the generic uniform-page
path the moment a drafter registers SlidingWindowSpec layers; that path cannot
serve this model (MLA/indexer/tail page sizes are mutually hostile) and boot
dies in warmup. The fix partitions the drafter's SWA layers out before the
all-MLA check, builds the GLM groups unchanged, and appends ONE drafter group
that slot-shares the MLA tensors when the geometry fits exactly (never
page_size_padded -- a padded spec routes the runner into a strided view that
overruns under kernel-block splitting). Standalone mode (compact per-layer
tensors) is the fallback for geometries that cannot fill the MLA page exactly.

Mechanism informed by the public SM121 GLM+DFlash2 work
(tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark overlay-dflash2); anchors
re-derived for this base image (vLLM 0.28.1rc1.dev580+g385dce36b), whose KV
layout code uses packed per-layer tensors (add_tensor with offset/strides).

Inert without a drafter (draft_group stays None).

Idempotent (marker GLM53-DFLASH2-DRAFTER-GROUP); fails loudly on anchor drift.
Usage: python3 0004-glm-dflash2-drafter-group.py [--root ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "GLM53-DFLASH2-DRAFTER-GROUP"
DEFAULT_ROOT = "/usr/local/lib/python3.12/dist-packages"
TARGET = "vllm/v1/core/kv_cache_utils.py"

EDITS = []

# --- groups: partition drafter specs out of attn_specs ----------------------
EDITS.append((
    "groups: partition drafter SlidingWindowSpec layers out",
    """    tail_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, KpoolTailSpec)
    }
    attn_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not isinstance(spec, (MambaSpec, KpoolTailSpec))
    }
""",
    """    tail_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, KpoolTailSpec)
    }
    # GLM53-DFLASH2-DRAFTER-GROUP: a spec-decode drafter (DFlash2) adds plain
    # SlidingWindowSpec layers on top of the GLM-5-Next hybrid. Partition them
    # out (exact type: KpoolTailSpec subclasses SlidingWindowSpec) so they do
    # not disqualify the model from this fast path; they are appended as one
    # extra group below.
    draft_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if type(spec) is SlidingWindowSpec
    }
    attn_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not isinstance(spec, (MambaSpec, KpoolTailSpec))
        and type(spec) is not SlidingWindowSpec
    }
"""))

# --- groups: build + append the drafter group -------------------------------
EDITS.append((
    "groups: build + append drafter group (exact-fit / standalone)",
    """    mamba_grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
    for index, name in enumerate(mamba_specs):
        mamba_grouped_names[index % num_groups].append(name)

    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
    )
""",
    """    mamba_grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
    for index, name in enumerate(mamba_specs):
        mamba_grouped_names[index % num_groups].append(name)

    # GLM53-DFLASH2-DRAFTER-GROUP: one extra group for the drafter's
    # SlidingWindowSpec layers, appended LAST so existing group ids stay
    # stable. NEVER page_size_padded (a padded spec routes the runner into the
    # strided-view reshape, invalid when the backend splits the manager block
    # into smaller kernel blocks). Exact fit rescales the drafter's block so
    # its real page equals the MLA page; the runner then takes the ordinary
    # contiguous reshape and drafter layer i co-owns MLA tensor i.
    draft_group: KVCacheGroupSpec | None = None
    if draft_specs:
        any_draft = next(iter(draft_specs.values()))
        assert all(spec == any_draft for spec in draft_specs.values()), (
            "drafter SlidingWindowSpec layers must share one spec"
        )
        draft_bytes_per_token = any_draft.page_size_bytes // any_draft.block_size
        mla_block = mla_specs[mla_names[0]].block_size
        fit_block = (
            mla_page // draft_bytes_per_token
            if mla_page % draft_bytes_per_token == 0
            else 0
        )
        if (
            fit_block
            and fit_block % 64 == 0
            and (fit_block % mla_block == 0 or mla_block % fit_block == 0)
            and len(draft_specs) <= len(mla_names)
        ):
            new_draft_specs: dict[str, KVCacheSpec] = {
                name: replace(spec, block_size=fit_block)
                for name, spec in draft_specs.items()
            }
        else:
            new_draft_specs = dict(draft_specs)
        draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_draft_specs)
        assert draft_uniform is not None
        draft_group = KVCacheGroupSpec(list(new_draft_specs), draft_uniform)

    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
        + ([draft_group] if draft_group is not None else [])
    )
"""))

# --- layout: return-type annotation gains draft_group -----------------------
EDITS.append((
    "layout: return-type annotation gains draft_group",
    """        list[str],
        int,
    ]
    | None
):
""",
    """        list[str],
        int,
        KVCacheGroupSpec | None,
    ]
    | None
):
"""))

# --- layout: detect the drafter SWA group -----------------------------------
EDITS.append((
    "layout: detect drafter SWA uniform group",
    """    attn_group: KVCacheGroupSpec | None = None
    tail_group: KVCacheGroupSpec | None = None
    for group in uniform_groups:
        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):
            attn_group = group
        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):
            tail_group = group
""",
    """    attn_group: KVCacheGroupSpec | None = None
    tail_group: KVCacheGroupSpec | None = None
    draft_group: KVCacheGroupSpec | None = None
    for group in uniform_groups:
        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):
            attn_group = group
        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):
            tail_group = group
        elif inner and all(
            type(spec) is SlidingWindowSpec for spec in inner.values()
        ):
            # GLM53-DFLASH2-DRAFTER-GROUP: the drafter's SWA group (validated
            # below once mla_page is known).
            draft_group = group
"""))

# --- layout: validate the drafter group -------------------------------------
EDITS.append((
    "layout: validate drafter (uniform page, never padded)",
    """    if any(group.kv_cache_spec.page_size_bytes != mla_page for group in mamba_groups):
        return None

    tail_names: list[str] = []
""",
    """    if any(group.kv_cache_spec.page_size_bytes != mla_page for group in mamba_groups):
        return None
    if draft_group is not None:
        # GLM53-DFLASH2-DRAFTER-GROUP: one uniform page across drafter layers
        # and NEVER page_size_padded. page == mla_page means exact-fit
        # slot-sharing (needs one MLA tensor per drafter layer).
        draft_inner = cast(
            UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
        ).kv_cache_specs
        draft_pages = {spec.page_size_bytes for spec in draft_inner.values()}
        if len(draft_pages) != 1:
            return None
        if any(spec.page_size_padded is not None for spec in draft_inner.values()):
            return None
        if (
            draft_pages.pop() == mla_page
            and len(draft_group.layer_names) > len(mla_names)
        ):
            return None

    tail_names: list[str] = []
"""))

# --- layout: return draft_group ---------------------------------------------
EDITS.append((
    "layout: return draft_group (9th element)",
    """    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_page,
        tail_names,
        tail_page,
    )
""",
    """    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_page,
        tail_names,
        tail_page,
        draft_group,
    )
"""))

# --- _get_kv_cache_bytes_per_block: 9-tuple + standalone bytes --------------
EDITS.append((
    "_get_kv_cache_bytes_per_block: standalone drafter bytes",
    """    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5_layout
        return len(mla_names) * mla_page + len(idx_names) * idx_page
""",
    """    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        (
            _,
            _,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            _,
            _,
            draft_group,
        ) = glm5_layout
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        if draft_group is not None:
            # GLM53-DFLASH2-DRAFTER-GROUP: exact-fit drafter adds no bytes
            # (rides the MLA tensors); standalone adds one page per layer.
            draft_page = next(
                iter(
                    cast(
                        UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
                    ).kv_cache_specs.values()
                )
            ).page_size_bytes
            if draft_page != mla_page:
                per_block += len(draft_group.layer_names) * draft_page
        return per_block
"""))

# --- get_kv_cache_config_from_groups: destructure + per-block cost ----------
EDITS.append((
    "config: destructure + per-block cost incl. standalone drafter",
    """    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
        ) = glm5_layout
        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        num_blocks = may_override_num_blocks(
            vllm_config, available_memory // bytes_per_block
        )
""",
    """    if (glm5_layout := _glm5_next_tensor_layout(kv_cache_groups)) is not None:
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
            draft_group,
        ) = glm5_layout
        draft_names: list[str] = []
        draft_inner: dict[str, KVCacheSpec] = {}
        draft_page = 0
        draft_shared = False
        if draft_group is not None:
            draft_names = list(draft_group.layer_names)
            draft_inner = cast(
                UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
            ).kv_cache_specs
            draft_page = next(iter(draft_inner.values())).page_size_bytes
            draft_shared = draft_page == mla_page
        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        if draft_names and not draft_shared:
            # GLM53-DFLASH2-DRAFTER-GROUP (standalone): drafter pages are part
            # of every block's byte cost.
            bytes_per_block += len(draft_names) * draft_page
        num_blocks = may_override_num_blocks(
            vllm_config, available_memory // bytes_per_block
        )
"""))

# --- get_kv_cache_config_from_groups: drafter tensors -----------------------
EDITS.append((
    "config: drafter tensors (exact-fit alias / standalone)",
    """                add_tensor(tail_name, tail_specs[tail_name], offset)

        return KVCacheConfig(
""",
    """                add_tensor(tail_name, tail_specs[tail_name], offset)

        if draft_names:
            # GLM53-DFLASH2-DRAFTER-GROUP: exact fit -> drafter layer i rides
            # MLA tensor i at the same offset (disjoint block ids, contiguous
            # view); standalone -> compact tensors after the idx/tail region.
            draft_base = idx_base + len(idx_names) * idx_page * num_blocks
            for index, draft_name in enumerate(draft_names):
                offset = (
                    index * mla_page * num_blocks
                    if draft_shared
                    else draft_base + index * draft_page * num_blocks
                )
                add_tensor(draft_name, draft_inner[draft_name], offset)

        return KVCacheConfig(
"""))

# --- _max_memory_usage_bytes_from_groups: destructure -----------------------
EDITS.append((
    "max-mem: destructure gains draft_group",
    """        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
        ) = glm5_layout
        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
""",
    """        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
            draft_group,
        ) = glm5_layout
        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
"""))

# --- _max_memory_usage_bytes_from_groups: drafter demand --------------------
EDITS.append((
    "max-mem: charge drafter block-id demand + standalone bytes",
    """        if tail_names:
            total_blocks += 1
        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)
""",
    """        if tail_names:
            total_blocks += 1
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        if draft_group is not None:
            # GLM53-DFLASH2-DRAFTER-GROUP: charge the drafter's window-bounded
            # block-id demand; a standalone drafter also adds its pages to
            # every block's byte cost.
            draft_uniform = draft_group.kv_cache_spec
            assert isinstance(draft_uniform, UniformTypeKVCacheSpecs)
            total_blocks += draft_uniform.max_memory_usage_pages(vllm_config)
            draft_page = next(
                iter(draft_uniform.kv_cache_specs.values())
            ).page_size_bytes
            if draft_page != mla_page:
                per_block += len(draft_group.layer_names) * draft_page
        return total_blocks * per_block
"""))


def patch(root: Path, dry_run: bool) -> int:
    path = root / TARGET
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[P4] {TARGET}: already patched; no-op")
        return 0
    for required in (
        "def _get_kv_cache_groups_glm5_next",
        "def _glm5_next_tensor_layout",
        "def _get_kv_cache_bytes_per_block",
        "SlidingWindowSpec",
        "UniformTypeKVCacheSpecs",
    ):
        assert required in text, f"[P4] precheck failed: {required!r} missing"
    for name, anchor, replacement in EDITS:
        n = text.count(anchor)
        assert n == 1, (
            f"[P4] ANCHOR FAILED [{name}]: expected 1 occurrence, found {n}."
            f"\n--- anchor ---\n{anchor}\n--------------"
        )
        text = text.replace(anchor, replacement, 1)
    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as e:
        raise AssertionError(f"[P4] post-edit ast.parse failed: {e}") from e
    if dry_run:
        print(f"[P4] DRY RUN {TARGET}: {len(EDITS)} edits validated")
    else:
        path.write_text(text, encoding="utf-8")
        print(f"[P4] {TARGET}: {len(EDITS)} edits applied, ast.parse OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    return patch(Path(a.root), a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
