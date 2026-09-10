#!/usr/bin/env python3
"""Static audit of nvidia/GLM-5.3-Flash-NVFP4 before any GPU work.

Model facts this gate encodes (verified 2026-09-09 against the artifact):
  - NVFP4 (modelopt) covers routed experts + dense MLP of layers 0-44 only.
  - The MTP layer 45 is stored BF16 (no scale companions) -- expected, not a fault.
  - Any weight carrying a PARTIAL companion set is a fault (vLLM #54189 class).
  - Vision tensors must exist and must not be quantized (multimodality is a hard
    campaign requirement).

Hard-fail (exit 1) on:
  1. Any layer 0-44 expert/dense weight missing weight_scale/weight_scale_2/input_scale.
  2. Any weight with a partial companion set (some scales, not all).
  3. No MTP-layer tensors under model.language_model.layers.45.*
  4. No vision tensors.
  5. Index references a shard that does not exist.

Usage: python3 audit_checkpoint_static.py <model_dir>
"""
import json
import sys
from collections import Counter
from pathlib import Path

COMPANIONS = (".weight_scale", ".weight_scale_2", ".input_scale")


def is_main_quant_weight(name: str) -> bool:
    """Expert/dense weight in layers 0-44 (the quantized region)."""
    if not name.endswith(".weight"):
        return False
    if ".shared_experts." in name or ".self_attn." in name:
        return False
    if "visual" in name or name.endswith("lm_head.weight"):
        return False
    if ".layers.45." in name:
        return False
    return (
        ".mlp.experts." in name
        or ".mlp.gate_proj.weight" in name
        or ".mlp.up_proj.weight" in name
        or ".mlp.down_proj.weight" in name
        or ".mlp.gate_up_proj.weight" in name
    )


def audit(model_dir: Path) -> dict:
    idx = json.load(open(model_dir / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    cfg = json.load(open(model_dir / "config.json"))
    qc = cfg.get("quantization_config", {})
    hfqc = json.load(open(model_dir / "hf_quant_config.json"))

    main_weights = sorted(n for n in wm if is_main_quant_weight(n))
    missing = []
    for n in main_weights:
        base = n[: -len(".weight")]
        for s in COMPANIONS:
            if base + s not in wm:
                missing.append(base + s)

    # partial companion sets anywhere (including MTP) are a fault
    partial = []
    for n in wm:
        if not n.endswith(".weight"):
            continue
        base = n[: -len(".weight")]
        have = [s for s in COMPANIONS if base + s in wm]
        if have and len(have) != len(COMPANIONS):
            partial.append({"weight": n, "have": have})

    mtp_keys = [n for n in wm if ".layers.45." in n]
    mtp_scales = [n for n in mtp_keys if n.endswith((".weight_scale", ".weight_scale_2"))]
    vision = [n for n in wm if "visual" in n]
    vision_quant = [n for n in vision if n.endswith((".weight_scale", ".weight_scale_2"))]
    dangling = sorted({s for s in wm.values() if not (model_dir / s).exists()})

    def proj_of(n: str) -> str:
        for p in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
            if f".{p}.weight" in n:
                return p
        return "other"

    breakdown = Counter(proj_of(n) for n in main_weights)
    out = {
        "verdict": "PASS",
        "quant_algo": qc.get("quant_algo"),
        "producer": qc.get("producer"),
        "kv_cache_quant_algo": hfqc["quantization"].get("kv_cache_quant_algo"),
        "n_index_keys": len(wm),
        "n_main_quant_weights": len(main_weights),
        "n_missing_companions": len(missing),
        "missing_companions_sample": missing[:10],
        "n_partial_companion_weights": len(partial),
        "partial_sample": partial[:5],
        "breakdown": dict(breakdown),
        "n_mtp_keys": len(mtp_keys),
        "n_mtp_scale_keys": len(mtp_scales),
        "mtp_is_bf16": len(mtp_scales) == 0,
        "n_vision_keys": len(vision),
        "n_vision_quantized": len(vision_quant),
        "n_dangling_shards": len(dangling),
        "dangling_sample": dangling[:5],
    }
    if missing or partial or not mtp_keys or not vision or vision_quant or dangling:
        out["verdict"] = "FAIL"
    print(json.dumps(out, indent=1))
    return out


def main() -> int:
    out = audit(Path(sys.argv[1]))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
