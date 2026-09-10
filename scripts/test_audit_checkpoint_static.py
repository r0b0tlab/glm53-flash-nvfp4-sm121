#!/usr/bin/env python3
"""Unit test for audit_checkpoint_static. Builds a synthetic tiny checkpoint:
  - complete: every NVFP4 weight has all 3 companions -> PASS
  - missing input_scale on one module -> FAIL
  - no MTP layer -> FAIL
  - no vision tensors -> FAIL
Run: python3 test_audit_checkpoint_static.py   (expect "AUDIT_TEST_PASS", rc 0)
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "audit_checkpoint_static.py"


def make_dir(tmp: Path, drop=(), shard="model-00001-of-00001.safetensors"):
    names = [
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.3.mlp.experts.0.up_proj.weight",
        "model.language_model.layers.3.mlp.experts.0.down_proj.weight",
        "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.language_model.layers.3.self_attn.q_a_proj.weight",
        "model.language_model.layers.45.mlp.experts.0.gate_proj.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "lm_head.weight",
    ]
    wm = {}
    for n in names:
        wm[n] = shard
        if n.endswith(".weight") and (
            ".mlp.experts." in n or ".mlp.gate_proj.weight" in n
            or ".mlp.up_proj.weight" in n or ".mlp.down_proj.weight" in n
        ):
            if ".shared_experts." in n or ".self_attn." in n:
                continue
            base = n[: -len(".weight")]
            for s in (".weight_scale", ".weight_scale_2", ".input_scale"):
                if f"{base}{s}" not in drop:
                    wm[base + s] = shard
    for d in drop:
        wm.pop(d, None)
    (tmp / shard).write_bytes(b"\0")
    json.dump({"metadata": {}, "weight_map": wm},
              open(tmp / "model.safetensors.index.json", "w"))
    json.dump({"quantization_config": {"quant_algo": "NVFP4", "producer": {"name": "modelopt"}}},
              open(tmp / "config.json", "w"))
    json.dump({"quantization": {"kv_cache_quant_algo": "FP8"}},
              open(tmp / "hf_quant_config.json", "w"))


def run(tmp: Path) -> int:
    return subprocess.run([sys.executable, str(AUDIT), str(tmp)],
                          capture_output=True, text=True).returncode


def main():
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        make_dir(t)
        assert run(t) == 0, "complete checkpoint must PASS"
        t2 = Path(d) / "a"; t2.mkdir()
        make_dir(t2, drop=("model.language_model.layers.3.mlp.experts.0.gate_proj.input_scale",))
        assert run(t2) == 1, "missing input_scale must FAIL"
        t3 = Path(d) / "b"; t3.mkdir()
        make_dir(t3)
        wm = json.load(open(t3 / "model.safetensors.index.json"))["weight_map"]
        wm = {k: v for k, v in wm.items() if ".layers.45." not in k}
        json.dump({"metadata": {}, "weight_map": wm},
                  open(t3 / "model.safetensors.index.json", "w"))
        assert run(t3) == 1, "missing MTP layer must FAIL"
        t4 = Path(d) / "c"; t4.mkdir()
        make_dir(t4)
        wm = json.load(open(t4 / "model.safetensors.index.json"))["weight_map"]
        wm = {k: v for k, v in wm.items() if "visual" not in k}
        json.dump({"metadata": {}, "weight_map": wm},
                  open(t4 / "model.safetensors.index.json", "w"))
        assert run(t4) == 1, "missing vision tensors must FAIL"
    print("AUDIT_TEST_PASS")


if __name__ == "__main__":
    main()
