#!/usr/bin/env python3
"""P3: DFlash2/EAGLE-3 aux hidden-state capture for GLM-5.3-Flash.

Adds SupportsEagle3 + EagleModelMixin to the Glm5Next target and captures aux
hidden states at the drafter's tap layers with the mHC contraction GLM needs
(each layer defers its final hc_post to the next layer's fused pre, so mid-stack
capture must materialize layer.hc_post(...) then contract the 4 streams to one
4096-wide tensor; hc_contract == mean(dim=1)).

Inert without a drafter: aux_hidden_state_layers defaults to () so the capture
branch never fires and forward returns a plain tensor.

Mechanism informed by the public SM121 GLM+DFlash2 work
(tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark overlay-dflash2); anchors
re-derived for this base image (vLLM 0.28.1rc1.dev580+g385dce36b).

Idempotent (marker GLM53-DFLASH2-AUX-CAPTURE); fails loudly on anchor drift.
Usage: python3 0003-glm-dflash2-aux-capture.py [--root ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "GLM53-DFLASH2-AUX-CAPTURE"
DEFAULT_ROOT = "/usr/local/lib/python3.12/dist-packages"
TARGET = "vllm/models/glm5next/nvidia/model.py"

EDITS = [
    (
        "import EagleModelMixin + SupportsEagle3",
        """from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
""",
        """from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)
""",
    ),
    (
        "Glm5NextModel gains EagleModelMixin",
        """class Glm5NextModel(nn.Module):
""",
        """class Glm5NextModel(nn.Module, EagleModelMixin):  # GLM53-DFLASH2-AUX-CAPTURE
""",
    ),
    (
        "decoder loop: aux capture + mHC contraction",
        """        for layer in self._active_layers:
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
""",
        """        # GLM53-DFLASH2-AUX-CAPTURE: mirrors DeepseekV4Model.forward. The
        # runner converts DFlash target_layer_ids to id+1 semantics, so
        # `idx + 1 in aux_hidden_state_layers` captures the OUTPUT of 0-based
        # layer idx.
        aux_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(self._active_layers, start=self.start_layer):
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
            if idx + 1 in self.aux_hidden_state_layers:
                if post is not None:
                    # Mid-stack mHC layer: its final hc_post is deferred to
                    # the next layer's fused pre. Materialize it (pure op) and
                    # contract the streams exactly like the last layer does.
                    aux_recon = layer.hc_post(hidden_states, residual, post, comb)
                    aux_hidden_state = hc_contract(aux_recon, layer.n)
                else:
                    aux_hidden_state = hidden_states
                if self.is_sequence_parallel:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[
                        :full_num_tokens
                    ]
                aux_hidden_states.append(aux_hidden_state)
""",
    ),
    (
        "forward tail: return (hidden_states, aux_hidden_states)",
        """        hidden_states = self.norm(hidden_states)
        return hidden_states
""",
        """        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states
""",
    ),
    (
        "Glm5NextForCausalLM declares SupportsEagle3",
        """class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
""",
        """class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, SupportsEagle3, MixtureOfExperts, IsHybrid
):
""",
    ),
    (
        "Glm5NextForConditionalGeneration declares SupportsEagle3",
        """class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, MixtureOfExperts
):
""",
        """class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
):
""",
    ),
]


def patch(root: Path, dry_run: bool) -> int:
    path = root / TARGET
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[P3] {TARGET}: already patched; no-op")
        return 0
    for required in ("class Glm5NextModel", "hc_contract", "sp_all_gather"):
        assert required in text, f"[P3] precheck failed: {required!r} missing"
    for name, anchor, replacement in EDITS:
        n = text.count(anchor)
        assert n == 1, (
            f"[P3] ANCHOR FAILED [{name}]: expected 1 occurrence, found {n}."
            f"\n--- anchor ---\n{anchor}\n--------------"
        )
        text = text.replace(anchor, replacement, 1)
    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as e:
        raise AssertionError(f"[P3] post-edit ast.parse failed: {e}") from e
    if dry_run:
        print(f"[P3] DRY RUN {TARGET}: {len(EDITS)} edits validated")
    else:
        path.write_text(text, encoding="utf-8")
        print(f"[P3] {TARGET}: {len(EDITS)} edits applied, ast.parse OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    return patch(Path(a.root), a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
