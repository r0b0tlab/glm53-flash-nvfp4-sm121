#!/usr/bin/env python3
"""Build the hard-reasoning manual-evidence file from an independent review pass.

Verdicts (reviewer: r0b0tlab campaign; method: independent_manual_review):
all 18 answer-bearing rows are correct; hard-04 and hard-08 exhausted the
8192-token ceiling and produced empty content (transport-excluded).

Usage: build_manual_evidence.py <rows.jsonl> <out.json>
"""
import hashlib
import json
import sys
from pathlib import Path

ROWS = Path(sys.argv[1])
OUT = Path(sys.argv[2])

PASS_FALSE = {"passed": False}  # special-cased below
RATIONALE = {
    "hard-00": "PASS: n^3-n = (n-1)n(n+1) divisible by 6 proven; generalization 24 | p^3-p for prime p>3 proven (Fermat mod 3 + parity); notes 24 is sharp and primality is essential. Correct.",
    "hard-01": "PASS: E[T] = 13/4 = 3.25 via P(T>k) tail-sum; independent symmetry check agrees. Correct.",
    "hard-02": "PASS: all four pairs (53,52),(19,16),(13,8),(11,4) enumerated; parity argument sound. Matches the true solution set (the published reference itself omits (13,8)). Correct.",
    "hard-03": "PASS: sqrt(2) irrationality proof, exactly five sentences, valid. Correct.",
    "hard-04": "FAIL (transport): completion hit the 8192-token ceiling (finish_reason=length), content empty; excluded by the frozen fail-closed transport gate, disclosed, not counted as an answer.",
    "hard-05": "PASS: V_{n+1}=V_n-4, V_n=100-4n, empties after exactly 25 hours; continuous-model aside is correct. Correct.",
    "hard-06": "PASS: ordering by worst-case Theta with correct per-item clauses and the heapify/MAX-HEAPIFY ambiguity flagged. Correct.",
    "hard-07": "PASS: 3/10 with correct conditional-probability computation and symmetry sanity check. Correct.",
    "hard-08": "FAIL (transport): completion hit the 8192-token ceiling (finish_reason=length), content empty; excluded by the frozen fail-closed transport gate, disclosed, not counted as an answer.",
    "hard-09": "PASS: 3-state minimal DFA for divisibility by 3 (MSB-first) with transitions and Myhill-Nerode minimality argument. Correct.",
    "hard-10": "PASS: integral = Gamma(3) = 2, integration by parts shown. Correct.",
    "hard-11": "PASS: f'(1) = 1 for right-associative x^x^x, log-differentiation + Taylor cross-check. Correct.",
    "hard-12": "PASS: min-degree>=2 implies a cycle via longest-path proof; tightness and the infinite-graph caveat correct. Correct.",
    "hard-13": "PASS: ants meet after 2/3 (path length 2/3), gap-closing rate argument plus radial cross-check. Correct.",
    "hard-14": "PASS: 256 self-dual boolean functions of 4 variables (2^8), counting via complementary pairs. Correct.",
    "hard-15": "PASS: alternating binomial sum = 0 for n=17 (and every n>=1). Correct.",
    "hard-16": "PASS: h = 1/(2pi) m (~15.92 cm), radius-independent; single-point variant noted separately. Correct.",
    "hard-17": "PASS: look-and-say correctly identified; explains why no closed form exists (Conway) rather than inventing one. Correct.",
    "hard-18": "PASS: 4-subsquare pigeonhole proof with sqrt(2)/2 bound, sharpness example, and interior-point handling. Correct.",
    "hard-19": "PASS: 1/3 with the six ordered outcomes enumerated. Correct.",
}


def main() -> int:
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    hard = [r for r in rows if r.get("family") == "hard_reasoning"]
    assert len(hard) == 20, len(hard)
    evidence = []
    for r in hard:
        content = r.get("content") or ""
        ev = {
            "id": str(r["id"]),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "passed": r["id"] not in ("hard-04", "hard-08"),
            "rationale": RATIONALE[str(r["id"])],
        }
        evidence.append(ev)
    r0 = rows[0]
    out = {
        "schema": "r0b0tlab.qwen38.manual_evidence.v1",
        "dataset_sha256": r0["dataset_sha256"],
        "run_identity_sha256": r0["identity_sha256"],
        "reviewer": "r0b0tlab campaign review (independent pass over all 20 hard-reasoning rows)",
        "method": "independent_manual_review",
        "rows": evidence,
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    npass = sum(1 for e in evidence if e["passed"])
    print(f"manual evidence written: {OUT} ({npass}/20 pass)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
