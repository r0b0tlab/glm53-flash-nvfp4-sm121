#!/usr/bin/env python3
"""r0b0bench-vision v1.0 runner.

Runs the frozen 4-suite vision benchmark against an OpenAI-compatible endpoint,
grades deterministically, and writes rows + summary + identity bindings.

Usage:
  python3 run_vision.py --base-url http://127.0.0.1:8000 --data-dir ~/rbv-data \
      --out-dir ~/rbv-out/<run-id> --model ling-3.0-flash-vl-nvfp4-mp \
      --image-id sha256:<id> --profile-id <id> --candidate-id <id> \
      [--only cvbench,mmvp] [--workers 4] [--max-rows N]
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os as _os
CONTRACT = json.loads(Path(_os.environ.get("RVB_CONTRACT", HERE / "benchmark.json")).read_text())

INSTRUCTION = "Answer with the letter or value only."


# ---------------------------------------------------------------- images
def _magic(raw: bytes) -> str:
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if raw[:2] == b"BM":
        return "image/bmp"
    return "image/jpeg"


def img_field_to_bytes(value) -> bytes:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            return bytes(raw)
        path = value.get("path")
        if path:
            return Path(path).read_bytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str) and Path(value).is_file():
        return Path(value).read_bytes()
    raise ValueError(f"unsupported image field: {type(value).__name__}")


def data_url(raw: bytes) -> str:
    return f"data:{_magic(raw)};base64," + base64.b64encode(raw).decode()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- loaders
def load_cvbench(data_dir: Path, max_rows: int | None):
    rows = []
    for split_file in ("test_2d.parquet", "test_3d.parquet"):
        import pyarrow.parquet as pq

        table = pq.read_table(data_dir / "cv-bench" / split_file).to_pylist()
        for r in table:
            rows.append(
                {
                    "id": f"cvbench:{r['idx']}",
                    "suite": "cvbench",
                    "subaxis": r.get("task"),
                    "prompt": r["prompt"],
                    "gold": r["answer"],
                    "image": img_field_to_bytes(r["image"]),
                }
            )
    return rows[:max_rows] if max_rows else rows


def load_mmvp(data_dir: Path, max_rows: int | None):
    base = data_dir / "mmvp"
    rows = []
    with (base / "Questions.csv").open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            idx = int(r["Index"])
            rows.append(
                {
                    "id": f"mmvp:{idx}",
                    "suite": "mmvp",
                    "subaxis": "paired",
                    "prompt": f"{r['Question']}\nOptions: {r['Options']}",
                    "gold": r["Correct Answer"],
                    "image": (base / "MMVP Images" / f"{idx}.jpg").read_bytes(),
                }
            )
    return rows[:max_rows] if max_rows else rows


def load_realworldqa(data_dir: Path, max_rows: int | None):
    import pyarrow.parquet as pq

    rows = []
    for f in ("data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"):
        for r in pq.read_table(data_dir / "realworldqa" / f).to_pylist():
            rows.append(
                {
                    "id": f"rwqa:{len(rows)}",
                    "suite": "realworldqa",
                    "subaxis": "real_world",
                    "prompt": r["question"],
                    "gold": r["answer"],
                    "image": img_field_to_bytes(r["image"]),
                }
            )
    return rows[:max_rows] if max_rows else rows


def load_ocrbench(data_dir: Path, max_rows: int | None):
    import pyarrow.parquet as pq

    rows = []
    for r in pq.read_table(data_dir / "ocrbench" / "data/test-00000-of-00001.parquet").to_pylist():
        ans = r["answer"]
        if isinstance(ans, (list, tuple)):
            gold = list(ans)
        else:
            gold = [ans]
        rows.append(
            {
                "id": f"ocr:{len(rows)}",
                "suite": "ocrbench",
                "subaxis": r.get("question_type"),
                "dataset_name": r.get("dataset"),
                "prompt": r["question"],
                "gold": gold,
                "image": img_field_to_bytes(r["image"]),
            }
        )
    return rows[:max_rows] if max_rows else rows


LOADERS = {
    "cvbench": load_cvbench,
    "mmvp": load_mmvp,
    "realworldqa": load_realworldqa,
    "ocrbench": load_ocrbench,
}


# ---------------------------------------------------------------- graders
def extract_choice(resp: str, allowed: str) -> str | None:
    resp = (resp or "").strip()
    m = re.match(rf"^\(?([{allowed}{allowed.lower()}])\)?[.\s]*$", resp)
    if m:
        return m.group(1).upper()
    m = re.search(rf"\b([{allowed}])\b", resp)
    if m:
        return m.group(1)
    m = re.search(rf"\b([{allowed.lower()}])\b", resp)
    if m:
        return m.group(1).upper()
    return None


def norm_text(s: str) -> str:
    s = (s or "").lower().strip().replace("\n", " ")
    s = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)
    return re.sub(r"\s+", " ", s)


def grade(row: dict, resp: str):
    suite = row["suite"]
    gold = row["gold"]
    if suite in ("cvbench", "mmvp"):
        allowed = "AB" if suite == "mmvp" else "ABCDEF"
        want = re.sub(r"[^A-Za-z]", "", str(gold)).upper()[:1]
        got = extract_choice(resp, allowed)
        return got == want, got
    if suite == "realworldqa":
        g = str(gold).strip()
        if re.fullmatch(r"[A-Da-d]", g):
            got = extract_choice(resp, "ABCD")
            return got == g.upper(), got
        gnorm = norm_text(g)
        rnorm = norm_text(resp)
        return (gnorm == rnorm or (gnorm and gnorm in rnorm)), resp.strip()[:40]
    if suite == "ocrbench":
        # official rule: lowercased answer containment; HME100k strips spaces
        pred = (resp or "").lower().strip().replace("\n", " ")
        hme = row.get("dataset_name") == "HME100k"
        if hme:
            pred = pred.replace(" ", "")
        for a in gold:
            a2 = str(a).lower().strip().replace("\n", " ")
            if hme:
                a2 = a2.replace(" ", "")
            if a2 and a2 in pred:
                return True, resp.strip()[:40]
        return False, resp.strip()[:40]
    raise ValueError(suite)


# ---------------------------------------------------------------- serving
def chat(base_url: str, model: str, prompt: str, image: bytes, max_tokens: int, timeout: float):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"reasoning_effort": "max"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url(image)}},
                    {"type": "text", "text": prompt + "\n" + INSTRUCTION},
                ],
            }
        ],
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    el = time.perf_counter() - t0
    msg = d["choices"][0]["message"]
    return (msg.get("content") or ""), d.get("usage") or {}, el, d.get("choices", [{}])[0].get("finish_reason")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((centre - half) / denom, 4), round((centre + half) / denom, 4)]


def mmvp_paired(rows: list[dict]) -> dict:
    """Paired metric: both questions of a CLIP-blind pair must be correct."""
    by_id = {r["id"]: r for r in rows}
    pairs = 0
    paired = 0
    for i in range(1, 301, 2):
        a, b = by_id.get(f"mmvp:{i}"), by_id.get(f"mmvp:{i+1}")
        if a and b:
            pairs += 1
            paired += int(a["passed"] and b["passed"])
    return {
        "pairs": pairs,
        "both_correct": paired,
        "paired_accuracy": round(paired / pairs, 4) if pairs else None,
    }


def ocrbench_official(rows: list[dict]) -> dict:
    """Official OCRBench per-question-type tallies (score, n)."""
    out: dict[str, list[int]] = {}
    for r in rows:
        t = r.get("subaxis") or "?"
        out.setdefault(t, [0, 0])
        out[t][0] += int(r["passed"])
        out[t][1] += 1
    return {t: {"score": v[0], "n": v[1]} for t, v in sorted(out.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--profile-id", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    suites = [s["id"] for s in CONTRACT["suites"]]
    if args.only:
        suites = [s for s in args.only.split(",") if s in suites]
    if not suites:
        print("no suites selected", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "rows.jsonl"
    per_suite = {}
    all_rows = []

    for suite in suites:
        spec = next(s for s in CONTRACT["suites"] if s["id"] == suite)
        print(f"=== {suite}: loading ===", flush=True)
        rows = LOADERS[suite](args.data_dir, args.max_rows or None)
        print(f"=== {suite}: {len(rows)} rows, running with {args.workers} workers ===", flush=True)
        max_tokens = CONTRACT["protocol"]["max_tokens"][suite]
        done = 0

        def work(row):
            resp, usage, el, finish = chat(
                args.base_url, args.model, row["prompt"], row["image"], max_tokens, args.timeout
            )
            passed, extracted = grade(row, resp)
            return {
                "id": row["id"],
                "suite": suite,
                "subaxis": row.get("subaxis"),
                "dataset_name": row.get("dataset_name"),
                "prompt_sha256": sha256(row["prompt"].encode()),
                "image_sha256": sha256(row["image"]),
                "gold": row["gold"],
                "response": resp,
                "extracted": extracted,
                "passed": bool(passed),
                "finish_reason": finish,
                "elapsed_seconds": round(el, 3),
                "usage": usage,
                "image_id": args.image_id,
                "profile_id": args.profile_id,
                "candidate_id": args.candidate_id,
                "model": args.model,
            }

        t0 = time.perf_counter()
        with rows_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(work, r): r for r in rows}
            for fut in as_completed(futs):
                rec = fut.result()
                all_rows.append(rec)
                out.write(json.dumps(rec, sort_keys=True) + "\n")
                out.flush()
                done += 1
                if done % 100 == 0:
                    print(f"  {suite}: {done}/{len(rows)}", flush=True)
        wall = time.perf_counter() - t0
        suite_rows = [r for r in all_rows if r["suite"] == suite]
        k = sum(1 for r in suite_rows if r["passed"])
        per_suite[suite] = {
            "rows": len(suite_rows),
            "correct": k,
            "accuracy": round(k / len(suite_rows), 4) if suite_rows else None,
            "wilson95": wilson(k, len(suite_rows)),
            "wall_seconds": round(wall, 1),
            "license": spec["license"],
            "revision": spec["revision"],
        }
        if spec.get("subaxis_field"):
            axes = {}
            for r in suite_rows:
                axes.setdefault(r.get("subaxis") or "?", []).append(r)
            per_suite[suite]["subaxes"] = {
                a: {
                    "n": len(v),
                    "correct": sum(1 for x in v if x["passed"]),
                    "accuracy": round(sum(1 for x in v if x["passed"]) / len(v), 4),
                }
                for a, v in sorted(axes.items())
            }
        if suite == "mmvp":
            per_suite[suite]["paired"] = mmvp_paired(suite_rows)
        if suite == "ocrbench":
            per_suite[suite]["official_type_scores"] = ocrbench_official(suite_rows)

    total = len(all_rows)
    correct = sum(1 for r in all_rows if r["passed"])
    summary = {
        "schema": "r0b0bench.vision.v1",
        "benchmark": CONTRACT["name"],
        "version": CONTRACT["version"],
        "protocol": CONTRACT["protocol"],
        "model": args.model,
        "image_id": args.image_id,
        "profile_id": args.profile_id,
        "candidate_id": args.candidate_id,
        "runner_sha256": sha256((HERE / "run_vision.py").read_bytes()),
        "contract_sha256": sha256(Path(_os.environ.get("RVB_CONTRACT", HERE / "benchmark.json")).read_bytes()),
        "suites": per_suite,
        "total_rows": total,
        "total_correct": correct,
        "total_accuracy": round(correct / total, 4) if total else None,
        "total_wilson95": wilson(correct, total),
        "mean_row_seconds": round(statistics.mean([r["elapsed_seconds"] for r in all_rows]), 3) if all_rows else None,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
