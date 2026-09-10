#!/usr/bin/env python3
"""Build the sanitized evidence bundle + MANIFEST.sha256 for publication.

- Copies evidence/ (text formats only) into a staging dir,
- redacts private IPs and home paths,
- writes MANIFEST.sha256 over the staged files.

Usage: package_evidence.py <src_evidence_dir> <out_dir>
"""
from __future__ import annotations

import hashlib
import re
import shutil
import sys
from pathlib import Path

TEXT_SUFFIXES = {".json", ".md", ".txt", ".log", ".jsonl"}

# Public-bundle policy (ling precedent): no raw prompts/responses, no failed-run
# noise. Drop raw row files and model-output dumps; keep verdicts, summaries,
# metrics, and gate evidence.
SKIP_NAME_SUFFIXES = (".rows.jsonl",)
SKIP_NAMES = {"rows.jsonl", "glm53-q200v2-20260910T053201Z.rows.jsonl"}
SKIP_DIRS = {"spec"}                # lossless records carry full responses
SKIP_PATH_PARTS = {("bfcl-hard20", "bfcl-run", "result"),
                   ("bfcl-hard20", "bfcl-run", "score")}
SKIP_DIR_NAMES = {"20260910T053136Z"}  # failed first Q200 attempt (404 arm)

REDACTIONS = [
    (re.compile(r"100\.\d+\.\d+\.\d+"), "<host>"),
    (re.compile(r"192\.168\.\d+\.\d+"), "<lan>"),
    (re.compile(r"/home/[a-z0-9_]+"), "<home>"),
    (re.compile(r"/Users/[A-Za-z0-9_]+"), "<home>"),
    (re.compile(r"tail[0-9a-f]+\.ts\.net"), "<mesh>"),
]


def main() -> int:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    n = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = p.relative_to(src)
        if rel.name == "MANIFEST.sha256":
            continue
        if rel.name.endswith(SKIP_NAME_SUFFIXES) or rel.name in SKIP_NAMES:
            continue
        if any(part in SKIP_DIR_NAMES for part in rel.parts):
            continue
        if rel.parts and rel.parts[0] in SKIP_DIRS:
            continue
        if any(rel.parts[: len(pp)] == pp for pp in SKIP_PATH_PARTS):
            continue
        data = p.read_text(errors="replace")
        for rx, rep in REDACTIONS:
            data = rx.sub(rep, data)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(data)
        n += 1
    manifest = out / "MANIFEST.sha256"
    lines = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.relative_to(out)}")
    manifest.write_text("\n".join(lines) + "\n")
    print(f"staged {n} files -> {out}; manifest lines: {len(lines)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
