#!/usr/bin/env python3
"""Deterministic vision probes for the VL gate. Stdlib only (zlib PNG writer)."""
import struct
import zlib
from pathlib import Path

import os as _os
OUT = Path(_os.environ.get("GLM53_MEDIA", Path.home() / "glm53-flash-nvfp4" / "media"))
OUT.mkdir(parents=True, exist_ok=True)
W = H = 448

SEG = {  # 7-segment digit -> segments (a,b,c,d,e,f,g)
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgecd", "7": "abc", "8": "abcdefg", "9": "abfgcd",
}


def png(path, pix):
    raw = bytearray()
    for y in range(H):
        raw.append(0)
        for x in range(W):
            raw += bytes(pix(x, y))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


def squares(n, path):
    centers = [(96, 96), (256, 96), (96, 256), (256, 256), (176, 176), (336, 336)][:n]

    def pix(x, y):
        for cx, cy in centers:
            if abs(x - cx) < 40 and abs(y - cy) < 40:
                return (220, 30, 30)
        return (255, 255, 255)

    png(path, pix)


def digits(number, path):
    global W, H  # noqa: PLW0603 - single-module generator
    W, H = 4 * 64 + 32, 128
    buf = [[(255, 255, 255)] * W for _ in range(H)]
    for i, d in enumerate(number):
        x0 = 16 + i * 64
        w, h, t = 44, 96, 10
        on = SEG[d]
        segs = {
            "a": (t, 0, w - t, t), "g": (t, h // 2 - t // 2, w - t, h // 2 + t // 2),
            "d": (t, h - t, w - t, h), "f": (0, t, t, h // 2), "b": (w - t, t, w, h // 2),
            "e": (0, h // 2, t, h - t), "c": (w - t, h // 2, w, h - t),
        }
        for s in on:
            x1, y1, x2, y2 = segs[s]
            for y in range(y1, y2):
                for x in range(x1, x2):
                    buf[8 + y][x0 + x] = (0, 0, 0)
    raw = bytearray()
    for row in buf:
        raw.append(0)
        for px in row:
            raw += bytes(px)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


def bars(path):
    global W, H
    W, H = 448, 320
    heights = {"red": 100, "green": 220, "blue": 160}
    cols = {"red": (220, 30, 30), "green": (30, 170, 30), "blue": (30, 30, 220)}
    order = ["red", "green", "blue"]

    def pix(x, y):
        for i, name in enumerate(order):
            x1, x2 = 60 + i * 120, 60 + i * 120 + 80
            if x1 <= x < x2 and y >= H - heights[name]:
                return cols[name]
        return (255, 255, 255)

    png(path, pix)


def solid(color, path):
    global W, H
    W = H = 224
    png(path, lambda x, y: color)


def main():
    for n in (3, 4, 5, 6):
        squares(n, OUT / f"count_{n}.png")
    digits("4271", OUT / "digits_4271.png")
    bars(OUT / "bars.png")
    for i, c in enumerate([(220, 30, 30), (30, 170, 30), (30, 30, 220), (230, 210, 30)]):
        solid(c, OUT / f"frame_{i}.png")
    print("MEDIA_OK", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
