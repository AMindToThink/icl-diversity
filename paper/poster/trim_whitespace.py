# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Trim uniform white border from a PNG, leaving a small pad.

Usage: uv run trim_whitespace.py <in.png> <out.png> [pad_px]
Used by build_poster.sh to crop the Fig 1 pipeline render (which is produced
on a borderless article page because this TeX dist lacks standalone/preview).
"""
import sys

from PIL import Image, ImageChops


def trim(in_path: str, out_path: str, pad: int = 24) -> None:
    im = Image.open(in_path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox is None:
        raise SystemExit(f"{in_path} is blank; nothing to trim")
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(im.width, right + pad)
    bottom = min(im.height, bottom + pad)
    cropped = im.crop((left, top, right, bottom))
    cropped.save(out_path)
    print(f"trimmed {in_path} {im.size} -> {out_path} {cropped.size}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: trim_whitespace.py <in.png> <out.png> [pad_px]")
    pad_px = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    trim(sys.argv[1], sys.argv[2], pad_px)
