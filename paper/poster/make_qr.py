# /// script
# requires-python = ">=3.10"
# dependencies = ["qrcode[pil]"]
# ///
"""Generate the poster/slide QR codes from URLs.

Usage: uv run make_qr.py <url> <out.png>
Burgundy modules on white, high error correction (logo-free but robust to
print). Called by build_poster.sh for the paper (arXiv) and code (GitHub) QRs.
"""
import sys

import qrcode
from qrcode.constants import ERROR_CORRECT_H

BURGUNDY = (140, 5, 28)
WHITE = (255, 255, 255)


def make(url: str, out_path: str) -> None:
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_H, box_size=20, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color=BURGUNDY, back_color=WHITE)
    img.save(out_path)
    print(f"QR {url} -> {out_path} {img.size}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_qr.py <url> <out.png>")
    make(sys.argv[1], sys.argv[2])
