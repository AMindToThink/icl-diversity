# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright"]
# ///
"""Render an HTML poster/slide to PDF via headless Chromium (print path).

Usage: uv run render_html.py <in.html> <out.pdf> <width> <height>
  width/height are CSS sizes, e.g. 594mm 841mm (A1) or 338.7mm 190.5mm (16:9).
Honors the @page size via prefer_css_page_size and forces background colors so
the burgundy panels print. Requires `playwright install chromium` beforehand.
"""
import asyncio
import os
import sys

from playwright.async_api import async_playwright


async def render(html_path: str, pdf_path: str, width: str, height: str) -> None:
    url = "file://" + os.path.abspath(html_path)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        # If the page uses KaTeX, make sure typesetting has finished before we
        # snapshot to PDF. The page's own inline script calls renderMathInElement
        # on DOMContentLoaded; we additionally wait for the produced .katex nodes
        # so the PDF never captures a half-typeset (or raw-LaTeX) frame.
        uses_katex = await page.evaluate(
            "() => typeof window.renderMathInElement === 'function'"
        )
        if uses_katex:
            # Ensure the auto-render pass ran (idempotent if it already did).
            await page.evaluate(
                """() => {
                    if (window.__katexDone) return;
                    renderMathInElement(document.body, {
                        delimiters: [
                            {left: '$$', right: '$$', display: true},
                            {left: '\\\\(', right: '\\\\)', display: false},
                        ],
                        throwOnError: true,
                    });
                    window.__katexDone = true;
                }"""
            )
            # Wait for at least one rendered math node, then let the web fonts
            # settle so glyph metrics are final.
            await page.wait_for_selector(".katex", timeout=15000)
            await page.evaluate("() => document.fonts.ready")
            # Guard: no raw delimiter / control sequence survives OUTSIDE a
            # rendered .katex node. KaTeX keeps the original TeX in a hidden
            # <annotation> MathML node, so we must strip .katex subtrees before
            # scanning, otherwise that legitimate source triggers false leaks.
            leaked = await page.evaluate(
                r"""() => {
                    const clone = document.body.cloneNode(true);
                    clone.querySelectorAll('.katex').forEach(n => n.remove());
                    const t = clone.textContent || "";
                    const bad = ['\\(', '\\)', '\\theta', '\\times', '\\log',
                                 '\\pi', '\\mathrm', '\\rho', '\\ge', '\\mid',
                                 '\\boldsymbol'];
                    return bad.filter(s => t.includes(s));
                }"""
            )
            if leaked:
                raise SystemExit(f"unrendered LaTeX leaked into output: {leaked}")
        await page.emulate_media(media="print")
        await page.pdf(
            path=pdf_path,
            width=width,
            height=height,
            print_background=True,
            prefer_css_page_size=True,
        )
        await browser.close()
    print(f"rendered {html_path} -> {pdf_path} ({width} x {height})")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: render_html.py <in.html> <out.pdf> <width> <height>")
    asyncio.run(render(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
