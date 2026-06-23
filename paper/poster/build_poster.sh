#!/usr/bin/env bash
# Reproducibly build the ICML 2026 workshop poster + lightning-talk slide.
# Regenerates every asset from source, then renders both PDFs via headless
# Chromium. Fail-fast: any missing input or failed step aborts the build.
#
# Prereq (one time): uv run --with playwright playwright install chromium
#
# Usage:  bash paper/poster/build_poster.sh
set -euo pipefail

cd "$(dirname "$0")"                 # paper/poster/
REPO=../..                          # repo root
FIG="$REPO/figures"
mkdir -p assets

echo "==> [1/7] ERA logo"
unzip -o ERA_Poster_Template.zip ERA_logo.jpg -d assets/ >/dev/null

echo "==> [2/7] Figure 1 (TikZ pipeline) + Table 2 -> trimmed PNGs"
pdflatex -interaction=nonstopmode -halt-on-error fig1_pipeline_standalone.tex >/tmp/fig1_build.log 2>&1
pdftocairo -png -r 300 -singlefile fig1_pipeline_standalone.pdf assets/fig1_pipeline_raw
uv run trim_whitespace.py assets/fig1_pipeline_raw.png assets/fig1_pipeline.png 28
rm -f assets/fig1_pipeline_raw.png
# Table 2 = the full Tevet Diversity-Eval results table (numbers from the same
# generated results/tables/contest_rho_oca.tex the paper uses).
pdflatex -interaction=nonstopmode -halt-on-error table2_standalone.tex >/tmp/t2_build.log 2>&1
pdftocairo -png -r 300 -singlefile table2_standalone.pdf assets/table2_raw
uv run trim_whitespace.py assets/table2_raw.png assets/table2.png 30
rm -f assets/table2_raw.png

echo "==> [3/7] Figures 2 & 18 (PDF -> PNG, 300 dpi)"
pdftocairo -png -r 300 -singlefile "$FIG/rlhf_experiment/ak_curves_overlay_alpacaeval_lm.pdf"        assets/fig2_ak
pdftocairo -png -r 300 -singlefile "$FIG/rlhf_experiment/per_prompt_D_violin_alpacaeval_lm.pdf"      assets/fig2_violin
pdftocairo -png -r 300 -singlefile "$FIG/rlhf_experiment/metric_correlation_scatter_alpacaeval_lm.pdf" assets/fig18_scatter

echo "==> [4/7] Figure 5 (mode-count, already PNG)"
cp "$FIG/mode_count/qwen2.5-3b/ak_curves_overlay.png" assets/fig5_modecount.png

echo "==> [5/7] QR codes (paper, code, LinkedIn, X/Twitter)"
uv run make_qr.py "https://arxiv.org/abs/2606.01811"               assets/qr_paper.png
uv run make_qr.py "https://github.com/AMindToThink/icl-diversity"  assets/qr_code.png
uv run make_qr.py "https://www.linkedin.com/in/matthew-khoriaty"   assets/qr_linkedin.png
uv run make_qr.py "https://x.com/KhoriatyMatthew"                  assets/qr_twitter.png

echo "==> [6/7] (KaTeX is vendored in assets/katex/; no network needed at render)"

echo "==> [7/7] Render poster (A1) + slide (16:9) to PDF"
uv run render_html.py icl_diversity_poster.html icl_diversity_poster.pdf 594mm   841mm
uv run render_html.py icl_diversity_slide.html  icl_diversity_slide.pdf  338.667mm 190.5mm

echo
echo "Done. Outputs:"
echo "  poster -> paper/poster/icl_diversity_poster.pdf   (rename to {ID}_poster.pdf)"
echo "  slide  -> paper/poster/icl_diversity_slide.pdf    (rename to {ID}_slide.pdf)"
pdfinfo icl_diversity_poster.pdf | grep -E "Page size"
pdfinfo icl_diversity_slide.pdf  | grep -E "Page size"
