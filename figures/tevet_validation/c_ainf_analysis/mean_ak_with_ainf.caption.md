## Mean a_k curves (bits/byte) with a_∞ annotation

Source: `scripts/analyze_c_ainf.py`, `plot_mean_ak_with_ainf()`.
Model: Qwen2.5-3B, 50 permutations, completion format, per-byte normalization.
Numeric values (a_∞ annotations) are computed inline by the plotting function via `fit_exponential()` from `scripts/fit_ak_curves.py` — re-run the script to verify.

Three subplots: McDiv_nuggets with_hds × {prompt_gen, resp_gen, story_gen}. Each shows:
- Mean a_k curve ± SEM band for high-diversity (red) and low-diversity (blue) groups
- Exponential fit (dashed lines) extrapolated 3 points beyond k=5
- Horizontal dotted lines at fitted a_∞ for each group, with value annotated

This is the key visual for understanding why C × a_∞ works: the gap between the two horizontal lines (a_∞ for high vs low diversity) IS the diversity signal. The area above the line (E) is small and similar for both groups — there's little learnable structure in either case with only 5 responses.

The k=5 uptick (a_5 > a_4) is visible in some curves — this is a known unexplained anomaly (see docs/tevet_redo_report.md).
