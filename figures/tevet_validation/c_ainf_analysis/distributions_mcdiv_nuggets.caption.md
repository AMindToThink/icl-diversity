## Score distributions: McDiv_nuggets high vs low diversity

Source: `scripts/analyze_c_ainf.py`, `plot_distributions()`.
Model: Qwen2.5-3B, 50 permutations, completion format, per-byte normalization.

Layout: 4 rows (metrics) × 6 columns (datasets). Each subplot shows overlaid histograms for high-diversity (red) and low-diversity (blue) groups, with vertical dashed lines at group means.

Rows:
1. C×a_inf_fit (pb) — proposed score with exponential fit
2. D_fit (pb) — existing score C × E_fit
3. a_inf_fit (pb) — raw floor estimate without C weighting
4. D_disc (pb) — original score C × E_discrete

The C×a_inf_fit row shows modest separation (the two distributions overlap substantially but means differ). D_fit shows less separation. D_disc shows inverted separation — the low-diversity group has higher scores, explaining the below-chance AUC.

Note: this plot shows C×a_inf_fit, not C×a_n. The latter (which performs better per the ROC analysis) is not included in the distribution grid. The a_inf_fit degeneracy on 5-point curves (documented in the E_fit outlier caption) means some samples have extreme values that compress the histogram range.
