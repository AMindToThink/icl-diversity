## ROC curves: McDiv_nuggets binary classification

Source: `scripts/analyze_c_ainf.py` → `summary_table.txt` (ROC AUC column).
Model: Qwen2.5-3B, 50 permutations, completion format, per-byte normalization.

Six metrics overlaid per subplot. C×a_n (orange dashed) dominates across all 6 McDiv_nuggets subsets.

AUCs for with_hds subsets (from summary_table.txt):

| Task | C×a_n (pb) | C×a_inf_fit (pb) | D_fit (pb) | D_disc (pb) | a_inf_fit (pb) | a_1 (pb) |
|---|---|---|---|---|---|---|
| prompt_gen | 0.894 | 0.529 | 0.642 | 0.244 | 0.492 | 0.463 |
| resp_gen | 0.759 | 0.699 | 0.430 | 0.294 | 0.625 | 0.438 |
| story_gen | 0.733 | 0.684 | 0.320 | 0.226 | 0.523 | 0.277 |

Key observations:
- D_disc is consistently below 0.5 (worse than random) — E_discrete is anti-correlated with diversity.
- D_fit improves over D_disc but is still weak (0.3–0.6 range).
- a_1 (unconditional surprise) is near chance — the signal comes from conditioning on other responses, not from raw surprise.
- C×a_inf_fit is much weaker than C×a_n — the exponential fit degenerates on 5-point curves, making a_n the better estimator of a_∞.
- The no_hds variants (right 3 panels) have ~4× more samples and show the same pattern.
- Story_gen is weakest for all metrics, consistent with the dataset construction confound (Appendix B).
