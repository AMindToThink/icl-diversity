## ROC curves: ConTest binary classification

Source: `scripts/analyze_c_ainf.py` → `summary_table.txt` (ROC AUC column).
Model: Qwen2.5-3B, 50 permutations, completion format, per-byte normalization.

Only 2 subplots: resp_gen and story_gen. Prompt_gen has only one label class (all samples are the same label), so ROC is undefined — the summary_table.txt shows "—" for that dataset.

AUCs (from summary_table.txt):

| Task | C×a_n (pb) | C×a_inf_fit (pb) | D_fit (pb) | D_disc (pb) | a_inf_fit (pb) | a_1 (pb) |
|---|---|---|---|---|---|---|
| resp_gen | 0.707 | 0.667 | 0.469 | 0.339 | 0.632 | 0.503 |
| story_gen | 0.937 | 0.886 | 0.288 | 0.110 | 0.714 | 0.353 |

Story_gen is the standout: C×a_n achieves AUC = 0.937, nearly perfect separation. D_disc is at 0.110 — almost perfectly inverted (anti-classifying). This is the strongest evidence that E-based metrics fail in the low-n regime while a_∞-based metrics succeed.

Note ConTest has fewer samples than McDiv_nuggets (100–160 per task vs 191–790), so confidence intervals are wider.
