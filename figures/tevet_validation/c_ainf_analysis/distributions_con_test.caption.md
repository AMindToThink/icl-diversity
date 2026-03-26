## Score distributions: ConTest high vs low diversity

Source: `scripts/analyze_c_ainf.py`, `plot_distributions()`.
Model: Qwen2.5-3B, 50 permutations, completion format, per-byte normalization.

Layout: 4 rows (metrics) × 2 columns (resp_gen, story_gen). Prompt_gen is absent because it has only one label class (all samples same label).

Same metric rows as the McDiv_nuggets distribution plot. Story_gen shows the clearest visual separation, consistent with its high AUC (0.937 for C×a_n).

Note: smaller sample sizes than McDiv_nuggets (134 resp_gen, 160 story_gen), so histograms are noisier.
