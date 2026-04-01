# Qwen3-30B-A3B-Base via Tinker API

Pilot run: ConTest prompt_gen only (200 samples, $0.09).

## Validation results (ConTest prompt_gen with_hds)

C×a_n (per-byte): Spearman ρ = +0.592, ROC AUC = 0.842, OCA = 0.805

Compare to Qwen2.5-3B (local, run_tag qwen25_completion_v3):
C×a_n (per-byte): Spearman ρ = +0.584, ROC AUC = 0.837, OCA = 0.785

Modest improvement from scaling 3B → 30B (MoE).

## TODO

When running the full experiment (all ConTest + McDiv Nuggets sub-experiments),
compare the prompt_gen values to confirm they match this pilot run, then
overwrite this directory with the full results.
