# Pre-Registered Hypotheses: Pairwise Matrix & Token Attribution

## Context

At m=10, Qwen2.5-3B's a_k curve shows steep exponential decay from position 1 rather than the expected flat discovery phase. Prior experiments (02) showed cross-mode surprise reduction exists for at least one pair (lab notebook → philosophy). These experiments systematically measure cross-mode information sharing across all mode pairs and localize where in the target response the model benefits.

## Hypotheses

### H_matrix_1: Diagonal dominance

The diagonal of the pairwise surprise-reduction matrix (same-mode pairs) will show large positive values. Off-diagonal entries will be near zero on average, but with specific pairs showing significant effects.

### H_matrix_2: Informative modes

Some modes will be consistently "informative" (high row mean) — likely modes with unusual structure (e.g., JSON, code, legal text) that calibrate the model's expectations about the range of possible response formats.

### H_matrix_3: Approximate symmetry

The matrix should be approximately symmetric — if mode_i reduces surprise for mode_j by X bits, then mode_j should reduce surprise for mode_i by approximately X bits. Large asymmetries would indicate directional information flow (e.g., unusual modes calibrating expectations for typical modes but not vice versa).

### H_token_1: Front-loaded attribution

For cross-mode pairs with significant surprise reduction, the improvement will be concentrated in the first few tokens of the target response (structural/format learning), not spread evenly throughout the response.

## Date

2026-03-11
