# Changes from ICML Workshop paper to NeurIPS submission

Started: 2026-05-06.
Branch: `anon-submission`.

The NeurIPS variant `paper/main_neurips.tex` reuses the same `paper/sections/*.tex` content files as the ICML workshop wrapper `paper/main_icml_workshop.tex`, so most edits below appear in **both** PDFs. Edits that apply to the NeurIPS wrapper only are flagged with **(neurips wrapper only)**.

## Wrapper / style

- **(neurips wrapper only)** Created `paper/main_neurips.tex`. Single-column NeurIPS layout via `\usepackage{neurips_2026}` (no track option = anonymous double-blind with line numbers).
- **(neurips wrapper only)** Bibliography style switched from `icml2026` to `plainnat`.
- **(neurips wrapper only)** Style files added to `paper/`: `neurips_2026.sty`, `checklist.tex`, plus dependencies missing from TinyTeX (`environ.sty`, `lineno.sty`, `trimspaces.sty`).
- **(neurips wrapper only)** Mandatory NeurIPS Paper Checklist appendix (`paper/checklist.tex`) created with all 16 questions answered.

## Section moves (apply to both wrappers since both `\input{}` the same files; the ICML wrapper still includes them in their original positions, so the move is technically only realised in `main_neurips.tex`'s `\input` order).

- **(neurips wrapper only)** `\input{sections/10_related}` moved from main body to first appendix entry (saves ~½ page in the 9-page main-text budget). The Motivation paragraph already cites the key prior work (`du2019boosting`, `li2016diversity`, `zhang2018aim`) so no new prose was added in main body. The forward reference text changed from `Section~\ref{sec:related}` to `Appendix~\ref{sec:related}` in `01_motivation_workshop.tex` and `checklist.tex`.
- **(neurips wrapper only)** Figure 2 (`fig:rlhf-ak-violin`, the AlpacaEval $\bar a_k$ overlay + per-prompt $D$ violin) moved from `07_6_rlhf_workshop.tex` body to `appF_rlhf_cross_metric.tex` appendix. Body sentence in `07_6_rlhf_workshop.tex` rewritten to "underlying $\bar a_k$ curves and per-prompt $D_{Ca_n}$ distributions are visualised in Appendix~\ref{app:rlhf-cross-metric}". Rationale: same data is already in Table 4 (`tab:rlhf-diversity`).

## Table merge

- **`03_method_workshop.tex` Table 1 (`tab:edge-cases`)** — combined the qualitative edge-case-prediction table with the empirical $D_{Ca_n}$ columns from the appendix scenario-validation table. New columns: `GPT-2 (124M)` and `Qwen2.5-3B base` empirical $D_{Ca_n}$ for each of the 5 scenarios. Caption rewritten to flag that Mixed empirically scores highest (the predicted "mid" position is wrong), pointing readers to Appendix Table 5 for the full per-metric breakdown.
- The appendix scenario-validation table (`tab:scenarios` in `07_2_scenario_validation.tex`) is **unchanged** — it still carries the full 7-metric × 2-model breakdown plus $\sigma_\ell$.
- Both captions had a "practitioners can reweight (e.g., $C^\alpha \times a_n$)" sentence that asserted untested behaviour; that sentence was removed and replaced with a cross-reference to Section~\ref{sec:exp-discussion}, where a single, honestly-hedged paragraph was added that says reweighting is one untested family of variants.

## Limitations rewrites

- "Cross-mode learning is model-dependent" merged into "Measurement is relative to $\theta$'s perception" (was a separate `\paragraph{}`; is now the second sentence of the first limitation, with the same Section~\ref{sec:cross-mode-learning} cross-reference).
- "Prompt format is fixed" paragraph dropped entirely. The point is a minor caveat; we no longer flag it in §Limitations. The prompt-engineering discussion in `07_8_discussion.tex` is unchanged.
- "Length-matching drops short-response prompts" paragraph compressed from ~7 sentences to a single sentence in §Limitations, with the full mechanics moved to `07_8_discussion.tex` under the new `\paragraph{Bits/byte vs.\ total bits}` heading. Rationale: this is methodological detail about the bits/byte normalisation choice (which is itself one of the alternative-metric design choices), so it now lives next to "The framework admits other metrics."

## New / expanded discussion paragraph (appendix)

- Added `\paragraph{Bits/byte vs.\ total bits, and the length-matching consequence.}` at the end of `07_8_discussion.tex` covering: why we default to bits/byte, why per-byte is not length-invariant in causal LMs, the OLMo-2-7B length-matching protocol, the 111/300 dropped-prompt count, and the un-truncated robustness check ($p<10^{-13}$ on both prompt sets).
- Added one sentence in the existing "framework admits other metrics" paragraph noting that coherence-reweighting variants ($C^\alpha \times a_n$) are untested and we make no empirical claim about them.

## Edits applicable to the workshop paper too (suggested, not applied)

The user can decide whether to port these back to `main_icml_workshop.pdf`:

1. **Remove the "practitioners can reweight" claims from Table~\ref{tab:edge-cases} and Table~\ref{tab:scenarios} captions.** Both captions made an empirical claim about $C^\alpha \times a_n$ that we did not test. Replacing with a cross-reference is more honest.
2. **Merge "Cross-mode learning is model-dependent" into "Measurement is relative to $\theta$'s perception".** Saves a paragraph header and reads more naturally.
3. **Move the long "Length-matching drops short-response prompts" detail out of §Limitations** into the discussion section (where the bits/byte normalisation choice is debated). Limitations gets a 1-sentence flag instead.
4. **Add the "Bits/byte vs.\ total bits" paragraph** to the discussion section. This articulates a real design choice that wasn't previously surfaced.

Items 2-4 are content reorganisations that would shorten §Limitations in the ICML version too; if the workshop is also tight on pages this is useful even outside the NeurIPS context. The ICML workshop is two-column, so the page budget arithmetic is different; verify space saving before committing.

## Heads-up / latent issues for a future session

These are real problems left behind by this NeurIPS pass. None block submission, but each is worth fixing in a calmer session.

### 1. Cross-reference text in `01_motivation_workshop.tex` line 7

The motivation paragraph was changed from `(Section~\ref{sec:related})` to `(Appendix~\ref{sec:related})`. **This is correct in `main_neurips.tex` (where Related Work is in the appendix) but wrong in `main_icml_workshop.tex` (where Related Work is still in the main body).** The label resolves either way, only the noun phrase is wrong. Two clean fixes:
   - **(preferred)** Move `\input{sections/10_related}` in `main_icml_workshop.tex` into its appendix block too. Then both wrappers agree and the prose is correct in both. The Motivation paragraph already does enough related-work duty in main body for the workshop too.
   - **(alternative)** Use a wrapper-defined macro `\relatedRef` that the NeurIPS wrapper sets to `Appendix` and the ICML wrapper sets to `Section`, then write `(\relatedRef~\ref{sec:related})` in `01_motivation_workshop.tex`. More mechanism than the move warrants.

### 2. Wrapper-only edits that affect both PDFs (from shared `sections/`)

Because both wrappers `\input` the same `sections/*.tex` files, every "shared" edit listed above is **already live in both builds**. Specifically the workshop PDF (when next rebuilt) will show:
   - The new 6-column Table 1 (with empirical $D_{Ca_n}$ columns) instead of the original 4-column qualitative one.
   - The "Practitioners can reweight" sentences gone from both Table 1 and Table 5 captions, replaced with a Section~\ref{sec:exp-discussion} pointer.
   - The merged "Cross-mode" + "Measurement" Limitations paragraph and the dropped "Prompt format is fixed" paragraph.
   - The compressed Length-matching paragraph in Limitations + the new bits/byte paragraph in `07_8_discussion.tex`.
   - **Figure 2 (`fig:rlhf-ak-violin`) gone from `07_6_rlhf_workshop.tex`.** The ICML workshop wrapper that previously displayed this figure in main body will now find it in the appendix instead. Verify this is intended for the workshop too. If not, the figure block needs to be either (a) restored in the section file (then conditionally hidden in the NeurIPS wrapper) or (b) duplicated in `appF_rlhf_cross_metric.tex` for NeurIPS only.

### 3. ICML workshop two-column layout may break the new Table 1

The ICML two-column body width is ~3.25" per column. The new 6-column Table 1 may not fit a single column at `\small`. If the workshop wrapper uses `\begin{table*}` (spans both columns) it should fit; if it uses `\begin{table}`, the wider table could overflow. Verify on next workshop rebuild. The current source uses `\begin{table}[t]`, which means it's column-width — likely fine because the qualitative columns are short, but worth a visual check.

### 4. Inline numbers in the new Table 1 are imported via macros (now correct)

The empirical D_Can values for the 5 scenarios × 2 models in Table 1 were initially hand-typed. They are now defined as `\scenario{Scenario}{Model}DCan` macros in `results/tables/paper_macros.tex`, generated by `scripts/build_paper_macros.py::scenario_validation_macros()`. The function reads `results/scenario_metrics_v3_{gpt2,qwen3b}_100perm.json` and computes `D_Can = mean(C × a_n_per_byte)` per scenario over the 5 prompts, matching `scripts/generate_scenario_table.py` exactly (it does NOT use the JSON's `diversity_score_D` field, which is the C × E variant per the deprecated naming). Future change: if the scenarios JSONs are regenerated, re-run `uv run python scripts/build_paper_macros.py` before rebuilding the paper.

### 5. The macro-coverage unit test does not exercise the new macros

`tests/test_paper_macros.py` checks (a) the build script runs and (b) every macro the paper references resolves. Adding the 10 new `scenario*DCan` macros is fine for (a) and (b) since the paper now uses them, but there is currently no negative regression test asserting that the literals (e.g. `0.52` for Mixed/GPT) are gone from `03_method_workshop.tex` prose. If the paper later reverts the macro reference back to a literal, no test will catch it. A future improvement: add such forbidden-substring assertions to `test_paper_macros.py` for these specific values.

## Edits NOT applied to either paper

- `\input{sections/impact_workshop}` is **still included in both wrappers** in its original position (right before the bibliography). The earlier exploratory edit that folded it into the Conclusion was reverted at user request.
- The §Conclusion text was reverted to its original form.
- Numerical results, model choices, and citations are unchanged.
