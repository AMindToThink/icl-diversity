# Changes from ICML Workshop paper to NeurIPS submission

Started: 2026-05-06.
Branch: `anon-submission`.

The NeurIPS variant `paper/main_neurips.tex` reuses the same `paper/sections/*.tex` content files as the ICML workshop wrapper `paper/main_icml_workshop.tex`, so most edits below appear in **both** PDFs. Edits that apply to the NeurIPS wrapper only are flagged with **(neurips wrapper only)**.

## Citation additions (per Q12 audit) and figures-after-checklist fix

**Q12 license-claim audit (2026-05-07).** A subagent verified the Q12 "Licenses for existing assets" answer against the actual paper text. Three real issues were found and fixed:

- **Qwen3-30B-A3B-Base** was used (Appendix~\ref{app:qwen3-comparison}, §Limitations) but uncited. Added `[[cite]] yang2025qwen3` in `refs_ids.toml` (`arxiv:2505.09388`, Qwen3 Technical Report; `skip_authors = ["Qwen Team", "Qwen"]` to keep natbib short-label as "Yang et al."). Cited at first mention in `08_limitations_workshop.tex` line 16 and `appE_qwen3_comparison.tex` line 3.
- **Llama 1B/3B/8B/70B** used in the cross-mode scaling experiment but uncited and missing from Q12. Subagent-verified that `arXiv:2407.21783` (The Llama 3 Herd of Models) covers 8B/70B but **not** 1B/3B (paper line 141: "a herd of three multilingual language models with 8B, 70B, and 405B"). The 1B/3B were released only via the Llama 3.2 Meta blog post; no arXiv technical report exists. Added two entries: `grattafiori2024llama3` (arxiv:2407.21783) and `meta2024llama32` (manual, blog URL). Cited together at first mention in `07_4_cross_mode.tex` line 95.
- **AlpacaEval prompts** in Q12 was the wrong name; the body cites `dubois2024alpacafarm`. Q12 wording rewritten to "AlpacaFarm/AlpacaEval prompts".
- **Hugging Face Transformers and PyTorch** were claimed in the previous Q12 but not cited anywhere in the paper. Removed from Q12 entirely; the paper does not need to separately cite standard ML infrastructure.

Q12 was also rewritten to inline-cite each asset (`\citep{...}`) rather than make a vague "all are cited" claim, and to qualify the license blanket statement: most assets are Apache-2.0 / MIT / research licenses, but Llama 3 and Llama 3.2 are released under the bespoke Meta Llama Community License (more restrictive but permits research use).

**Figures-after-checklist fix.** `paper/main_neurips.tex` had `\newpage \input{checklist.tex}` before the closing `\end{document}`; LaTeX float-queued ~10 appendix figures past the checklist heading because `\newpage` does not flush pending floats. Replaced with `\clearpage`, which forces all pending floats to be placed before the new page. After the fix, all 18 figures appear before "NeurIPS Paper Checklist" in the rendered PDF.

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

## Method Table 1: bolding made strictly predictive

Removed the `\textbf{}` from the Mixed empirical cells. The only bolded entry in the table is now "high" in the Predicted-$D_{Ca_n}$ column (multi-mode coherent), the row the table is designed to predict as the winner. Empirical columns no longer have bolding, sidestepping the stale-prone hand-coded "empirical winner" semantic that previously double-bolded Mixed.

## Reframe: positioning shift from "we propose $D_{Ca_n}$" to "we propose an ICL-based diversity-measurement approach, of which $D_{Ca_n}$ is the working instance"

Author motivation: the metric $D_{Ca_n}$ is one scalar the team found to work; the actual contribution is the ICL-based-diversity approach itself. Repo name `icl-diversity` reflects this. Earlier framing read as advocating $D_{Ca_n}$ as if it were the contribution. Subagent surveyed every section for advocacy-style sentences and recommended a tier-1/tier-2 list.

Phrasing pattern used (per author's preference): "a new approach to measuring diversity using in-context learning, of which Decan ($D_{Ca_n} = C \times a_n$) is the working instance we evaluate." The author rejected "family of metrics" because only one instance has been validated.

**Tier 1 (sets the paper's framing)**:
- `abstract_workshop.tex` line 2: full reframe of the "We propose..." sentence.
- `conclusion_workshop.tex` lines 3-4: mirror reframe of the "is an information-theoretic diversity metric" sentence; "It requires no..." became "The approach requires no...".
- `01_motivation_workshop.tex` line 9: reframe of "We propose a metric, Decan or...".
- `01_motivation_workshop.tex` line 11: "The metric uses only..." → "The approach uses only...".
- `01_motivation_workshop.tex` line 17: "The product...is the score" → "The product...is the working scalar we adopt".

**Tier 2 (light "metric" → "approach/framework" softening)**:
- `07_5_tevet_workshop.tex` line 23: "an information-theoretic, embedding-free metric reaches..." → "an ICL-based diversity metric reaches...".
- `07_6_rlhf_workshop.tex` line 67: "an information-theoretic metric that needs neither..." → "an ICL-based diversity metric that needs neither...".
- `10_related.tex` line 4: "Our metric operates at the distributional level..." → "Our approach operates...".
- `10_related.tex` line 20: "Our metric leverages a related insight..." → "Our framework leverages...".

Specific numerical-result sentences (e.g., "$D_{Ca_n}$ reaches OCA 0.846") and per-experiment instance-specific claims were left as-is. The §07_8 "framework admits other metrics" appendix paragraph (already framed at framework level) was left untouched. All edits are in shared sections; both wrappers see them.

Net body-content cost of all nine reframes: ~6 PDF lines. Body still fits in 9 pages with ~25 lines of headroom on page 9.

## §5 OLMo discussion: dropped "deferred to a longer venue" sentence

The Discussion paragraph at the end of §5 ended with "A cross-model comparison on the same NoveltyBench-curated prompts is deferred to a longer venue." The author flagged that this future-work promise is unreliable (work on the paper is contingent on reviewer feedback), so the sentence has been removed. The TeX comment block at lines 69-79 of `07_6_rlhf_workshop.tex` is left intact as internal documentation that the cross-model macros exist; it does not render.

## §4 Tevet headline: cherry-pick caveat for McDiv prompt_gen

The Headline-result paragraph (`07_5_tevet_workshop.tex`) cited McDiv prompt_gen as the headline number without flagging that this is the row where $D_{Ca_n}$ does best of the nine binary tasks. Inserted "where $D_{Ca_n}$ performs best" inline so the framing matches the abstract (`abstract_workshop.tex` already says "the McDiv prompt\_gen set where it performs best"). Smallest faithful edit; no other claim shifts.

## §4 Tevet setup: forward pointer to the completion-format definition

The Tevet setup paragraph (`07_5_tevet_workshop.tex`) used the term "completion format" without defining it; the definition lives in the Practical Considerations appendix (`06_practical.tex`, subsection "Formatting the Conditioning Context"). Two edits:
1. Added `\label{sec:formatting}` to the appendix subsection so it can be referenced.
2. Inserted a brief parenthetical "(Appendix~\ref{sec:formatting})" after the first use of "completion format" in §4.

Shared-section edits; both wrappers see them.

## Method Table 1 caption: added one-sentence pointer to the Qwen incoherent-vs-coherent flip

Appended one short sentence to the end of the existing Table 1 caption: "On Qwen2.5-3B, multi-mode incoherent also outranks multi-mode coherent; Section~\ref{sec:scalar} discusses." This is just a forward pointer to the §3.3 paragraph (also added this session) where the finding is discussed; the caption itself does not now carry the claim. Shared-section edit; both wrappers see it.

## Method §3.3 (`03_method_workshop.tex`): added Qwen-vs-GPT-2 ICL-power example after Table 1

Added a 3-sentence paragraph after the line "$\theta$'s in-context learning capability is the lens through which diversity is measured", concretising the abstract claim that stronger base models tighten the metric. Specifically: notes that Qwen2.5-3B scores multi-mode incoherent above multi-mode coherent (reversing the predicted ranking, with all four numbers macro-imported), points to Figure~\ref{fig:scenario-curves} where Qwen's $\bar{a}_k$ curve drops over $k$ on the multi-mode incoherent scenario as the mechanism (Qwen's ICL recognises the shared template structure as a learnable pattern despite within-response scrambling), and observes that GPT-2's weaker ICL fits the predicted ordering on this row. Closes with the recommendation to use stronger base models when possible. Shared-section edit; both wrappers see it.

## Appendix Table 5 (`tab:scenarios`): tightened column padding to fix overfull right margin

The 15-column scenario-validation table (`07_2_scenario_validation.tex`) was overfull by 88.4pt (~1.22 inches, ~22% of the 5.5-inch text column) at the default `\tabcolsep=6pt` even with `\small`. The wrap was a `table*` so the table did not run off the page, but it extended noticeably into the right margin. Reduced `\tabcolsep` to `2pt` only inside this `table*` block (with a paired `\setlength{\tabcolsep}{6pt}` after `\input` to restore the article default for any subsequent tables). The math: 15 columns × 2 padding sides × 4pt savings = 120pt, which clears the 88.4pt overflow with ~32pt headroom. Single Overfull warning at line 812 of `main_neurips.log` is gone after the fix; no other layout side-effects observed. Pure typography fix; no content change.

## Em-dash removed from `07_6_rlhf_workshop.tex` footnote (per "no em-dashes in public writing")

The footnote about UTF-8 byte truncation contained one Unicode em-dash separating two ideas about tokenizer dependence. Replaced with a sentence break (`. That variation is harmless...`). No claim shift.

## Method §3.2 (`03_method_workshop.tex`): softened motivation for the geometric form

The line that motivated the geometric mean ("Geometric averaging (rather than arithmetic) is what gives $C$ the desired suppression: a single fluent response cannot rescue a set in which the rest are incoherent...") had two over-claims that the empirical Mixed scenario in Table 1 contradicts: "desired suppression" (the Mixed row scores highest of all five, so the suppression is empirically incomplete) and "cannot rescue" (it can; we observe it).

Rewritten as: "Perplexity is a standard metric of incoherence, and the geometric form is intended to suppress sets containing incoherent responses: a single sample with high per-byte cross-entropy drives $C$ toward zero, limiting the rescue effect of any single fluent response on an otherwise incoherent set."

Changes:
- Lead motivation switched from "geometric averaging gives C the desired suppression" to "perplexity is a standard metric of incoherence" — appeals to a known concept rather than a property of our specific formula.
- "Desired" → "intended to" — hedges the empirical claim.
- "Cannot rescue" → "limiting the rescue effect" — honest about empirical incompleteness while still describing what the formula mathematically does.

This edit is in a shared section file; both wrappers see it.

## Setup section: corrected overclaim about cross-entropy and bits/byte rationale

In `02_setup.tex`:

- Line 19 used to call the total cross-entropy "a property of the string itself, independent of $\theta$'s tokenizer." The first half overstates: the total cross-entropy depends on the string *and* on $\theta$'s distribution. Rewritten as "a function of the string and of $\theta$'s distribution but not of $\theta$'s tokenizer (since the chain rule yields the same total regardless of how the sequence is factored)." Tokenizer-independence claim retained because it remains correct.
- Line 28 used to motivate the per-byte rate as "makes this rate independent of $\theta$'s tokenizer, enabling comparisons across base models with different vocabularies." That framing was misleading: total bits (Eq.~\ref{eq:total-xent}) is *already* tokenizer-independent, so the per-byte vs total-bits choice is not driven by tokenizer-independence. The actual reason we use bits/byte is empirical, not theoretical, and we have not investigated why. Rewritten to say so plainly: "We adopt this per-byte rate because it works better than total bits in our experiments; we have not investigated why. Normalising by byte count rather than token count keeps the rate independent of $\theta$'s tokenizer when comparing base models with different vocabularies." (The second sentence answers a separate, smaller question — why use bytes rather than tokens as the denominator — for which tokenizer-independence is the correct rationale.)

The same overclaim survives at `03_method_workshop.tex` line 14: "We normalize each $a_k$ by the byte count $\|r_k\|$ to get a per-byte (bits/byte) curve that is independent of $\theta$'s tokenizer." Like line 28 of setup, this frames bits/byte as the path to tokenizer-independence; bits would already give that. Not changed in this commit because the user's directive was scoped to the setup section. Suggested replacement, if you want it: "We normalise each $a_k$ by the byte count $\|r_k\|$ for an empirically better-behaved rate (bits/byte); see Section~\ref{sec:setup-notation}."

## Appendix A "Diversity evaluation benchmarks" (`10_related.tex`): McDiv-validation sentence repointed to body section first, and retracted "label contamination" claim removed

The "Diversity evaluation benchmarks" paragraph previously read: `We use McDiv for validation (Appendix~\ref{app:confound}), though we identify data quality issues including label contamination and a construction confound.` Two problems:

1. **Wrong target.** The parenthetical pointed only to the confound appendix (Appendix~F in NeurIPS, `appC_mcdiv_confounds.tex`), not to where the validation results actually live (Section~\ref{sec:tevet} = `07_5_tevet_workshop.tex`). The reader was being sent to the caveat before locating the validation itself.
2. **Retracted claim.** The "label contamination" half of "data quality issues including label contamination and a construction confound" was no longer supported. An earlier session interpreted McDiv\_nuggets rows that share a context but have disjoint response sets as duplicate-label contamination, then later determined those were the surviving fragment of Tevet's ConTest pair structure (high-div / low-div response sets keyed to the same prompt), not contamination. The corresponding `--dedup` flag and "Label Contamination" appendix subsection were removed at that point; this sentence in Related Work was the last live reference to the retracted claim. (See project memory `project_tevet_dedup.md` and `investigations/tevet_overlap_followup.md` for the trail.)

Rewritten as: `We use McDiv for validation (Section~\ref{sec:tevet}), though we identify a construction confound in how its low-diversity sets are produced (Appendix~\ref{app:confound}).` Body section is named first; the appendix is reached as the follow-up; the retracted contamination claim and the now-singular "data quality issues" plural are gone. Both labels resolve in both wrappers (in the workshop, `sec:tevet` is also a Section in the body, just before the appendices), so the prose is correct in both PDFs. Shared-section edit; both wrappers see it.

## Appendix B.3 "Dependence on Sample Ordering" (`06_practical.tex`): broadened the source-of-jaggedness explanation

The opening of Appendix B.3 attributed the jaggedness of a single ordering's $a_k$ curve solely to length differences across responses: `Since raw $a_k$ is in total bits, responses of different lengths produce different values even at the same per-byte rate, making a single ordering's curve jagged. Averaging over random permutations removes this length effect: each position averages over all responses...`. This is too narrow. Even at matched lengths the curve is jagged, because individual responses differ in how surprising they are to $\theta$ given whichever responses happen to precede them; the length story is one special case of that more general dependence (and it does not apply to the per-byte curve, where length is divided out).

Rewritten as: `Individual responses differ in how surprising they are to $\theta$ given the responses that precede them, so $a_k$ depends on which response sits at position $k$ and which others precede it, making a single ordering's curve jagged. Averaging over random permutations removes this dependence: each position averages over many choices of response and preceding context, so the curve reflects only how $\theta$'s predictions improve with more context.` Single mechanism; no claim added; no claim retracted; just broadened from the length-only framing. Shared-section edit; both wrappers see it.

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
