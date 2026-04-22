# Citation Claim Verification Report

**Date:** 2026-04-22
**Paper:** `paper/in_context_diversity_metric.tex`
**Previous audit:** `paper/citation_verification_report.md` (2026-04-21, metadata-focused)
**This audit:** claim-level — does each cited paper actually *say what we attribute to it*?
**Method:** For each of the 12 `\cite{}` entries in `refs_ids.toml`, the prose context was grepped from the .tex, then the actual cited paper was fetched (arXiv HTML, ar5iv, ACL Anthology, or direct web search). Each claim was cross-checked against a specific section / table / equation of the source.

Since the `bibliography-from-ids` run on 2026-04-21 already cleaned up metadata (author lists, arXiv IDs), this pass focused purely on whether the prose near each citation is supported by the referenced paper.

---

## Summary Table

| Bibkey                          | Status               | Severity | Issue                                                                                                          |
| ------------------------------- | -------------------- | -------- | -------------------------------------------------------------------------------------------------------------- |
| `lai2024llmmeans`               | PARTIALLY SUPPORTED  | LOW      | Paper is a clustering/interpretability method, not self-described as a diversity metric. Framing slightly overstates scope. |
| `gao2020pile`                   | VERIFIED             | —        | BPB values 0.7177 (GPT-3 davinci) and 1.0468 (GPT-2 XL) match Table 2 exactly; citing paper rounds to 0.72/1.05. |
| `radford2019language`           | VERIFIED (w/ nuance) | LOW      | Paper Table 2 says 117M; we use 124M from HF checkpoint. Context length 1024 confirmed (§2.2). |
| `yang2024qwen25`                | VERIFIED             | —        | Qwen2.5-32B = 128K context, Qwen2.5-3B = 32K; both match paper Table 1. |
| `tevet2020evaluating`           | PARTIALLY SUPPORTED  | MEDIUM   | decTest/conTest definitions and McDiv_nuggets construction verified. BUT conTest size 670 in prose disagrees with paper's stated "200 sets" (100 per class); McDiv ≈6K and McDiv_nuggets ≈3K verified. |
| `crutchfield2003regularities`   | VERIFIED             | LOW      | Excess entropy definition supported. Journal citation is Chaos **vol. 13**, pp. 25–54 (2003) — confirm bib does not record vol. 15. |
| `holtzman2020curious`           | PARTIALLY SUPPORTED  | LOW      | Self-BLEU verified (§5.2); Distinct-n is NOT used by the paper. Claim cites "n-gram overlap, self-BLEU" — self-BLEU is n-gram, so defensible, but Distinct-n attribution would be wrong. |
| `guo2024linguistic`             | VERIFIED             | —        | Form/content taxonomy adoption confirmed (§3); 5 NLG tasks confirmed (Table 1/§4.1). |
| `zhang2025commonsense`          | VERIFIED             | —        | GPT-4o annotator explicitly named in §4.1.1; meta-evaluation of diversity metrics confirmed. |
| `zhang2025noveltybench`         | VERIFIED (w/ nuance) | LOW      | DeBERTa classifier, functional equivalence, Distinct_k all verified in §3.2. 79% vs 71% internal inconsistency between §3.2 and Appendix A.3 confirmed in cited paper. |
| `qiu2026selfimprovement`        | VERIFIED             | —        | Theorem 4.2 (Gibbs → softmax over coherence), Theorem 5.5 (description-length regularization optimality) and Appendix-C reductions (debate, bootstrap, ICM) all confirmed. Abstract phrasing matches. |
| `wen2025unsupervised`           | PARTIALLY SUPPORTED  | LOW      | ICM + Algorithm 1 in §2.3 confirmed. "Latent knowledge" phrasing is NOT the paper's own wording (paper uses "capabilities / concepts / skills"); "latent knowledge" only appears once, in a Burns 2022 reference. Paraphrase is defensible but not literal. |

**Breakdown:** 7 VERIFIED, 5 PARTIALLY SUPPORTED, 0 UNSUPPORTED, 0 UNABLE TO VERIFY.

**No HIGH-severity findings.** The 2026-04-21 audit's fabricated-authors problems have been resolved — the `refs_ids.toml` pipeline now has the correct identifiers, so every citation in this pass resolves to a real paper that is, at worst, *slightly over-framed* in our prose.

---

## Detailed Findings

### 1. `lai2024llmmeans` — §1 (Motivation) and §11 (Embedding-based diversity)

**Prose (line 37):** "Existing approaches typically measure diversity via embedding-space clustering \citep{lai2024llmmeans} or surface-level $n$-gram statistics."

**Prose (line 608):** "Prior work uses embedding distances or clustering (e.g., $k$-LLMmeans \citep{lai2024llmmeans}) to measure diversity."

**Cited paper:** Diaz-Rodriguez (2025), "Summaries as Centroids for Interpretable and Scalable Text Clustering," arXiv:2502.09667.

**Verification:**
- Sole authorship confirmed (matches `skip_authors` pattern).
- Embeddings + k-means + LLM-summary centroids confirmed (Section 2 / Algorithm 1, "Preliminaries: k-Means for text clustering").
- **The paper is framed as clustering + interpretability, not diversity measurement.** Our prose pigeonholes it as a "diversity measure via embedding-space clustering," which is overreach. k-LLMmeans never positions itself as a diversity metric.

**Severity: LOW** (the prose is loose but not factually false — k-means clustering *can* be used for diversity; the paper just isn't itself the diversity method). Consider softening to "embedding-space clustering approaches (e.g., k-LLMmeans is an interpretable centroid-based variant)" or swap to a paper that genuinely proposes an embedding-based diversity *metric*.

**Precise reference:** Diaz-Rodriguez (2025), §2 or Algorithm 1.

---

### 2. `gao2020pile` — §5.3 (Typical range of C)

**Prose (line 252):** "Published bits-per-byte values for English text range from $\sim$0.72 (GPT-3 davinci on The Pile \citep{gao2020pile}) to $\sim$1.05 (GPT-2 XL), corresponding to $C \in [0.49, 0.61]$."

**Cited paper:** Gao et al. (2020), "The Pile," arXiv:2101.00027.

**Verification:**
- Table 2 reports BPB on overall Pile: GPT-3 davinci = **0.7177**, GPT-2 XL = **1.0468**.
- Rounded to 2 d.p.: 0.72 and 1.05 ✓.

**Severity: none.** Fully VERIFIED.

**Precise reference:** Gao et al. (2020), Table 2, overall Pile row.

---

### 3. `radford2019language` — §8.1 (Experimental Setup)

**Prose (line 342):** "GPT-2 (124M parameters, 1024-token context window) \citep{radford2019language}."

**Cited paper:** Radford et al. (2019), "Language Models are Unsupervised Multitask Learners."

**Verification:**
- Table 2 of the paper lists smallest model at **117M** parameters, not 124M. 124M is the HuggingFace `gpt2` checkpoint count (includes embedding params).
- 1024-token context length confirmed (Section 2.2 "Input Representation").

**Severity: LOW.** Factually there is a 7M-parameter gap between the paper's stated number and the HF checkpoint we actually use. TOML's `claim = …` field already documents the discrepancy. The prose could add a one-word footnote, but this is a well-known convention.

**Precise reference:** Radford et al. (2019), Table 2 and §2.2.

---

### 4. `yang2024qwen25` — §8.1 (Experimental Setup)

**Prose (line 342):** "Qwen2.5-32B (32B parameters, 128K-token context window) \citep{yang2024qwen25}."

**Cited paper:** Yang et al. (2024), "Qwen2.5 Technical Report," arXiv:2412.15115.

**Verification:**
- Table 1 confirmed: Qwen2.5-32B = **128K** context; Qwen2.5-3B = **32K** context.
- The 2026-04-21 audit's 32K-vs-128K fix has been applied correctly in current prose.

**Severity: none.** Fully VERIFIED.

**Precise reference:** Yang et al. (2024), Table 1.

---

### 5. `tevet2020evaluating` — §8.5 (External Validation) and App. C (Confound)

**Prose (line 509):** "We evaluate against Tevet and Berant's diversity-eval benchmark \citep{tevet2020evaluating}, which comprises two diagnostic tests (decTest, conTest) and the McDiv dataset (including the McDiv\_nuggets subset), totalling roughly 6K McDiv sets, 3K McDiv\_nuggets sets, 3.6K decTest sets, and 670 conTest sets…"

**Prose (line 910):** "Workers first write five *different* continuations (the high-diversity set). The same worker is then asked to *self-select one* of their own five responses and paraphrase it five times…"

**Cited paper:** Tevet & Berant (EACL 2021; arXiv:2004.02990).

**Verification:**
- decTest / conTest definitions confirmed (§3.1, §3.2).
- McDiv ≈ 6K sets: "6K {c,𝒮c} pairs (2K for each storyGen, respGen, promptGen)" ✓.
- McDiv_nuggets ≈ 3K sets ✓.
- decTest ≈ 3.6K: the paper reports **1,000 contexts × ~10 temperatures = 1,000 sets**, then 200 used for human eval. Our "3.6K" does not obviously map to the paper's own numbers — this was also flagged in the 2026-04-21 audit, which found ~13,929 in released CSVs but couldn't reconcile it with the paper. The 3.6K is probably the total CSV row count for decTest after grouping; it should be traceable to a script output.
- conTest ≈ 670: paper explicitly says "200 sets of 5 responses each per task (100 sets per class)" — **so paper-stated count is 200, not 670**. Our 670 likely comes from the released CSV (multiple tasks × per-set × …); but the prose attributes it to Tevet & Berant §6.4 directly, which is inaccurate.
- McDiv_nuggets self-selection: paper says "we asked the same workers to choose a **single response they wrote**, and rephrase it 5 times such that the original content will be preserved, while changing the form." — **exactly** matches our Appendix C prose. ✓

**Severity: MEDIUM.** The "3.6K / 670" numbers are attributed to Tevet & Berant §6.3–6.4 but don't match the paper's round numbers. They are likely CSV-derived figures that should either (a) be rederived by a script with a cited source, or (b) be replaced with Tevet's original round numbers ("≈1K decTest contexts, 200 conTest sets").

**Precise reference:** Tevet & Berant (2021), §3.1, §3.2, §6.3, §6.4.

---

### 6. `crutchfield2003regularities` — §11 and App. B

**Prose (line 602):** "The $a_k$ curve also admits an excess-entropy summary related to the computational mechanics literature \citep{crutchfield2003regularities}; see Appendix~\ref{app:excess-entropy}."

**Prose (line 724):** "(connecting to the excess entropy of computational mechanics \citep{crutchfield2003regularities}…)"

**Cited paper:** Crutchfield & Feldman (2003), "Regularities Unseen, Randomness Observed: Levels of Entropy Convergence," Chaos **vol. 13, pp. 25–54**.

**Verification:**
- Title, authors, arXiv (cond-mat/0102181) all confirm.
- Journal is **Chaos 13**(1), pp. 25–54, 2003. The 2026-04-21 audit report discussed "vol. 15" in several places — **this appears to have been a typo in that report; the actual volume is 13**. Verify `refs.bib` reflects volume 13.
- Excess entropy definition: paper's §III.B Eq. 48 gives the standard form E = Σ_L [h_μ(L) − h_μ]. Units: bits. Matches our analogy.

**Severity: LOW.** The analogy is defensible. Just make sure bib has Chaos **13**, not 15.

**Precise reference:** Crutchfield & Feldman (2003), Chaos 13:25–54, §III.B Eq. 48.

---

### 7. `holtzman2020curious` — §11 (Sampling diversity metrics)

**Prose (line 610):** "Work on decoding-time diversity \citep{holtzman2020curious} typically operates at the surface level ($n$-gram overlap, self-BLEU)."

**Cited paper:** Holtzman et al. (2020), "The Curious Case of Neural Text Degeneration," ICLR 2020, arXiv:1904.09751.

**Verification:**
- Self-BLEU is used in **§5.2** ("Distributional Statistical Evaluation") ✓.
- The paper **does NOT use Distinct-n** as a metric (it uses Zipf coefficient, Self-BLEU, HUSE, repetition). Our prose says "$n$-gram overlap, self-BLEU" — self-BLEU *is* an n-gram-overlap metric, so technically this is fine. But if the implicit referent for "n-gram overlap" is Distinct-n, that would be wrong for this citation.
- Paper introduces nucleus/top-p sampling ✓.
- ICLR 2020 venue ✓.

**Severity: LOW.** Claim holds because self-BLEU *is* an n-gram-overlap metric; but if any nearby sentence (in this or future revisions) attributes Distinct-n to this paper, that would be unsupported.

**Precise reference:** Holtzman et al. (2020), §5.2.

---

### 8. `guo2024linguistic` — §11 (Diversity evaluation benchmarks)

**Prose (line 618):** "\citet{guo2024linguistic} benchmark linguistic diversity of LLMs, extending Tevet and Berant's framework to a broader set of generation tasks."

**Cited paper:** Guo, Shang, Clavel (2024), "Benchmarking Linguistic Diversity of Large Language Models," arXiv:2412.10271.

**Verification:**
- §3 states: "According to Tevet and Berant (2021), diversity can be divided into two primary dimensions: form diversity and content diversity…" — clearly adopts their taxonomy ✓.
- 5 NLG tasks confirmed (Table 1, §4.1): Language Modeling, Machine Translation, Summarization, Next Utterance Generation, Automatic Story Generation.
- BUT §3 also says: "We build on the linguistic diversity evaluation framework and preprocessing methods of **Guo et al. (2024b)**" — so the primary lineage is Guo 2024b, with Tevet & Berant supplying the form/content taxonomy.

Our characterization "extending Tevet and Berant's framework" is defensible (the taxonomy *is* Tevet's) but slightly overstates the lineage. A more precise phrasing would be "adopting Tevet and Berant's form/content taxonomy across a broader set of generation tasks, building on Guo et al. 2024b."

**Severity: LOW.** Acceptable paraphrase.

**Precise reference:** Guo et al. (2024), §3 and Table 1 / §4.1.

---

### 9. `zhang2025commonsense` — §11 (Diversity evaluation benchmarks)

**Prose (line 617):** "\citet{zhang2025commonsense} conduct a meta-evaluation of diversity metrics for constrained commonsense generation, using GPT-4o as an annotator in place of crowd workers."

**Cited paper:** Zhang, Peng, Bollegala (2025; ACL 2025 long paper), arXiv:2506.00514.

**Verification:**
- "meta-evaluation of diversity metrics" confirmed (Abstract, §4).
- Commonsense generation / CommonGen setting confirmed.
- **GPT-4o as annotator explicitly named in §4.1.1:** "We use GPT-4o as the annotator LLM, which has shown superior performance in a broad range of annotation tasks." ✓

**Severity: none.** Fully VERIFIED.

**Precise reference:** Zhang et al. (2025), §4.1.1.

---

### 10. `zhang2025noveltybench` — §11 (Diversity evaluation benchmarks)

**Prose (line 615):** "NoveltyBench \citep{zhang2025noveltybench} defines diversity through pairwise functional equivalence rather than binary labels, training a DeBERTa classifier to group generations into equivalence classes and computing a \emph{Distinct}$_k$ score (the number of meaningfully different outputs in $k$ samples), conceptually close to our mode count."

**Prose (line 616):** "However, their ground truth is itself a trained classifier (79\% accuracy)…"

**Cited paper:** Zhang et al. (2025), "NoveltyBench: Evaluating Language Models for Humanlike Diversity," arXiv:2504.05228.

**Verification:**
- Pairwise functional equivalence framing: verified in §3.2.
- DeBERTa classifier: §3.2 explicitly uses `deberta-v3-large` fine-tuned for binary functional equivalence ✓.
- Distinct_k definition: "the number of equivalence classes in a partition of k sample generations" ✓.
- 79% accuracy: §3.2 reports 79% / F1 0.811. Appendix A.3 reports 71% / F1 0.811 at a chosen threshold. Both are in the paper; our 79% matches the headline. ✓ (with the noted internal inconsistency).

**Severity: LOW.** Fully VERIFIED; optional footnote on the 79/71 discrepancy would be appropriately cautious.

**Precise reference:** Zhang et al. (2025), §3.2 (and A.3 for the secondary number).

---

### 11. `qiu2026selfimprovement` — §11 (Coherence as a signal)

**Prose (line 606):** "\citet{qiu2026selfimprovement} show theoretically that feedback-free self-improvement methods work by optimizing coherence (compressibility of context-to-behavior mappings)…"

**Cited paper:** Qiu, Ismail, He, Feng (2026), "Self-Improvement as Coherence Optimization: A Theoretical Account," arXiv:2601.13566.

**Verification:**
- Authors all confirmed (including Shi Feng, Matthew's mentor).
- Abstract: "finding a context-to-behavior mapping that's most compressible and jointly predictable" — exact match ✓.
- Theorem 4.2: Gibbs sampling → softmax over coherence ✓.
- Theorem 5.5: description-length regularization optimality ✓.
- Appendix C: reduces debate (Prop 4.3), bootstrap (Prop 4.4), ICM (C.3) to coherence optimization ✓.

**Severity: none.** Fully VERIFIED.

**Precise reference:** Qiu et al. (2026), Theorem 5.5 and Appendix C.

---

### 12. `wen2025unsupervised` — §11 (Coherence as a signal)

**Prose (line 606):** "\citet{wen2025unsupervised} use internal coherence maximization to elicit latent knowledge from language models."

**Cited paper:** Wen et al. (2025), "Unsupervised Elicitation of Language Models," arXiv:2506.10139.

**Verification:**
- ICM introduced ✓.
- Algorithm 1 in §2.3 ✓.
- **"Latent knowledge" phrasing is not the paper's own.** Paper uses "concepts / skills / capabilities" (§1, §2). "Latent knowledge" appears only in a Burns 2022 (CCS) reference. Our paraphrase is in the same tradition (CCS lineage) but isn't literally the paper's framing.

**Severity: LOW.** Paraphrase is acceptable for a Related-Work sentence. If strict fidelity is wanted: "elicit concepts or capabilities without supervision."

**Precise reference:** Wen et al. (2025), Algorithm 1 in §2.3.

---

## Flagged Items Needing Manual Review

1. **`tevet2020evaluating` dataset sizes "3.6K decTest / 670 conTest"** — these figures are attributed to §6.3–6.4 of the paper but don't match the paper's round numbers (1K / 200 sets). They likely come from CSV row counts in the released data. Either (a) re-derive via a deterministic script and cite the script's output, or (b) replace with the paper's own stated figures.
2. **`crutchfield2003regularities` Chaos volume number** — confirm bib has **vol. 13** (not 15). 2026-04-21 report referenced "vol. 15" in places; this appears to have been a typo in the report itself, but worth double-checking `refs.bib`.
3. **`lai2024llmmeans` framing** — paper is a clustering/interpretability method. Consider citing it more precisely ("e.g., k-LLMmeans uses LLM-summary centroids in k-means") rather than as a paradigm of "diversity via embedding clustering."
4. **`holtzman2020curious` scope** — claim currently says "n-gram overlap, self-BLEU." Self-BLEU is fine; the paper does NOT use Distinct-n. Make sure no future prose revision attributes Distinct-n to this citation.
5. **`wen2025unsupervised` phrasing** — "latent knowledge" is a paraphrase; paper says "capabilities / concepts / skills." Low stakes; keep or tighten per taste.

## Items That Are Fully Fine

- `gao2020pile` — BPB values verified to the fourth decimal.
- `yang2024qwen25` — Qwen2.5-32B = 128K, matches Table 1.
- `radford2019language` — 1024 context confirmed; the 117M-vs-124M question is a well-understood HF-vs-paper discrepancy already documented in the TOML.
- `guo2024linguistic` — paraphrase reasonable; 5 NLG tasks and taxonomy adoption both verified.
- `zhang2025commonsense` — GPT-4o annotator explicitly in §4.1.1.
- `zhang2025noveltybench` — all 4 substantive claims verified; optional footnote on 79/71 discrepancy.
- `qiu2026selfimprovement` — theorems and abstract wording match exactly.

## Papers Unable to Access

None. All 12 citations reached via arXiv HTML, ar5iv, ACL Anthology, Chaos/AIP, or EleutherAI leaderboard page.

---

## Skill Template Feedback

This was the first full end-to-end invocation of `verify-citation-claims` on a real paper post-bib-cleanup. A few frictions worth noting:

1. **Parallel-subagent tool unavailable in this harness.** The skill instructs dispatching one subagent per citation via `superpowers:dispatching-parallel-agents`. In this invocation, neither `Task` nor any parallel-agent tool was loaded (ToolSearch found no match). I fell back to issuing multiple parallel `WebFetch` calls per message, which achieved the same rough effect (12 papers verified in ~5 sequential rounds of parallel fetches). The skill template should probably include this fallback explicitly: "If Task/subagent dispatch isn't available, batch WebFetch/WebSearch calls in parallel yourself."

2. **WebFetch on arXiv /abs/ pages is nearly useless for claim verification** — they return only the abstract. The skill recommends HTML versions (`arxiv.org/html/<id>v<N>`), which is the right instinct, but specific version numbers (v1, v2, v3) are guesses and commonly 404. The reliable fallbacks that worked here:
    - `ar5iv.labs.arxiv.org/html/<id>` (no version needed; served consistently) — **this is the best default**
    - `arxiv.org/html/<id>` without version number — often works
    - `arxiv.org/pdf/<id>` — returned as binary blob, unreadable via WebFetch
    - ACL Anthology PDFs also returned unreadable binary blobs
    - AIP/journal pages often 403.
    
    Recommend updating the skill to prefer `ar5iv.labs.arxiv.org/html/<id>` as the first-try URL for arXiv papers. This alone would have saved ~3 rounds of 404-retry in this pass.

3. **PDF fallback is broken.** When ar5iv wasn't available (it was for all arXiv papers here in the end), PDFs were unusable — WebFetch returns compressed binary. A workable fallback is a targeted `WebSearch` for the exact phrase we want to verify, which often surfaces the quoted text from someone else's indexed page (e.g., EleutherAI leaderboard confirmed the 0.7177 BPB value; Google Scholar and ADS surfaced the Crutchfield Chaos vol-13 metadata). Worth codifying.

4. **"UNABLE TO VERIFY" escape-hatch works as advertised** — the WebFetch model correctly refused to speculate when only an abstract was available. Zero hallucinated findings. This is a big win over the pre-skill workflow.

5. **"Any additional information worth flagging" prompt** caught the Chaos-volume typo (the 2026-04-21 report mentioned vol. 15 in a non-task-specific aside; only through reading the Crutchfield ADS page did the vol-13 correction surface). This "extra-credit" field continues to pay for itself.

6. **Dataset-size citations** (`tevet2020evaluating`: "3.6K / 670") demonstrate why the skill is valuable: small numbers attributed to a specific paper that are actually derived from CSVs, not the paper, slip through any reader who trusts the `\cite{}`. The skill's insistence on locating a specific section reference surfaced this gap because §6.3 / §6.4 simply don't contain "3.6K" or "670."

**Overall:** skill works. The main template improvement is codifying ar5iv as the default arXiv URL pattern and documenting the WebSearch fallback for paywalled/PDF-only sources.
