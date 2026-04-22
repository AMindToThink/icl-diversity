# Citation Verification Report

Date: 2026-04-21
Paper: `paper/in_context_diversity_metric.tex`
Method: Each citation was verified by a parallel subagent that located the actual paper via web search and cross-checked the specific claim in our paper against the cited work.

> **Resolution (2026-04-22):** The fabricated-author keys below (`chen2025linguistic`, `he2025commonsense`, `lam2025noveltybench`) were renamed to match the real first authors (`guo2024linguistic`, `zhang2025commonsense`, `zhang2025noveltybench`) in commit `1ef7a33`. The fourth fabricated-author key, `lai2024llmmeans`, was missed in that sweep and renamed to `diaz2025llmmeans` in a later pass. The old keys are preserved here for historical accuracy; anyone `grep`ping the current tree will find them only in this report. The `zhang2024writingprompts` HIGH-severity entry was replaced by `holtzman2020curious` earlier (commit `7313b47`). A follow-up claim-level audit ran 2026-04-22 — see `citation_claim_verification_2026-04-22.md`.

## Summary Table

| Bibkey | Status | Severity | Issue |
|---|---|---|---|
| `lai2024llmmeans` | **FABRICATED authors** | HIGH | Real paper has sole author Jairo Diaz-Rodriguez (2025), not the 7-author list in our bib |
| `gao2020pile` | **Verified** | — | Numbers match Table 2 exactly |
| `radford2019language` | Minor | LOW | 124M is HF checkpoint count; the paper's Table 2 says 117M |
| `yang2024qwen25` | **Wrong number** | MEDIUM | Qwen2.5-32B context is 128K, not 32K as our paper states |
| `tevet2020evaluating` | Multiple issues | MEDIUM | "three frameworks" is loose; 13,679 count doesn't match; Appendix C.1 construction description is wrong |
| `crutchfield2003regularities` | Verified + title typo | LOW | Title should be "Levels of Entropy Convergence" not "entropy convergence hierarchy"; otherwise supported as analogy |
| `zhang2024writingprompts` | **Wrong year + wrong claim** | HIGH | Paper is from 2020, not 2024; uses Shannon entropy, not n-gram/self-BLEU. Citation does not support the claim |
| `chen2025linguistic` | **FABRICATED authors** | HIGH | Actual first author is Yanzhu Guo (not Chen); arXiv preprint (not TACL) |
| `he2025commonsense` | **FABRICATED authors** | HIGH | Actual authors: Tianhui Zhang, Bei Peng, Danushka Bollegala. No author named "He" on the paper |
| `lam2025noveltybench` | **FABRICATED authors** | HIGH | Actual first author is Yiming Zhang (not Lam). 79% figure has internal inconsistency with Appendix A.4 (71%) |
| `qiu2026selfimprovement` | **Verified** | — | Paper exists, authors correct, claim supported |
| `wen2025unsupervised` | **Verified** | — | Authors match, ICM is the paper's method. Minor wording note |

**High-severity issues: 5 (four fabricated author lists + one unsupported claim).**

---

## Detailed Findings

### 1. `lai2024llmmeans` — Section 1 (Motivation) and Section 11 (Related Concepts)

**Our paper claims:** embedding-based diversity clustering, e.g., k-LLMmeans.

**Bib entry says:** Viet Dac Lai, Chien Van Nguyen, Nghia Trung Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan A. Rossi, Thien Huu Nguyen — 2024.

**Actual paper:**
- arXiv:2502.09667 (submitted Feb 2025; v5 title renamed to "Summaries as Centroids for Interpretable and Scalable Text Clustering")
- Sole author: **Jairo Diaz-Rodriguez** (York University)
- Year: **2025**, not 2024

**Claim support:** Partial. The paper uses embeddings + k-means, but never frames itself as a diversity metric — it is a clustering/interpretability method. Citing it for "existing approaches measure diversity via embedding-space clustering" overstates its scope.

**Recommended fix:** Update the bib entry to the correct author and year. Rename key to `diazrodriguez2025kllmmeans`. Consider supplementing with a paper that actually proposes an embedding-based diversity metric (e.g., Tevet & Berant's BERT-based diversity, Du et al.'s embedding-cluster diversity).

**More precise reference:** Section 3 ("Preliminaries: k-Means for text clustering") or Algorithm 1 of the actual paper.

---

### 2. `gao2020pile` — Section 5.3 (Typical range of C)

**Our paper claims:** GPT-3 davinci ≈ 0.72 bits/byte on The Pile; GPT-2 XL ≈ 1.05 bits/byte.

**Actual values from Table 2 ("Test perplexity of the Pile…") of Gao et al. 2020:**
- GPT-3 davinci: **0.7177** → rounds to 0.72 ✓
- GPT-2 XL: **1.0468** → rounds to 1.05 ✓

**Verdict:** Fully verified.

**More precise reference:** Gao et al. (2020), Table 2, overall "The Pile" row (Section 4, "Evaluation").

---

### 3. `radford2019language` — Section 8.1 (Experimental Setup)

**Our paper claims:** GPT-2 (124M parameters, 1024-token context window).

**Actual paper's Table 2:** The smallest model is reported as **117M** parameters. Context window = 1024 ✓.

**Resolution:** 124M is the parameter count of the released HuggingFace `gpt2` checkpoint (includes embedding parameters the paper's table omits). HuggingFace's corrected count is widely accepted.

**Recommended fix:** Either (a) add a footnote clarifying that we use the HF checkpoint with 124,439,808 parameters (vs. Table 2's 117M), (b) change "124M" to "117M" to match the cited source, or (c) cite both the paper and the HF model card. Option (a) is cleanest.

**More precise reference:** Radford et al. (2019), Table 2, "Architecture hyperparameters for the 4 model sizes" (between Sections 2.1 and 2.2); Section 2.2 "Input Representation" for the 1024 context length.

---

### 4. `yang2024qwen25` — Section 8.1 (Experimental Setup)

**Our paper claims:** Qwen2.5-32B (32B parameters, 32K-token context window).

**Actual paper (Yang et al. 2024, Table 1, "Model architecture and license of Qwen2.5 open-weight models"):**
- 32B parameters ✓
- Context/generation length: **128K / 8K** for 32B (not 32K)

The 32K figure corresponds to Qwen2.5-3B (and 0.5B, 1.5B); Qwen2.5-7B/14B/32B/72B are all 128K.

**Recommended fix:** Change "32K-token context window" → "**128K-token context window**" and cite Yang et al. (2024), Table 1.

Note: During pre-training the base model is first extended to 32,768 tokens and then via YaRN/DCA to 131,072 (Yang et al., §3.1.3). The released/supported context is 128K.

---

### 5. `tevet2020evaluating` — Section 8.5 (Tevet) and Appendix C.1 (Confound)

**Our paper makes multiple claims; several have issues.**

**(a) "Three evaluation frameworks"** — *Imprecise.* The paper defines **two tests** (decTest, conTest) plus **one dataset** (McDiv, with McDiv_nuggets as a subset). Recommend rephrasing to "two diagnostic tests (decTest, conTest) and the McDiv dataset (including the McDiv_nuggets subset)."

**(b) "13,679 response sets"** — *Does not match.* Summing the released CSVs yields ~13,929. The paper's round numbers are ≈6K McDiv + 3K McDiv_nuggets + ≈3.6K decTest + ≈670 conTest. Our 13,679 figure should be either (i) rederived with a deterministic script we own and cited, or (ii) replaced with Tevet's round numbers.

**(c) McDiv_nuggets construction in Appendix C.1** — *Partially wrong.* Our paper says: "McDiv_nuggets creates low-diversity response sets by giving crowd-workers a specific ending and asking them to paraphrase it five times. The chosen endings tend to be specific or dramatic..."

Actual procedure (Tevet & Berant §6.4, "Data and settings," p. 332-333): For high content diversity, workers give 5 different responses. Then "we asked the **same workers** to choose a **single response they wrote**, and rephrase it 5 times." Workers are **not** given a pre-chosen dramatic ending; they **self-select** one of their own responses. This changes the confound story: the "dramatic/specific" flavor comes from within-worker-pool self-selection, not experimenter-curated endings.

**(d) "Two diversity levels"** (Section 11) — ✓ Correct.

**(e) Per-benchmark structure:**
- conTest: 5 responses per set, binary labels ✓ (§6.4, "Data and settings")
- decTest: 10 responses per set, temperature-labeled (temperature sweep over [0.2, 1.2]) ✓ (§6.3)
- McDiv_nuggets: 5 responses per set, binary labels ✓ (§6.4)

**More precise references:** Tevet & Berant (2021), §3.1 (decTest definition), §3.2 (conTest definition), §6.3 (decTest data/settings), §6.4 (conTest/McDiv data/settings, p. 332-333).

---

### 6. `crutchfield2003regularities` — Section 11 and Appendix B

**Our paper:** Cites as source of "excess entropy from computational mechanics."

**Actual paper:**
- arXiv:cond-mat/0102181 (NOT the 0108181 that appears in some references)
- Actual title: **"Regularities Unseen, Randomness Observed: Levels of Entropy Convergence"** — our bib says "The entropy convergence hierarchy" (close but not the published title)
- Published: Chaos 15, 25–54 (2003)

**Excess entropy definition (Section III.B, Eq. 48):** E ≡ I₁ = Σ_L [h_μ(L) − h_μ], where h_μ(L) is the block-entropy estimate and h_μ is the asymptotic entropy rate. Units: bits.

**Claim support:** Our formula E = Σ_k (a_k − a_∞) is a structural analog (conditional cross-entropies under θ sum their excess above a floor). The analogy is defensible but not identical: their setup uses true entropy rates of a stationary stochastic process; ours uses cross-entropy under a base model (so our a_∞ absorbs both irreducible noise and a KL term). Their excess entropy has additional properties (e.g., past-future mutual information equivalence) that do not transfer cleanly.

**Recommended fix:** (i) Fix the title in the bib entry to "Regularities Unseen, Randomness Observed: Levels of Entropy Convergence." (ii) Optionally soften language like "the concept of excess entropy from computational mechanics" to "inspired by" or "analogous to" if strict precision is desired — the connection is an analogy, not a direct application.

**More precise reference:** Crutchfield & Feldman (2003), Section III.B, Eq. 48 (definition of excess entropy E).

---

### 7. `zhang2024writingprompts` — Section 11 (Sampling diversity metrics)

**Our paper claims:** "Work on decoding-time diversity [cite] typically operates at the surface level (n-gram overlap, self-BLEU)."

**Actual paper (Zhang, Duckworth, Ippolito, Neelakantan):**
- arXiv:2004.10450
- Submitted **22 April 2020** — our bib's year (2024) is wrong
- The paper's diversity metric is **Shannon entropy** H(p) = −E[log p(x)] of the model distribution (Section 2). It does **not** use n-gram overlap or self-BLEU. BLEU appears only in passing in Related Work.
- The bib key mentions "writingprompts," but the paper uses the GPT-2 test set, not the WritingPrompts dataset.

**Claim support:** Weak. The paper is about decoding-time diversity (temperature, top-k, nucleus), but it does not use the n-gram / self-BLEU metrics we attribute to it.

**Recommended fix:** Replace with a citation that actually uses n-gram / self-BLEU for decoding-time diversity. Strong candidates:
- **Holtzman et al. (2020)** "The Curious Case of Neural Text Degeneration" (ICLR 2020) — uses self-BLEU and distinct-n to evaluate nucleus sampling. This is probably the ideal single citation.
- Zhu et al. (2018), Texygen (arXiv:1802.01886) — introduced self-BLEU as a diversity metric.
- Li et al. (2016) "A Diversity-Promoting Objective Function" — uses distinct-n.

If the Zhang et al. paper is kept, fix the year (2020) and rename the key (e.g., `zhang2020trading`); but the claim still isn't supported.

---

### 8. `chen2025linguistic` — Section 11 (Diversity evaluation benchmarks)

**Our paper:** Cites "Chen, Wei, Wang (2025), Benchmarking linguistic diversity of LLMs, TACL 2025" as extending Tevet's framework to broader generation tasks.

**Actual paper:**
- arXiv:2412.10271 (v2, Jul 2025) — **arXiv preprint, not TACL**
- Actual authors: **Yanzhu Guo, Guokan Shang, Chloé Clavel** (Inria / MBZUAI). First name matches but surname is Guo, not Chen. The other two co-authors are different.

**Claim support:** Partial. Guo et al. adopt Tevet & Berant's form/content diversity taxonomy (§3) but explicitly build on **Guo et al. 2024b**'s framework, not Tevet's. They evaluate across 5 NLG tasks (LM, MT, summarization, next-utterance, story generation). So "extending Tevet's framework" slightly overclaims the lineage.

**Recommended fix:**
- Correct authors to Yanzhu Guo, Guokan Shang, Chloé Clavel.
- Correct venue to arXiv:2412.10271 (not TACL).
- Rename bib key (e.g., `guo2024benchmarking`).
- Consider rewording: "adopting Tevet and Berant's taxonomy across a broader set of generation tasks" rather than "extending Tevet and Berant's framework."

---

### 9. `he2025commonsense` — Section 11 (Diversity evaluation benchmarks)

**Our paper:** "[cite] conduct a meta-evaluation of diversity metrics for constrained commonsense generation, using GPT-4o as an annotator in place of crowd workers." Bib says authors are "Yun He, Pengfei Liu, and others."

**Actual paper:**
- arXiv:2506.00514 ✓
- Title: "Evaluating the Evaluation of Diversity in Commonsense Generation" ✓
- **Actual authors: Tianhui Zhang, Bei Peng, Danushka Bollegala** (University of Liverpool) — **no author named "He" on the paper**

**Claim support:** All substantive claims verified:
- Meta-evaluation of 12 diversity metrics for GCR ✓ (Section 2, §4)
- Focus on commonsense generation (CommonGen dataset) ✓
- GPT-4o used as the annotator ✓ (§4.1.1)
- Replaces crowdsourcing for diversity annotation (§2, §4.1.2)

**Recommended fix:**
- Correct the bib authors to Tianhui Zhang, Bei Peng, Danushka Bollegala.
- Rename bib key (e.g., `zhang2025evaluating` or `zhang2025commonsense`).

**More precise reference:** Zhang et al. (2025), §4.1.1 (Prompt Engineering, "We use GPT-4o as the annotator LLM…").

---

### 10. `lam2025noveltybench` — Section 11 (Diversity evaluation benchmarks)

**Our paper makes 5 specific claims.** Bib says authors are "Xuan Phi Lam, Gia-Huy Do, and others."

**Actual paper:**
- arXiv:2504.05228
- Title: **"NoveltyBench: Evaluating Language Models for Humanlike Diversity"** (our bib title is wrong)
- **Actual authors (8): Yiming Zhang, Harshita Diddee, Susan Holm, Hanchen Liu, Xinyue Liu, Vinay Samuel, Barry Wang, Daphne Ippolito** — **neither "Lam" nor "Do" appears**

**Claim support (all 5):**
1. "Pairwise functional equivalence rather than binary labels" ✓ (§3.2)
2. "DeBERTa classifier to group generations into equivalence classes" ✓ — specifically `deberta-v3-large` fine-tuned for binary functional equivalence (§3.2)
3. "Distinct_k score (number of meaningfully different outputs in k samples)" ✓ (Eq. 1)
4. "Conceptually close to our mode count" ✓ (editorial judgment, reasonable)
5. "Ground truth is itself a trained classifier (79% accuracy)" — **supported but with internal paper inconsistency.** Main body (§3.2) reports 79% accuracy / 0.811 F1; Appendix A.4 reports 71.0% accuracy / 0.811 F1 for apparently the same classifier. The F1 matches; the accuracies do not. Our 79% matches the headline number; a footnote acknowledging the discrepancy would be appropriately cautious.

**Recommended fix:**
- Correct authors to Yiming Zhang et al.
- Correct title to "NoveltyBench: Evaluating Language Models for Humanlike Diversity."
- Rename bib key (e.g., `zhang2025noveltybench`).
- Optional: footnote the 79% / 71% discrepancy in the source.

---

### 11. `qiu2026selfimprovement` — Section 11 (Coherence as a signal)

**Our paper claims:** "[cite] show theoretically that feedback-free self-improvement methods work by optimizing coherence (compressibility of context-to-behavior mappings)."

**Actual paper (arXiv:2601.13566, submitted 20 Jan 2026):**
- Title: "Self-Improvement as Coherence Optimization: A Theoretical Account" ✓
- Authors: Tianyi Qiu, Ahmed Hani Ismail, Zhonghao He, Shi Feng ✓ (includes Matthew's mentor Shi Feng)
- Theoretical results: Theorem 4.2 (Gibbs-sampling convergence), Theorem 5.5 (optimality of description-length regularization); Appendix C shows debate, bootstrap, ICM are reducible to coherence optimization.
- Abstract literally states: "finding a **context-to-behavior mapping that's most compressible** and jointly predictable" — our paraphrase is faithful.

**Verdict:** Fully verified.

**More precise reference:** Qiu et al. (2026), Theorem 5.5 (or Appendix C for the specific reduction of feedback-free methods).

---

### 12. `wen2025unsupervised` — Section 11 (Coherence as a signal)

**Our paper claims:** "[cite] use internal coherence maximization to elicit latent knowledge from language models."

**Actual paper (arXiv:2506.10139, Wen et al.):**
- Full 13-author list matches our bib exactly ✓
- Title: "Unsupervised Elicitation of Language Models" ✓
- Introduces Internal Coherence Maximization (ICM) — Algorithm 1, §2.3 ✓

**Claim support:** Mostly accurate. One minor wording note: the paper frames its goal as eliciting "concepts / skills / capabilities" from pretrained models. The phrase "latent knowledge" appears in the paper only when citing Burns et al. 2022 (CCS). Our characterization as "elicit latent knowledge" is a reasonable paraphrase in the same tradition, but if strict terminological fidelity matters, consider rewording to "elicit concepts/capabilities" or "elicit knowledge without supervision."

**More precise reference:** Wen et al. (2025), §2.3 Algorithm 1 (Internal Coherence Maximization), or Abstract for the overall framing.

---

## Flagged Items Needing Matthew's Manual Review

1. **Four bib entries with fabricated or wholly wrong author lists** (`lai2024llmmeans`, `chen2025linguistic`, `he2025commonsense`, `lam2025noveltybench`). These are the highest-priority fixes — they appear to be LLM-generated author hallucinations. Each real paper was identified; fixes are listed per entry above.
2. **One bib entry with wrong year + claim that is not actually supported** (`zhang2024writingprompts`). The paper is from 2020 and uses Shannon entropy, not n-gram/self-BLEU. Replacement candidates: Holtzman et al. (2020), Zhu et al. (2018), Li et al. (2016).
3. **Numerical discrepancy in Qwen2.5-32B context window** (32K → 128K in Section 8.1). This is a fact check, not a citation issue, but lives in a sentence that also cites Yang et al.
4. **Appendix C.1 (McDiv construction)** incorrectly describes how low-diversity sets are built. Workers self-select from their own high-diversity responses; they are not given "a specific ending." This affects the confound narrative somewhat.
5. **"13,679 response sets"** (Section 8.5 intro) should be rederived from a deterministic script or replaced with Tevet's round figures.
6. **Minor title typo** in `crutchfield2003regularities` — should be "Levels of Entropy Convergence."
7. **Minor parameter-count footnote** for GPT-2 (117M paper vs. 124M HF checkpoint).

## Items That Are Fully Fine

- `gao2020pile` — verified, numbers correct.
- `qiu2026selfimprovement` — verified, authors and claim accurate.
- `wen2025unsupervised` — verified, with an optional wording consideration around "latent knowledge" vs. "capabilities."

## Papers Unable to Access

None. All 12 citations were located and read.

---

## Title Verification Table

Exact title in our bib (case-insensitive) compared against the title on the canonical source (arXiv / OpenAI).

| Bibkey | Title in our bib | Actual title | Match |
|---|---|---|---|
| `lai2024llmmeans` | "k-LLMmeans: Combining LLMs and k-means for zero-shot clustering of text" | "Summaries as Centroids for Interpretable and Scalable Text Clustering" (current v5, ICLR 2026); v1 was "k-LLMmeans: Summaries as Centroids for Interpretable and Scalable LLM-Based Text Clustering" | **NO — our title appears fabricated** |
| `gao2020pile` | "The Pile: An 800GB dataset of diverse text for language modeling" | "The Pile: An 800GB Dataset of Diverse Text for Language Modeling" | YES (capitalization only) |
| `radford2019language` | "Language models are unsupervised multitask learners" | "Language Models are Unsupervised Multitask Learners" | YES (capitalization only) |
| `yang2024qwen25` | "Qwen2.5: A party of foundation models" | "Qwen2.5 Technical Report" | **NO** — the phrase "A party of foundation models" is the Qwen blog tagline, not the arXiv title |
| `tevet2020evaluating` | "Evaluating the evaluation of diversity in natural language generation" | "Evaluating the Evaluation of Diversity in Natural Language Generation" | YES (capitalization only) |
| `crutchfield2003regularities` | "Regularities unseen, randomness observed: The entropy convergence hierarchy" | "Regularities Unseen, Randomness Observed: Levels of Entropy Convergence" | **NO** — subtitle is "Levels of Entropy Convergence," not "The entropy convergence hierarchy" |
| `zhang2024writingprompts` | "Trading off diversity and quality in natural language generation" | "Trading Off Diversity and Quality in Natural Language Generation" | YES (capitalization only) |
| `chen2025linguistic` | "Benchmarking linguistic diversity of large language models" | "Benchmarking Linguistic Diversity of Large Language Models" | YES (capitalization only) — but authors/venue are still wrong |
| `he2025commonsense` | "Evaluating the evaluation of diversity in commonsense generation" | "Evaluating the Evaluation of Diversity in Commonsense Generation" | YES (capitalization only) — but authors are still wrong |
| `lam2025noveltybench` | "NoveltyBench: Evaluating creativity and diversity in language models" | "NoveltyBench: Evaluating Language Models for Humanlike Diversity" | **NO** |
| `qiu2026selfimprovement` | "Self-improvement as coherence optimization: A theoretical account" | "Self-Improvement as Coherence Optimization: A Theoretical Account" | YES (capitalization only) |
| `wen2025unsupervised` | "Unsupervised elicitation of language models" | "Unsupervised Elicitation of Language Models" | YES (capitalization only) |

**Four title mismatches beyond capitalization:** `lai2024llmmeans`, `yang2024qwen25`, `crutchfield2003regularities`, `lam2025noveltybench`.

---

## Links to All Papers

| Bibkey | Canonical URL |
|---|---|
| `lai2024llmmeans` | https://arxiv.org/abs/2502.09667 |
| `gao2020pile` | https://arxiv.org/abs/2101.00027 |
| `radford2019language` | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| `yang2024qwen25` | https://arxiv.org/abs/2412.15115 |
| `tevet2020evaluating` | https://arxiv.org/abs/2004.02990 (also https://aclanthology.org/2021.eacl-main.25/ for EACL version) |
| `crutchfield2003regularities` | https://arxiv.org/abs/cond-mat/0102181 (journal: Chaos 15, 25–54, 2003) |
| `zhang2024writingprompts` | https://arxiv.org/abs/2004.10450 |
| `chen2025linguistic` | https://arxiv.org/abs/2412.10271 |
| `he2025commonsense` | https://arxiv.org/abs/2506.00514 |
| `lam2025noveltybench` | https://arxiv.org/abs/2504.05228 |
| `qiu2026selfimprovement` | https://arxiv.org/abs/2601.13566 |
| `wen2025unsupervised` | https://arxiv.org/abs/2506.10139 |
