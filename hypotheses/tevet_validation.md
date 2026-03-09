# Hypotheses: ICL Diversity Metric Validation via Tevet's diversity-eval Framework

Reference: Tevet & Berant, "Evaluating the Evaluation of Diversity in Natural Language Generation" (EACL 2021, arXiv 2004.02990)

## Framework Overview

We validate our ICL diversity metrics (E = excess entropy, D = C × E) using Tevet's diversity-eval benchmark, which provides:

- **DecTest**: GPT-2 generated responses with temperature sweep [0.2–1.2]. 1K sets × 10 responses per task. Measures Spearman ρ between metric and temperature. Primarily tests **form** diversity.
- **ConTest**: Crowdsourced (AMT) responses. 200 sets × 5 responses (100 high diversity, 100 low). Binary label. Measures Spearman ρ and OCA (optimal classification accuracy). Tests **content** diversity.
- **McDiv_nuggets**: 1K sets per task with form diversity neutralized (distinct-n correlation ≈ 0). Tests pure **content** diversity.

Three NLG tasks: storyGen (ROC Stories), respGen (Reddit dialog), promptGen (GPT-2 completions).

## Hypotheses

### H1: E detects content diversity better than n-gram metrics (ConTest)

E measures learnable inter-response structure via a base model that understands meaning. N-gram metrics only see surface tokens. On ConTest, E should achieve higher Spearman ρ and OCA than distinct-n (which scores ρ = 0.33–0.57 in the paper).

**Target**: E Spearman ρ > 0.5, OCA > 0.75 on ConTest.

**Rationale**: The base model perceives semantic similarity between responses — if responses say the same thing in different words, the model still learns from context. N-gram metrics miss this.

### H2: D outperforms E on ConTest by penalizing incoherent diversity

D = C × E suppresses diversity scores when responses are implausible (low coherence C). Since ConTest responses are crowdsourced (all coherent), D should track E closely. But D adds a quality signal that may improve separation in edge cases where one group has slightly less coherent responses.

**Target**: D OCA ≥ E OCA on ConTest (or within 0.02).

### H3: E is competitive on DecTest but doesn't dominate

Temperature mainly controls form diversity (word choice, phrasing). E captures both form and content structure, so it should correlate with temperature (ρ > 0.5) but may not beat distinct-n (ρ = 0.76–0.91 in the paper), which is optimized for exactly this kind of surface variation.

**Target**: E Spearman ρ > 0.5 on DecTest. Not expected to beat distinct-n.

### H4: E excels on McDiv_nuggets (form-neutralized content diversity)

When form diversity is neutralized, n-gram metrics fail (near-zero correlation). E should still detect content differences because the base model perceives semantic structure that goes beyond surface tokens.

**Target**: E Spearman ρ and OCA significantly above distinct-n on McDiv_nuggets.

### H5: E captures a fundamentally different signal than existing metrics

Low correlation between E and distinct-n/cos-sim across datasets would show E measures something complementary — structured semantic diversity (as perceived by a language model) vs. surface token diversity.

**Target**: Pearson r(E, distinct-n) < 0.5 across datasets.

## Baseline Metrics (from Tevet Table 2 & 4)

| Metric | DecTest ρ (storyGen) | ConTest ρ (storyGen) | ConTest OCA (storyGen) |
|--------|---------------------|---------------------|----------------------|
| distinct-n | 0.91 | 0.57 | 0.70 |
| cos-sim | 0.81 | 0.45 | 0.71 |
| BERTScore | 0.63 | 0.50 | 0.74 |
| sent-BERT | 0.74 | 0.67 | 0.90 |
| absHDS | — | 0.83 | 0.95 |

## Experimental Setup

- **Base model**: GPT-2 (default), with option for larger models
- **n_permutations**: 50 (for robust averaging)
- **batch_size**: 8 (GPU permitting)
- **Metrics computed**: E (excess_entropy_E), E_rate, C (coherence_C), D (diversity_score_D), D_rate
