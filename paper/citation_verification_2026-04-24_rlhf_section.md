# Citation verification — `paper/rlhf_experiment.tex` (RLHF-diversity section)

Date: 2026-04-24. Scope: the 7 new citations added during the RLHF-diversity
experiment. Method: per-citation parallel WebFetch agents reading the actual
papers (not just abstracts) and locating section / table evidence for each
claim. Skill: `verify-citation-claims`.

## Summary

| Key | Status | Severity |
|---|---|---|
| `kirk2023rlhf` | Discrepancy on H1a citation + minor discrepancies | **HIGH** |
| `zhang2025verbalizedsampling` | All claims verified; H1' framing strengthened | OK (LOW) |
| `padmakumar2023writingreduces` | Discrepancy on H1a citation | **HIGH** |
| `olmo2024olmo2` | Two minor scope discrepancies | MEDIUM |
| `dubois2024alpacafarm` | Citation defensible but imprecise | LOW |
| `reimers2019sbert` | Citation defensible but loose | LOW |
| `dror2018hitchhiker` | Bonferroni attribution unsupported | **HIGH** |

Three HIGH-severity items are real prose / cite-target issues that would
mislead readers. They block "submission-ready" status for the section but
are easy to fix.

---

## HIGH

### `kirk2023rlhf` — H1a overreach

**Issue.** We cite Kirk et al. for H1a ("base > SFT" — "SFT narrows toward a
single helpful-assistant style"). Kirk et al.'s diversity experiment compares
**SFT vs RLHF vs BoN only** (Figs. 5–6). They do not sample from the
pretrained LLaMA-7B for diversity measurement, so they have no base→SFT
data point. The mode-collapse phrasing they do use (§6.2) is about RLHF
narrowing the across-input distribution **relative to SFT**, not base.

**Fix options:**
- Drop `kirk2023rlhf` from the H1a citation list; keep it for H1c (base
  vs RLVR; here "base" stands in for "before all post-training," which
  Kirk et al. don't measure either, but the cumulative RLHF-vs-SFT
  finding is theirs and reasonably extends).
- Reframe H1a's prose to "SFT may narrow…" with a softer hedge and
  cite Padmakumar & He (also imperfect — see below) plus a base-vs-SFT
  paper that actually measures base.
- Best: cite Bai et al. 2022 ("Training a Helpful and Harmless
  Assistant…") which does measure pretrain vs HH-SFT, OR omit a
  citation and let H1a stand on the empirical result.

### `kirk2023rlhf` — minor discrepancies

- **Metric count.** Our `refs_ids.toml` lists "EAD / distinct-n / SentBERT
  / NLI" as four separate metrics. Kirk §5.2 actually names **three**:
  distinct-N (with EAD de-biasing per Liu 2022), SentBERT, NLI. EAD is
  Kirk's implementation of distinct-N, not a separate metric. Reword
  the toml claim and the section's parenthetical "Kirk et al.'s EAD and
  distinct-n" to "distinct-n with EAD de-biasing".
- **Dataset scope.** Kirk's diversity finding is on **TL;DR only**;
  §6.2 explicitly says they saw "no meaningful differences" on
  AlpacaFarm and dropped it. Our toml claim "TL;DR + AlpacaFarm"
  describes the experimental setup but not the diversity finding.
  Reword toml.

### `padmakumar2023writingreduces` — H1a overreach

**Issue.** Paper compares GPT-3 (`davinci`) vs InstructGPT
(`text-davinci-003`). InstructGPT is SFT + RLHF; the paper cannot isolate
the SFT contribution. Citing it for H1a (the SFT-specific narrowing claim)
is the same overreach as Kirk.

**Fix options:**
- Move this citation from the H1a justification to H1c (base vs full
  post-training) where it fits cleanly.
- Or reframe H1a as "post-training (SFT and beyond) narrows" rather
  than singling out SFT.

### `dror2018hitchhiker` — Bonferroni attribution unsupported

**Issue.** Section 4 of Dror 2018 only **surveys** that 3/110 ACL-2017
papers used Bonferroni; it does not recommend it. The paper explicitly
defers multiple-comparison methodology to **Dror, Baumer, Bogomolov,
Reichart 2017 ("Replicability Analysis for NLP")**. The 2018 paper does
recommend Wilcoxon signed-rank as a valid sampling-free non-parametric
choice (§3.2.2), but its top-level recommendation is the Figure 1
decision tree, which prefers bootstrap / permutation when feasible.

**Fix options:**
- Add `dror2017replicability` (the actual replicability paper) as a
  separate citation for the Bonferroni step. Drop Bonferroni
  attribution from the dror2018hitchhiker `claim` and keep it as the
  Wilcoxon source.
- Switch the protocol to Holm-Bonferroni (uniformly more powerful
  than plain Bonferroni for the same family-wise α) and cite Holm 1979
  directly. Bonferroni-vs-Holm makes no difference for k=3 contrasts
  with all p < 0.001, so the practical result is unchanged.
- Soft fix: drop the explicit Bonferroni mention and just call it
  "family-wise α = 0.05 / 3 correction across the three pre-registered
  contrasts" without naming the method.

---

## MEDIUM

### `olmo2024olmo2`

- **RLVR domains.** Our toml says "verifiable rewards for math, code,
  IF". OLMo 2 §5 explicitly **excludes** code from RLVR ("Although
  Tülu 3 uses six categories including code-related tasks, we exclude
  this category"). Reword to "math + IF" for OLMo 2 specifically.
- **Sizes.** Our toml lists "7B, 13B, 32B" as released post-trained
  sizes. The paper body covers 7B and 13B; 32B was added in v3
  (Oct 2025) abstract only and the post-training recipe is described
  for 7B/13B. Our 7B experiment is unaffected; just trim the toml
  claim to "7B and 13B (32B base added in v3)".

The core claim — `-Instruct` is the RLVR-final stage — is verified
explicitly: Table 16 caption "**The final Instruct model is from the
RLVR stage.**" Figure 13 labelled "OLMo-2-1124-13B-Instruct (Final
RLVR)". Our experiment's stage labelling is correct.

---

## LOW

### `dubois2024alpacafarm`

- AlpacaFarm constructs and releases the 805-prompt eval set
  (Figure 15, prose around it). The "AlpacaEval" name is a later
  release built on top of the same prompts. Our citation is factually
  correct but imprecise.
- **Suggested fix:** keep `dubois2024alpacafarm` (or rename to
  `dubois2023alpacafarm` to match the year on arXiv submission), and
  optionally add an `alpaca_eval` `@misc` entry for the actual HF
  release. Reword the prose to "the 805-prompt AlpacaFarm evaluation
  set (later distributed as AlpacaEval)".

### `reimers2019sbert`

- `sentence-transformers/all-MiniLM-L6-v2` uses MiniLM (Wang 2020,
  arXiv:2002.10957) as the base encoder and a contrastive objective
  on 1B sentence pairs — **not** the original 2019 siamese-BERT
  setup. Same author (Reimers), same library, different recipe.
- **Suggested fix:** keep the citation as the methodology / library
  source (it's the canonical SBERT paper). Optionally add Wang 2020
  (MiniLM) for the base architecture. Or just rephrase to
  "sentence-transformers library [Reimers & Gurevych 2019]" and call
  it the library citation, not the model-card citation.

### `zhang2025verbalizedsampling` — bonus finding

- All explicit claims verified. The §3 typicality-bias mechanism IS
  preference-data-specific (derived from the Bradley-Terry / RLHF
  objective in Eq. 2), so our H1' framing — "the typicality-bias
  argument doesn't obviously apply to RLVR" — is defensible at the
  *theory* level.
- **However**, §5.3 (their Tülu-3 stage ablation) shows diversity
  drops at *every* post-training stage including RLVR. So while the
  *mechanism* doesn't extend to RLVR, the *empirical phenomenon*
  does in their data.
- **Suggested fix (optional, prose-only):** add a sentence in our H1'
  prose: "Zhang et al. observe diversity loss at the RLVR stage too
  (their §5.3) but do not extend their typicality-bias *mechanism* to
  verifiable rewards." This makes our exploratory framing more
  precise without weakening it.

---

## Recommended next steps (order)

1. Fix the three HIGH items (H1a citations + Bonferroni). Smallest
   actionable patch: drop kirk2023rlhf and padmakumar2023writingreduces
   from H1a, add `dror2017replicability` for the Bonferroni step.
2. Reword the OLMo 2 toml claim (drop "code" from RLVR, trim the
   sizes list).
3. Reword the Kirk toml claim (3 metrics not 4; TL;DR-only diversity
   finding).
4. Optionally add Wang 2020 (MiniLM) and the AlpacaEval `@misc`
   entry; or accept the LOW imprecision.
5. Optionally add the H1' hedging sentence (zhang2025verbalizedsampling).

All HIGH and MEDIUM fixes are prose / `refs_ids.toml` edits — none
require re-running the experiment. Bonferroni → Holm change would not
shift any p-value below the α = 0.05/3 threshold (all reported
contrasts are at p < 0.001).
