# Tevet & Berant (2021) dataset-size audit

**Flag origin:** MEDIUM in `/home/cs29824/matthew/icl-diversity/paper/citation_claim_verification_2026-04-22.md` on the `tevet2020evaluating` citation — paper's per-dataset totals looked like row counts, not Tevet-stated sizes.

## The claim in our paper

`/home/cs29824/matthew/icl-diversity/paper/in_context_diversity_metric.tex`, line 509:

> "We evaluate against Tevet and Berant's diversity-eval benchmark \citep{tevet2020evaluating}, which comprises two diagnostic tests (decTest, conTest) and the McDiv dataset (including the McDiv\_nuggets subset), totalling **roughly 6K McDiv sets, 3K McDiv\_nuggets sets, 3.6K decTest sets, and 670 conTest sets** (Tevet and Berant \S6.3--6.4)"

## The claim in Tevet

Per the published paper (via `ar5iv.labs.arxiv.org/html/2004.02990`, cross-checked against the CSV column schema):

- **decTest (\S6.3):** "1K sets in total for each of 1K contexts (10 per temperature) and evaluated 200 (2 random sets per temperature)." This is **per task**, and there are three NLG tasks (storyGen, respGen, promptGen). So Tevet ships ~1K unlabeled sets per task (~3K total) plus a 200-set labeled eval subset per task (~600 total).
- **conTest (\S6.4):** "For each task, we collected 200 sets of 5 responses each (100 sets per class)." → nominally 600 sets across 3 tasks; in practice some tasks ship >200 rows (see below).
- **McDiv:** "6K {c, S_c} pairs, (2K for each storyGen, respGen and promptGen)."
- **McDivnuggets:** subset of McDiv, ~3K sets (1K/task) per the CSV layout and the data card.

## The data on disk

Counted with `wc -l` minus header. Each row = one {context, set-of-responses} pair.

| Dataset / split | prompt_gen | resp_gen | story_gen | **Sum** |
|---|---:|---:|---:|---:|
| decTest 1000 (no_hds) | 1000 | 994 | 985 | **2,979** |
| decTest 200 (with_hds) | 202 | 202 | 202 | **606** |
| conTest 200 (with_hds) | 200 | 220 | 250 | **670** |
| McDiv full (no_hds) | 2000 | 2002 | 2000 | **6,002** |
| McDiv_nuggets 1K (no_hds) | 1008 | 1049 | 1012 | **3,069** |
| McDiv_nuggets 200 (with_hds) | 200 | 200 | 200 | **600** |

Files under `/home/cs29824/matthew/icl-diversity/diversity-eval/data/raw/{decTest,conTest,McDiv,McDiv_nuggets}/`. Our pipeline (`scripts/compute_icl_metrics_for_tevet.py`, loaded via dataset keys in `scripts/analyze_c_ainf.py` lines 112-140) uses the `dec_test_1000_no_hds_*` split for DecTest (Table 6 caption confirms: "1000 samples, no_hds").

## Reconciliation

Four of the five numbers in our paper match the data-on-disk perfectly (not just consistent with Tevet's paper — identical to the CSV row counts):
- **6K McDiv** ✓ (6,002 rows)
- **3K McDiv_nuggets** ✓ (3,069 rows)
- **670 conTest** ✓ (670 rows — exact; this is three tasks' CSVs summed, where Tevet's "200 sets per task" nominal count expanded to 200/220/250 in the released CSVs)
- **3.6K decTest** — **WRONG.** Correct on-disk total for the split we evaluate on (`dec_test_1000_no_hds`) is **2,979 rows**, rounded ~3K. The "3.6K" appears to come from adding the `1000_no_hds` split (2,979) to the `200_with_hds` labeled subset (606) = 3,585, but those subsets **overlap** (the 200 labeled sets are a subset of the 1000). We do not evaluate on both; Table 6 is computed on the 1000-split only.

## Recommendation

Fix line 509 in `in_context_diversity_metric.tex`. Change "3.6K decTest sets" to "3K decTest sets" (matches the 1000-split per-task × 3 tasks ≈ 3K we actually evaluate on, and matches Tevet's "1K sets per task" prose).

Optionally, note in the same sentence that "670 conTest sets" reflects the released CSV row counts (Tevet's prose says 200 per task; the released CSVs contain 200/220/250). Both framings are defensible but the prose currently cites \S6.3-6.4, which states the smaller nominal numbers.

No code or pipeline bug — only a paper prose error. The analysis tables (`results/tables/dectest_rho.tex`, `contest_rho_oca.tex`) are correctly generated and their captions (`1000 samples`, `ConTest (200, with_hds)`) are accurate.
