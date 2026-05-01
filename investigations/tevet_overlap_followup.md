# TODO: revisit remembered overlap in McDiv_nuggets scatter

**Status:** open, low priority. Vague recollection, looked into and not
reproduced. **If the original observation is real, the fault is on Tevet's
side, not in our code** — our pipeline treats every row as an independent
(prompt, response_set, label) datapoint via `sidecar_key`
(`scripts/analyze_c_ainf.py:55-60`), which is robust to anything Tevet did
to construct the released CSVs.

## What we remember

During an early audit of McDiv_nuggets, we recall seeing a **scatter plot** in
which red (high-diversity) and blue (low-diversity) points appeared to land on
top of each other in a way that suggested the same underlying datapoint was
labeled both ways — with each point representing a (prompt, responses) tuple,
not just a prompt. That observation drove us to write an appendix subsection
titled "Label Contamination" claiming "the same response set appears once
labeled as high diversity and once as low diversity."

## What we found when we re-checked (2026-04-30 / 2026-05-01)

The current Tevet-released CSV data does not support that claim:

- 27 sample_ids appear with both label values across the three with_hds CSVs
  (9 prompt_gen + 11 resp_gen + 7 story_gen).
- Every one of the 27 has the **same context** but **fully disjoint** five-
  response sets — zero shared responses between the high-div and low-div row,
  whether compared as ordered tuples or unordered sets.
- The Tevet-precomputed metric values (sentBERT, BERTsts, BERT-score, cosine,
  distinct-n) and our own per-byte E differ row-to-row, so the rows are also
  distinct in metric space and would not produce identical scatter markers.

The structural read of Tevet's data is that McDiv_nuggets is a sparsely-
labeled fragment of the underlying ConTest pair structure: for ~5% of prompts
in the nuggets subset, Tevet kept both pair members (one high-div response
set, one low-div response set) under the same `sample_id`. ConTest is fully
paired this way (every prompt has both); DecTest has zero conflicts (single
label per prompt). None of this is documented in Tevet's paper or release
notes; the schema has to be reverse-engineered from the CSV.

Reproduction of the disjoint-response check:

```
uv run python - <<'EOF'
import csv
from pathlib import Path
base = Path("results/tevet/qwen25_completion_v3/McDiv_nuggets")
RESP = ["resp_0","resp_1","resp_2","resp_3","resp_4"]
for p in sorted(base.glob("*_with_hds_*.csv")):
    rows = list(csv.DictReader(open(p)))
    by_id = {}
    for r in rows: by_id.setdefault(r["sample_id"], []).append(r)
    for sid, rs in by_id.items():
        if len({r["label_value"] for r in rs}) <= 1: continue
        a = frozenset(rs[0][c].strip() for c in RESP)
        b = frozenset(rs[1][c].strip() for c in RESP)
        assert not (a & b), (sid, a & b)
EOF
```

## Why this stays open

The original visual observation is real-as-recalled; we just cannot find the
scatter or reconstruct the conditions that produced it. **All viable
hypotheses for what we saw point at Tevet's release, not at our analysis
code:**

1. **Tevet shipped a different version of the data than the one currently
   checked in at `diversity-eval/data/with_metrics/`.** If an earlier release
   (or a different subset, task, or split that we did not check) had literal
   duplicate (prompt, responses) rows under conflicting labels, the scatter
   we remember would be a faithful rendering of a Tevet-side data bug. The
   current snapshot does not exhibit it, but we have no provenance trail
   confirming the snapshot is identical to what we originally pulled.
2. **A latent inconsistency in Tevet's labeling protocol** — e.g., HDS
   thresholds applied to the same response set in different annotation
   batches, producing rows that look distinct in the CSV (different IDs,
   different metric columns) but back-trace to the same underlying
   (prompt, responses) example in Tevet's source pipeline. We would not see
   this in the released CSVs as we checked them, only in Tevet's internal
   intermediate representation.
3. **The scatter we remember was a different visual** than what we have
   reconstructed from current scripts — e.g., a different x/y choice that
   genuinely places two distinct rows at coincident pixels. Distinct in
   metric space does not always mean visually distinct at the resolution of
   the plot. This one is on us, not Tevet, but only in the sense of "we
   misread our own plot," not "our analysis is wrong."

In none of these is our pipeline at fault: our headline analysis treats every
CSV row as its own (response_set, label) datapoint, which is the correct
posture whether or not Tevet's data has any of the issues above.

## What to do if revisited

- Try to find the original plot (git log, old `figures/` snapshots, slides,
  notebook outputs). The scatter we remember should be locatable as an
  artifact, not just a memory.
- If found, identify which two rows produced the overlapping markers and
  re-check whether they share `sample_id` only or also share responses
  against the **current** Tevet-released CSVs.
- If the result is "two distinct rows that visually coincided," close this
  TODO with a one-line note and delete this file.
- If it is "the underlying data has changed" or "Tevet shipped a version
  with literal duplicate rows somewhere we did not check," file it as a
  Tevet upstream issue. Our headline analysis would still be correct under
  the per-row independence posture; the upstream issue would be a finding
  about Tevet's release quality, not a correction to our results.

## Related code and paper state

- The deactivated `--dedup` flag in `scripts/compute_icl_metrics_for_tevet.py`
  was the corrective action we considered before re-checking the data. It is
  left in place but disabled (the flag now exits with an error) because, on
  the current released data, it would destroy Tevet's intended pair
  structure rather than fix anything.
- The "Label Contamination" subsection of `paper/sections/appC_mcdiv_confounds.tex`
  was removed when this note was created, since its empirical premise does
  not hold on the Tevet data we currently have.
