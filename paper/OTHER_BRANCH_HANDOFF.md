# Hand-off from the tooling branch

I am building a script-generated citation system on a separate branch (analog to our `import-content` skill for tables): a `paper/refs_ids.toml` listing identifiers, `scripts/build_bib.py` that resolves each to canonical BibTeX via arXiv / Crossref / ACL APIs, and `scripts/verify_cites.py` that lints every `\cite{…}` against the generated `paper/refs.bib`. When it lands, the `\begin{thebibliography}` block in `in_context_diversity_metric.tex` will be replaced with `\bibliography{refs}`, and all author lists / titles / venues will be authoritative and re-fetchable.

## Don't hand-edit the bibliography on this branch

The tool will overwrite it. Anything you fix manually will be wasted work and risks merge conflicts. In particular:

- Don't retype author lists.
- Don't fix title typos.
- Don't adjust preprint-vs-conference entries.

All of that is the tool's job.

## Do fix these prose-level issues

The tool can't touch these. They live in the paper's running text, not the bib.

### 1. Section 8.1 — Qwen2.5-32B context window

Change `32K-token context window` → `128K-token context window`. The Qwen2.5 Technical Report (Yang et al. 2024, Table 1) lists 128K for 7B / 14B / 32B / 72B; 32K is the 3B / 1.5B / 0.5B value. See `paper/citation_verification_report.md` for the exact reference.

### 2. Section 11 — "Sampling diversity metrics" paragraph

The `zhang2024writingprompts` citation resolves to Zhang, Duckworth, Ippolito, Neelakantan (2020), "Trading Off Diversity and Quality in NLG" (arXiv:2004.10450). That paper does **not** use self-BLEU or n-gram overlap — it uses Shannon entropy of the model distribution (§2).

Either:
- **Swap the citation** to Holtzman et al. 2020, "The Curious Case of Neural Text Degeneration" (ICLR 2020, arXiv:1904.09751) — canonical for self-BLEU + distinct-n at decoding time. If you swap, add a new entry to `paper/refs_ids.toml` when the tooling branch lands. For now, leave a `% TODO: swap citation to Holtzman et al. 2020` comment in the .tex.
- **Or weaken the claim** to match what Zhang 2020 actually does (Shannon entropy of the sampled distribution).

### 3. Appendix C.1 Mechanism — McDiv_nuggets construction

The current description is wrong. Our paper says workers "were given a specific ending and asked to paraphrase it five times."

Actual procedure (Tevet & Berant §6.4, p. 332–333): workers first write 5 diverse responses, then **self-select** one of their own responses and paraphrase it 5 times.

Rewrite the mechanism paragraph accordingly. The confound story still holds — low-diversity sets end up concentrated on whichever ending the worker happened to find most paraphrasable — but the causal description needs correcting.

Optional softened wording that preserves the "specific/dramatic endings" framing:

> "The endings workers self-select to paraphrase tend to be specific or dramatic, so the resulting low-diversity sets concentrate on intrinsically surprising content."

### 4. Section 8.5 intro — "13,679 response sets"

This number doesn't match Tevet's CSVs (actual count ≈ 13,929). Either:
- (a) Replace with a script-derived number (following the "cite-the-source" rule from memory: have `scripts/count_tevet.py` produce the number and footnote the source), or
- (b) Replace with Tevet's round figures ("~6K McDiv sets, ~3K McDiv_nuggets, ~3.6K decTest, ~670 conTest").

## Merge order when we're both done

1. Tooling branch merges first: bibliography regenerated from scratch; `\bibliography{refs}` wired up; no prose touched.
2. This branch rebases and merges: prose fixes only; `.bib` entries untouched.

Since the two branches edit disjoint parts of the .tex (this branch: prose; tooling branch: only the `\begin{thebibliography}...\end{thebibliography}` block and the `\bibliographystyle`/`\bibliography` directives), conflicts should be minimal or zero.

## Verification after merge

```
uv run python scripts/verify_cites.py
```

Should print `OK` and exit 0. If any `\cite{key}` in the .tex lacks a matching entry in `refs.bib`, it will exit non-zero. Add that to any pre-PR checklist.

## Full error inventory

See `paper/citation_verification_report.md` (shared between branches).
