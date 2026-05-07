# Citation System

The paper's bibliography is machine-generated. You do not hand-type author lists, titles, venues, or years — ever. This is a discipline, not a preference; an entire literature has documented LLM-generated papers fabricating 14–95 % of their citations, and we hit four fabricated author lists in a single verification pass on this paper. The system below exists to make that class of error impossible going forward.

## Workflow

1. Add or change a reference in `paper/refs_ids.toml`. That is the only file you edit. Each entry specifies a citation key and a stable identifier (arXiv ID, DOI, or ACL Anthology ID), plus a `claim = "..."` describing what the paper actually says.
2. Regenerate the BibTeX:
   ```
   uv run python scripts/build_bib.py
   ```
   The script hits arXiv / Crossref / ACL Anthology, writes `paper/refs.bib` atomically, and fails loudly if any identifier is unresolvable or any required field is missing. The output is committed so the LaTeX build stays offline.
3. Lint the `\cite{…}` keys:
   ```
   uv run python scripts/verify_cites.py
   ```
   Passes only when every `\cite{}` in the .tex matches a `@type{key,…}` in `refs.bib` and every entry in `refs.bib` is actually cited. Runs offline; suitable for a pre-PR check.
4. Build the PDF as usual:
   ```
   cd paper && latexmk -pdf
   ```

## Files

| Path | Role |
|---|---|
| `paper/refs_ids.toml` | Source of truth. Each `[[cite]]` is an identifier + claim + optional manual body. |
| `scripts/build_bib.py` | Resolves identifiers → writes `paper/refs.bib`. Atomic; fails loudly on unresolved / malformed entries. |
| `scripts/verify_cites.py` | Offline linter: checks `\cite{}` ↔ `refs.bib` correspondence. |
| `paper/refs.bib` | **Generated. Do not hand-edit.** Every entry has a `% source:` comment naming the authoritative identifier. |

## Identifier precedence

When a `[[cite]]` entry specifies more than one identifier, the resolver picks the first that applies in this order:

1. `acl = "<id>"` — ACL Anthology. Canonical for anything in ACL / EMNLP / NAACL / EACL / COLING / TACL. Returns the official `@inproceedings` / `@article` entry.
2. `doi = "<id>"` — Crossref content negotiation. Returns the publisher's canonical BibTeX.
3. `arxiv = "<id>"` — arXiv API. XML parsed into an `@misc` entry.
4. `manual = true` with an `entry = """..."""` body — hand-written BibTeX for papers without a stable digital identifier (e.g. OpenAI tech reports). Use sparingly; each manual entry reintroduces the hallucination risk.

The purpose of the precedence is to prefer *published* metadata (conference / journal) over preprint metadata when both exist.

## Design principles

- **One human-edited file.** Humans and Claude only touch `refs_ids.toml`. All other citation files are derived.
- **Fail loudly.** A missing field, unreachable identifier, or `\cite{}` with no matching entry exits non-zero. The previous `refs.bib` stays intact if a run fails partway through (atomic temp-file rename).
- **Annotate the output.** Every entry in `refs.bib` is prefixed with `% source:` and `% claim:` comments so a reader can see the authoritative identifier and the specific claim without opening `refs_ids.toml`.
- **Commit the generated .bib.** LaTeX builds don't need network access; only `build_bib.py` does. Reviewers diff `refs.bib` to see the effect of any identifier change.
- **Citation keys must match the first author.** Keys are not opaque IDs — they appear in `.tex` source, in `grep`, and in reviewers' mental models. A key like `lam2025noveltybench` on a paper whose first author is Yiming Zhang is a hallucinated attribution preserved in amber, even if natbib renders the PDF correctly from the `author=` field. Eliminating hallucinations means eliminating them in keys too. If a canonical community key exists (ACL Anthology format, DBLP-style), prefer it; otherwise construct `<firstauthor><year><shorttitleword>` from the real metadata. When you find a key that encodes a wrong attribution, rename it — both in `refs_ids.toml` and every `\cite{}` site.

## Limitations

This system prevents three classes of error:

1. Fabricated or misattributed author lists, titles, venues, years.
2. Drift when a preprint gets published (the `acl`/`doi` identifier starts resolving to the published form).
3. `\cite{}` keys that no longer match any entry in the bib.

It does **not** prevent:

1. **Miscited claims.** If the paper says "X showed Y" and X's paper does not in fact show Y, fetching X's canonical BibTeX does not help. This is a prose-level error. The `claim = "..."` field in `refs_ids.toml` exists to make these claims easy to audit later (either by a human re-read or an LLM span-check).
2. **Factual errors in the prose** next to a citation (e.g. a misquoted parameter count). Those live in the .tex, not the bib.
3. **Picking the wrong paper to cite.** If the right paper for a given sentence isn't in `refs_ids.toml`, the tool cannot suggest one.

The `claim = "..."` fields are the hook for a future span-verification pass: given each claim plus the cited paper's abstract, an LLM can flag claims that aren't supported. Out of scope for now.

## Background and sources

This system is a direct application of published best practice for LLM-assisted paper writing. The core idea — identifier-first authoring with programmatic resolution — has been independently proposed by several tool-lines:

- [Rebiber](https://github.com/yuchenlin/rebiber) (Bill Yuchen Lin, 2021). Normalizes an existing .bib against DBLP / ACL Anthology. Canonical for ACL-community papers. Our `build_bib.py` incorporates the same idea but starts from a list of identifiers rather than an existing .bib, to avoid the case where the hallucinated author list is kept because the hallucinated title happens to match a real paper.
- [reffix](https://github.com/kasnerz/reffix) (Kasner). Similar in spirit via the DBLP API.
- [arxiv2bib](https://github.com/nathangrigg/arxiv2bib), [doi2bib3](https://github.com/archisman-panigrahi/doi2bib3), [doi2bibtex](https://github.com/timothygebhard/doi2bibtex). ID → BibTeX CLIs. Our `build_bib.py` inlines the arXiv and Crossref calls; the recipe comes from these tools.

Recent literature quantifying the problem this tool addresses:

- [CheckIfExist: Detecting Citation Hallucinations in the Era of AI-Generated Content](https://arxiv.org/html/2602.15871v1) (2026). Reports fabrication rates of 14.23 %–94.93 % across LLMs and domains, and introduces a multi-source cross-check (Crossref + Semantic Scholar + OpenAlex) for flagging fabricated entries. The "multi-source confirmation" idea is what we'd need to extend `build_bib.py` with if we wanted to harden it against unknowingly using an identifier that points at a retracted / duplicate-of-something-else paper.
- [GhostCite: A Large-Scale Analysis of Citation Validity in the Age of Large Language Models](https://arxiv.org/html/2602.06718) (2026). Audits 2.2 M citations across 56 k papers at top venues; finds invalid-citation rate ~1.07 % with an 80.9 % increase in 2025 alone. Documents the "verification gap": 41.5 % of researchers copy-paste BibTeX without checking.
- [BibTeX Citation Hallucinations in Scientific Publishing Agents: Evaluation and Mitigation](https://arxiv.org/html/2604.03159) (2026). Empirical evaluation of mitigations; motivates the ID-first architecture our tool implements.
- [CiteAudit, BibAgent (see CheckIfExist for survey)]. Systems that ground citations in authoritative databases.

Useful APIs:

- [arXiv API basics](https://info.arxiv.org/help/api/basics.html) — what `build_bib.py` uses to resolve `arxiv:<id>`.
- [Crossref REST API](https://api.crossref.org/works/{doi}/transform/application/x-bibtex) — content-negotiation endpoint that returns BibTeX for a DOI directly. `build_bib.py` uses the equivalent `https://doi.org/<id>` endpoint with `Accept: application/x-bibtex`.
- [ACL Anthology](https://aclanthology.org/) — every ACL paper has a `https://aclanthology.org/<id>.bib` URL. No API, just a GET.

## Prior incident

A pre-tool audit found that 5 of 12 citations had high-severity errors (4 fabricated author lists, 1 unsupported claim), 3 had medium-severity errors (numerical / construction / framing), and 2 had minor errors. Every fabricated entry pointed to a real paper — the identifier would have been correct if we had recorded one. The fabrications were all in the author list and, for three entries, the title.

The system above is the remediation.
