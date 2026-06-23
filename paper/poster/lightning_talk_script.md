# 60-second lightning talk narration

Record this read over `icl_diversity_slide.pdf` (Zoom / screen capture), export as
`{ID}_video.mp4` (≤60s, one static slide, no animation). Target pace ~2.5 words/sec.
Word count ~140 → ~56s, leaving a small buffer.

Almost every sentence is lifted from the paper's abstract; spoken adaptations are
flagged at the bottom.

## Script

Measuring the diversity of creative outputs is central to evaluating post-training mode
collapse, comparing decoding strategies, and quantifying creative behavior in both AI and
human writing.

We propose a new approach to measuring diversity using in-context learning. Our metric,
D equals C times a-n, is a per-byte score read off the per-token log-probabilities of a
base model, in a single forward pass, with no embedding model, no reference corpus, and
no human labels.

The intuition: if responses are diverse, seeing one should not help the model predict
the next; if they collapse to a few modes, conditioning sharply reduces its surprise.

On Tevet and Berant's human-grounded benchmark, it lands close to a trained sentence-embedding
baseline. It is behind that baseline, but it uses no embeddings, no references, and no labels.
On the OLMo-2-7B post-training pipeline, it decreases monotonically across base, SFT, DPO, and
RLVR, detecting the type of diversity loss that creative-writing applications care about.

The same pipeline scores AI samples and human-written sets alike.

## Provenance / flags

- Sentence 1, 2, the OLMo sentence, and the last sentence are **verbatim from the abstract**
  (`paper/sections/abstract_workshop.tex`), with two spoken-form adaptations:
  - "D equals C times a-n" speaks the symbol `D_{Ca_n} = C \times a_n`.
  - "base, SFT, DPO, and RLVR" speaks the arrow chain `base -> SFT -> DPO -> RLVR`.
- The "intuition" sentence is condensed from `01_motivation_workshop.tex` (verbatim phrases,
  shortened for a 60s read). **Flagged as lightly adapted.**
- "OCA 0.846" and "SentBERT 0.897" trace to `results/tables/paper_macros.tex`
  (`\tevetMcDivPromptGenCxAnOCA`, `\tevetMcDivPromptGenSentBertOCA`).
