# POS patterns: structural redundancy with zero lexical overlap

**Question.** Follow-up to `reports/TEMPLATE_VS_SENTBERT.md`. There, repeated
frames shared literal function words, so averaged distinct-n also detected
the redundancy. Can we construct redundancy that beats SentBERT *and*
distinct-n at once: sentences where every word is freshly sampled (nothing
for n-gram overlap or embedding similarity to see) but all share one
part-of-speech pattern, the canonical
"{Noun} {verb} {preposition} {noun} {preposition} {noun}." e.g.
"Pelicans coughed beyond trolls without senators."?

**Answer: mixed, and worth being precise about.** The ICL metric does detect
pure POS-pattern redundancy: D = C x a_n rises significantly with the number
of patterns m (Spearman rho = +0.275, p = 0.0057), and a fixed-pattern set
shows a ~40% in-context drop in conditional surprise. But the effect is much
smaller than in the frames experiment, and the sweep does **not** cleanly
separate D from the baselines in rank terms: distinct-n correlates with m
*more* strongly (rho = +0.697), not through lexical overlap between
same-pattern sentences (there is none by construction) but through a
class-composition confound described below. The "beats both baselines"
demonstration therefore did not land in this form; see "What would fix it".

## Design

- **Data** (`src/icl_diversity/pos_pattern_scenarios.py`): 12 six-word POS
  patterns with zero fixed lexical tokens (a unit test enforces that outside
  `{Tag}` placeholders only spaces, commas, periods appear). Pattern 0 is
  the canonical N Vi P N P N. Word classes: 197 plural nouns, 106
  intransitive past verbs, 150 transitive past verbs, 35 prepositions, 60
  adverbs, 206 adjectives (counts: `len()` on the module lists). Every slot
  of every sentence is freshly sampled.
- **Conditions**, 40 responses per set, 20 independent draws each:
  `canonical` (pattern 0 only) and `patterns_m` for m in {1, 2, 4, 8, 12}.
  No paraphrase anchor (the claim is within-sweep; cross-metric anchoring
  was established in the first experiment).
- **Base model**: Qwen/Qwen2.5-3B, bfloat16. Metrics and conventions
  identical to the first experiment.
- **Scripts**: `scripts/run_pos_pattern_vs_baselines.py`,
  `scripts/plot_template_vs_sentbert.py` (shared analyzer). Raw runs in
  `results/pos_pattern/qwen2.5-3b.json`.

## Results

All numbers from `figures/pos_pattern/qwen2.5-3b/summary.txt`.

| condition   | D = C x a_n | SentBERT mean cosine | avg distinct-n | drop ratio a_n/a_1 |
|-------------|-------------|----------------------|----------------|--------------------|
| canonical   | 0.2862 ± 0.0416 | 0.2206 ± 0.0144 | 0.9384 ± 0.0041 | 0.597 ± 0.104 |
| patterns_1  | 0.2874 ± 0.0436 | 0.2394 ± 0.0172 | 0.9571 ± 0.0098 | 0.604 ± 0.117 |
| patterns_12 | 0.3229 ± 0.0454 | 0.2229 ± 0.0121 | 0.9744 ± 0.0031 | 0.664 ± 0.112 |

Spearman rho between m and metric (100 sweep runs):
D = +0.275 (p = 0.0057), a_n per byte = +0.263 (p = 0.0081),
SentBERT diversity = +0.290 (p = 0.0035), distinct-n = +0.697 (p = 7.6e-16).
Coherence C is flat across all conditions (0.217 to 0.221), confirming the
vocabulary control.

## Reading the result

1. **D detects pure POS-pattern redundancy, weakly.** The rise with m is
   real (p = 0.0057) and the mechanism is visible in
   `figures/pos_pattern/qwen2.5-3b/fig2_ak_curves.png`: all conditions start
   at ~2.2 bits/byte and the m=1 and canonical curves settle ~0.2 bits/byte
   below m=12. But knowing a sentence's POS pattern saves few bits: the
   open-class word identities, which dominate the byte count, stay
   unpredictable no matter how well the pattern is learned. Compare drop
   ratios: 0.604 vs 0.664 here (a 0.06 gap) against 0.546 vs 0.739 (a 0.19
   gap) in the frames experiment, where literal boilerplate became fully
   predictable.
2. **Distinct-n still correlates with m, for an artifactual reason.** The 12
   patterns differ in class composition (canonical has two preposition slots
   and three nouns; others spend slots on adjectives/adverbs). At m=1 one
   composition is repeated 40 times, so closed-class tokens (35
   prepositions) collide often; at m=12 compositions mix. The absolute range
   is tiny and near the ceiling (0.957 to 0.974, vs 0.70 to 0.94 across the
   first experiment's conditions), but per-draw variance is even tinier, so
   the rank correlation is strong. Note distinct-n rates canonical lowest
   (0.938) for this composition reason, not because it sees the structure.
3. **SentBERT's correlation (+0.290) is comparable to D's (+0.275)** with a
   minute absolute range (0.223 to 0.239 cosine). Neither metric's rank
   ordering separates from D's here.
4. **Where D is genuinely alone**: the absolute in-context diagnostic. For a
   single fixed-pattern response set, the a_k curve reveals that ~40% of
   per-byte surprise is learnable structure (drop ratio 0.60), while
   distinct-n (0.94, near max) and SentBERT (0.22 cosine, near max
   diversity) both read the same set as almost maximally diverse. The
   baselines have no analogue of this within-set signal; they only produce
   one number with no notion of "how much of this would a learner stop
   being surprised by".

## What would fix it (proposed v2)

The distinct-n confound is class composition, not structure. A
composition-matched design removes it provably: make every pattern a
word-order permutation of the canonical class multiset {N, N, N, Vi, P, P}
(canonical; fronted-PP "P N, N Vi P N"; subject-medial "N P N Vi P N";
double-fronted "P N P N, N Vi"; verb-final "N P N P N Vi"; locative
inversion "P N Vi N P N"). Word sampling then depends only on class counts,
which are identical across m, so distinct-n's distribution is the same at
every m by construction. Trade-offs: at most ~6 grammatical orderings
(smaller sweep range), and word-order-only contrasts may shrink D's signal
further; temperature sharpening (T < 1) and more draws are the available
amplifiers.

## Reproduce

```bash
uv run python scripts/run_pos_pattern_vs_baselines.py \
    --base-model Qwen/Qwen2.5-3B --device cuda:1 --sentbert-device cuda:1 \
    --torch-dtype bfloat16 --batch-size 8 --n-draws 20 \
    --output results/pos_pattern/qwen2.5-3b.json
uv run python scripts/plot_template_vs_sentbert.py \
    --input results/pos_pattern/qwen2.5-3b.json \
    --output-dir figures/pos_pattern/qwen2.5-3b
uv run pytest tests/test_pos_pattern_scenarios.py
```
