# POS patterns: structural redundancy with zero lexical overlap

**Question.** Follow-up to `reports/TEMPLATE_VS_SENTBERT.md`. There, repeated
frames shared literal function words, so averaged distinct-n also detected
the redundancy. Can we construct redundancy that beats SentBERT *and*
distinct-n at once: sentences where every word is freshly sampled (nothing
for n-gram overlap or embedding similarity to see) but all share one
part-of-speech pattern, the canonical
"{Noun} {verb} {preposition} {noun} {preposition} {noun}." e.g.
"Pelicans coughed beyond trolls without senators."?

**Answer: yes, with the right operationalization.** The first attempt (the
pattern-count sweep, kept below for the record) was the wrong experiment: it
re-imports the paper's mode-count logic, and its distinct-n correlation
turned out to be a class-composition confound. The corrected experiment is
single-set detection: one response set whose sentences all share the
canonical pattern, against a composition-matched control whose sentences
have no consistent word order. There, Decan's progressive surprise separates
the conditions with the correct sign and approximately the ground-truth
magnitude (a_n gap -0.174 ± 0.053 bits/byte against a true
generative-entropy gap of about -0.12), while SentBERT shows no significant
difference and distinct-n differs only through a capitalization artifact
that vanishes on lowercasing. Two honest caveats, detailed below: the scalar
D = C x a_n does not separate the conditions (the scrambled control's higher
diversity is offset by its lower coherence C), and the in-context drop ratio
is the same in both conditions. The detection signal is the a_n level, not D
or the drop ratio.

## Single-pattern detection: canonical vs scrambled control

The corrected operationalization: can Decan tell, from the response set
alone, that all sentences share one structure?

- **Conditions** (40 responses per set, 20 independent draws each,
  Qwen/Qwen2.5-3B bfloat16, same conventions as the sweep):
  - `canonical`: every sentence is pattern 0, N Vi P N P N
    ("Pelicans coughed beyond trolls without senators.").
  - `scrambled`: each sentence freshly samples the same class multiset
    (3 plural nouns, 1 intransitive past verb, 2 prepositions), then
    shuffles its own word order
    (`generate_scrambled_canonical_responses` in
    `src/icl_diversity/pos_pattern_scenarios.py`). Same word lists, same
    byte statistics (mean bytes/response 48.9 ± 0.5 vs 49.2 ± 0.7), no
    consistent order.
- **Ground truth.** The two generators use identical word choices and
  differ only in order, so their entropies differ by exactly
  log2(6!/(3! 1! 2!)) = log2(60) = 5.91 bits per sentence, i.e.
  5.91 / 49.2 = 0.12 bits/byte. The scrambled set is genuinely the more
  diverse one, by a known amount.
- **Scripts**: `scripts/run_pos_pattern_vs_baselines.py` with
  `--pattern-counts` (no values) `--include-scrambled`; raw runs in
  `results/pos_pattern/qwen2.5-3b_scrambled_control.json`.

All numbers below are from
`figures/pos_pattern/qwen2.5-3b_scrambled_control/summary.txt` (per-draw
means ± SD; diff = canonical - scrambled with Welch two-sided t-test) and
`distinctn_case_check.txt` in the same directory.

| metric | canonical | scrambled | diff ± SE | p |
|---|---|---|---|---|
| a_n (bits/byte) | 1.295 ± 0.187 | 1.469 ± 0.148 | -0.174 ± 0.053 | 0.0024 |
| a_1 (bits/byte) | 2.192 ± 0.240 | 2.444 ± 0.184 | -0.252 ± 0.068 | 0.00069 |
| drop ratio a_n/a_1 | 0.597 ± 0.104 | 0.604 ± 0.073 | -0.007 ± 0.028 | 0.81 |
| coherence C | 0.2211 ± 0.0043 | 0.1905 ± 0.0047 | +0.0305 ± 0.0014 | 1.1e-22 |
| D = C x a_n | 0.2862 ± 0.0416 | 0.2800 ± 0.0298 | +0.0062 ± 0.011 | 0.59 |
| SentBERT mean cosine | 0.2206 ± 0.0144 | 0.2271 ± 0.0106 | -0.0065 ± 0.004 | 0.11 |
| avg distinct-n | 0.9384 ± 0.0041 | 0.9445 ± 0.0036 | -0.0061 ± 0.0012 | 1.4e-05 |

Reading it:

1. **a_n detects the structure and approximately recovers the true entropy
   gap.** The canonical curve ends 0.174 ± 0.053 bits/byte below the
   scrambled one (p = 0.0024), within one SE of the ground-truth gap of
   0.12. The a_k overlay
   (`figures/pos_pattern/qwen2.5-3b_scrambled_control/fig2_ak_curves.png`)
   shows both curves falling steeply over the first few responses and
   plateauing with a persistent gap.
2. **In-context learning corrects the prior toward the truth.** At k = 1
   the gap is -0.252 ± 0.068: the base model's prior penalizes the
   ungrammatical order by more than the true entropy difference. By k = 40
   conditioning has shrunk the measured gap to -0.174, approaching the true
   -0.12 from above.
3. **SentBERT sees nothing** (p = 0.11), as predicted for a mean-pooling
   (approximately bag-of-words) encoder; its point estimate even runs the
   wrong way (scrambled slightly more mutually similar).
4. **distinct-n's difference is a capitalization artifact, not structure.**
   It is statistically detectable (p = 1.4e-05) but tiny (0.006 on a ~0.94
   near-ceiling value), and it vanishes entirely on lowercased text
   (diff -0.0002, p = 0.88; `distinctn_case_check.txt`, generated by
   `scripts/check_scrambled_distinctn_capitalization.py`). Mechanism: the
   canonical pattern always capitalizes a noun, while scrambling spreads
   sentence-initial capitalization across word classes, case-splitting more
   word types under the case-sensitive Tevet tokenization.
5. **Honest caveats.** (i) The drop ratio does not separate (0.597 vs
   0.604, p = 0.81). The scrambled control also carries in-context
   learnable redundancy (fixed class composition, restricted word lists,
   fixed six-words-plus-period format, and the model unlearning its grammar
   prior), so both curves fall by ~40%. Our advance prediction that
   canonical would drop proportionally further was wrong; the signal is the
   a_n level, and comparing levels across conditions is licensed here by
   the matched byte statistics. (ii) The scalar D = C x a_n does not
   separate (p = 0.59): C is higher for the grammatical canonical set
   (+0.0305, p = 1.1e-22), which cancels its lower a_n. That is D working
   as designed (it trades diversity against coherence, and the word salad
   is genuinely less coherent), but it means structure detection per se
   should be read from the a_k curve and a_n, with C reported alongside.

## The pattern-count sweep (first attempt, superseded)

The original attempt varied the number of patterns m, mirroring the paper's
mode-count experiments. Its mixed result is kept for the record; the
class-composition confound it uncovered motivated the matched control above.

### Design

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

### Results

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

### Reading the sweep result

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

### The fix, as run

The distinct-n confound is class composition, not structure, so the fix is a
composition-matched design in which word sampling depends only on class
counts. Rather than the ordering sweep originally proposed here (~6
grammatical permutations of the canonical multiset), we ran the simplest
two-condition form: canonical order vs no consistent order. That is the
"Single-pattern detection" experiment above, and it landed: distinct-n is
neutralized up to a capitalization artifact (checked by lowercasing), and
the remaining a_n gap tracks the ground-truth entropy difference.

## Reproduce

```bash
# Single-pattern detection (headline): canonical vs scrambled control
uv run python scripts/run_pos_pattern_vs_baselines.py \
    --base-model Qwen/Qwen2.5-3B --device cuda:1 --sentbert-device cuda:1 \
    --torch-dtype bfloat16 --batch-size 8 --n-draws 20 \
    --pattern-counts --include-scrambled \
    --output results/pos_pattern/qwen2.5-3b_scrambled_control.json
uv run python scripts/plot_template_vs_sentbert.py \
    --input results/pos_pattern/qwen2.5-3b_scrambled_control.json \
    --output-dir figures/pos_pattern/qwen2.5-3b_scrambled_control
uv run python scripts/check_scrambled_distinctn_capitalization.py

# Pattern-count sweep (first attempt)
uv run python scripts/run_pos_pattern_vs_baselines.py \
    --base-model Qwen/Qwen2.5-3B --device cuda:1 --sentbert-device cuda:1 \
    --torch-dtype bfloat16 --batch-size 8 --n-draws 20 \
    --output results/pos_pattern/qwen2.5-3b.json
uv run python scripts/plot_template_vs_sentbert.py \
    --input results/pos_pattern/qwen2.5-3b.json \
    --output-dir figures/pos_pattern/qwen2.5-3b

uv run pytest tests/test_pos_pattern_scenarios.py
```
