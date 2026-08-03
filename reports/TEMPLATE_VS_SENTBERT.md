# Template sentences: ICL diversity vs SentBERT

**Question.** Can the ICL diversity metric (the paper's primary scalar
D = C x a_n, `diversity_score_D_C_an`) detect a kind of redundancy that
embedding-similarity metrics such as Tevet & Berant's SentBERT baseline
cannot: a generator that always emits the same syntactic template
("The {adjective} {noun} {verb} the {adjective2} {noun2}.") while filling the
slots with semantically unrelated words?

**Answer.** Yes, with one honest caveat. Holding vocabulary (and therefore
semantic scatter) fixed while varying only the number of syntactic frames m,
D = C x a_n increases with m and the a_k curve shows the base model learning
the repeated frame within about five in-context examples. SentBERT places
every template condition near the top of its diversity range regardless of m;
its mean pairwise cosine barely distinguishes 1 frame from 20 (0.32 vs 0.21)
while genuine semantic redundancy (paraphrases) reads 0.81. The caveat: the
absolute D of a fixed-template set does not fall to paraphrase level, because
each sentence still carries real, irreducible lexical entropy (five fresh
random words), and D correctly keeps counting that part.

## Design

- **Data** (`src/icl_diversity/template_scenarios.py`): 20 structurally
  distinct sentence frames (declarative, question, passive, cleft,
  conditional, and so on), each filled from shared word lists (206
  adjectives, 203 nouns, 150 transitive verb triples; counts via
  `python -c "from icl_diversity.template_scenarios import *; ..."`).
  Frame 0 is the
  canonical "The {adj} {noun} {verb} the {adj2} {noun2}." All conditions
  sample words from the same lists, so lexical/semantic scatter is matched
  across conditions and only structural variety changes.
- **Conditions**, 20 responses per set:
  - `canonical`: every response uses frame 0.
  - `frames_m` for m in {1, 2, 5, 10, 20}: responses drawn evenly from m
    randomly chosen frames (fresh subset per draw).
  - `paraphrase`: 20 hand-written rewordings of one meaning; the agreement
    anchor where both metrics should read low.
- **Metrics**: `diversity_score_D_C_an` (and full a_k curves) from
  `compute_icl_diversity_metrics` (n_permutations=1 per draw, order
  shuffled per draw, statistical power from independent draws, as in
  `scripts/run_mode_count_experiment.py`); SentBERT diversity = negated mean
  pairwise cosine of `bert-large-nli-stsb-mean-tokens` embeddings and
  averaged distinct-n, both replicating diversity-eval conventions
  (`src/icl_diversity/baseline_metrics.py`).
- **Base models**: gpt2 (50 draws/condition) and Qwen/Qwen2.5-3B, bfloat16
  (20 draws/condition).
- **Scripts**: `scripts/run_template_vs_sentbert.py` (compute),
  `scripts/plot_template_vs_sentbert.py` (figures + summary). Raw runs in
  `results/template_vs_sentbert/{gpt2,qwen2.5-3b}.json`.

## Results

All numbers below are copied from the script-generated summaries
`figures/template_vs_sentbert/gpt2/summary.txt` and
`figures/template_vs_sentbert/qwen2.5-3b/summary.txt`; regenerate with the
commands at the bottom.

### Qwen2.5-3B (20 draws per condition, mean ± SD)

| condition  | D = C x a_n | SentBERT mean cosine | avg distinct-n |
|------------|-------------|----------------------|----------------|
| canonical  | 0.3534 ± 0.0464 | 0.2322 ± 0.0173 | 0.9266 ± 0.0049 |
| frames_1   | 0.2893 ± 0.0448 | 0.3207 ± 0.0681 | 0.8207 ± 0.0329 |
| frames_20  | 0.3940 ± 0.0414 | 0.2119 ± 0.0116 | 0.9406 ± 0.0034 |
| paraphrase | 0.1889 ± 0.0550 | 0.8135 ± 0.0000 | 0.7040 ± 0.0000 |

Normalized position on the paraphrase -> frames_20 scale (0 = as redundant as
paraphrases, 1 = as diverse as the 20-frame control):

| metric | canonical | frames_1 |
|--------|-----------|----------|
| D = C x a_n | +0.80 | +0.49 |
| SentBERT diversity | +0.97 | +0.82 |
| avg distinct-n | +0.94 | +0.49 |

In-context drop ratio a_n / a_1 (final conditional surprise as a fraction of
the first response's; lower = more structure learned in-context):
frames_1 = 0.546 ± 0.088, frames_20 = 0.739 ± 0.097,
paraphrase = 0.404 ± 0.101, canonical = 0.689 ± 0.101.

Spearman rho between m and metric across the 100 sweep runs:
D = +0.601 (p = 3.9e-11), a_n per byte = +0.673, SentBERT diversity = +0.787,
distinct-n = +0.894.

### GPT-2 (50 draws per condition)

Same qualitative picture, slightly weaker separation (GPT-2 is a weaker
in-context learner): D places frames_1 at +0.44 of the paraphrase -> frames_20
scale vs SentBERT's +0.78; drop ratios frames_1 = 0.543 ± 0.081 vs
frames_20 = 0.755 ± 0.105. Full table in
`figures/template_vs_sentbert/gpt2/summary.txt`.

## Reading the result

1. **The a_k curves are the demonstration**
   (`fig2_ak_curves.png` in each figures directory). Every template condition
   starts at the same unconditional surprise (the vocabulary control worked),
   then the single-frame curves plunge within about five responses while the
   20-frame curve stays high. The base model visibly learns the template
   in-context; that learned structure is exactly what D subtracts.
2. **SentBERT's failure is calibration, not ordering.** Its mean cosine does
   trend slightly with m (Spearman +0.787 within the sweep), but the entire
   template sweep lives in 0.21-0.32, a small corner near its
   maximum-diversity end, while real semantic redundancy sits at 0.81. An
   evaluator using SentBERT to detect template collapse would see "roughly
   82-97% of maximal diversity" and conclude the generator is fine. D places
   the same single-frame sets halfway down toward the paraphrase anchor.
3. **Caveat: the canonical frame has a high content floor.** Frame 0 is
   almost all content words, so most of its bytes stay unpredictable even
   after the frame is learned (drop ratio 0.689 vs 0.546 for a random single
   frame, whose boilerplate spans like ", nobody noticed" become fully
   predictable). Its absolute D (0.353) therefore sits closer to the
   20-frame control than the frames_1 average does. The clean comparison for
   the headline claim is frames_1 vs frames_20, where frame composition is
   matched in distribution.
4. **Distinct-n also detects the manipulation** (+0.49 normalized), which is
   expected: repeated frames are literal string overlap, the thing distinct-n
   measures. This experiment separates ICL diversity from embedding
   similarity, not from n-gram overlap; distinct-n's known failure mode is
   the opposite one (it over-rewards paraphrases that reword, compare its
   compressed 0.70-0.94 scale here).

## Reproduce

```bash
uv run python scripts/run_template_vs_sentbert.py --base-model gpt2 \
    --device cuda:0 --batch-size 16 --n-draws 50 \
    --output results/template_vs_sentbert/gpt2.json
uv run python scripts/run_template_vs_sentbert.py --base-model Qwen/Qwen2.5-3B \
    --device cuda:1 --sentbert-device cuda:1 --torch-dtype bfloat16 \
    --batch-size 8 --n-draws 20 \
    --output results/template_vs_sentbert/qwen2.5-3b.json
uv run python scripts/plot_template_vs_sentbert.py \
    --input results/template_vs_sentbert/gpt2.json \
    --output-dir figures/template_vs_sentbert/gpt2
uv run python scripts/plot_template_vs_sentbert.py \
    --input results/template_vs_sentbert/qwen2.5-3b.json \
    --output-dir figures/template_vs_sentbert/qwen2.5-3b
uv run pytest tests/test_template_scenarios.py tests/test_baseline_metrics.py
```
