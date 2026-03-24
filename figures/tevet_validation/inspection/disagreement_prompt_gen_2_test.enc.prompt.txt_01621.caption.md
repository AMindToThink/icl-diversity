## "eto" token analysis

Context: `"Do I have"`, response: `"to repeat myself?"`. Bug 1 (no space) produces `"Do I haveto repeat myself?"`. BPE splits `"haveto"` as `" hav"` + `"eto"`.

The model assigns 42% probability (1.3 bits) to `"eto"` after `" hav"`. The merged word `"haveto"` would be rare in well-edited text — it essentially only occurs as a typo (missing space in `"have to"`). So the `" hav"` token acts as a surprisingly strong typo detector: the model has seen enough instances of `"haveto"` in messy internet training data to learn that `"eto"` follows with 42% confidence. The BPE tokenization `" hav"` + `"eto"` is itself an artifact of this common misspelling being frequent enough to influence the vocabulary.

Top predictions after `"1. Do I hav"`:

| Token | Probability | Bits |
|-------|------------|------|
| `eto` | 0.418 | 1.3 |
| `▁to` | 0.131 | 2.9 |
| `\n` | 0.045 | 4.5 |
| `et` | 0.032 | 5.0 |
| `▁e` | 0.025 | 5.3 |

Note: `"e"` as a single token gets essentially zero probability (20.5 bits) — the model strongly prefers completing `"have to"` in one step via `"eto"` rather than decomposing as `"e"` + `"to"`.

This token is masked (part of context, not response), so it doesn't affect the a_k curve. But it illustrates how Bug 1 creates garbled text that the model handles surprisingly well due to BPE statistics.
