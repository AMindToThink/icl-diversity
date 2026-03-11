# BPE Boundary Merge Investigation

Investigates a tokenizer-dependent artifact in single-pass a_k curve computation: some tokenizers (e.g., Qwen2.5-3B) merge response-terminal punctuation with the separator into a single BPE token (e.g., `'.'` + `'\n\n'` → `'.\n\n'`), causing the model to predict a different token than in multi-pass computation.

**Conclusion**: The effect is real but small (~0.001 bits/byte, ~6% of curve decline). It does not affect curve shape. Documented as a paper caveat.

## Scripts

| Script | What it does | Requirements |
|--------|-------------|--------------|
| `01_demonstrate_merge.py` | Shows the BPE merge at response boundaries | Qwen2.5-3B + GPT-2 tokenizers |
| `02_boundary_comparison.py` | Compares old (progressive) vs new (offset-mapping) boundary detection | Qwen2.5-3B + GPT-2 tokenizers |
| `03_quantify_distortion.py` | Measures single-pass vs multi-pass distortion | Qwen2.5-3B on GPU (falls back to GPT-2 on CPU) |
| `04_separator_logprob.py` | Measures the separator overhead in merged tokens | Qwen2.5-3B on GPU |

## Running

```bash
uv run python investigations/bpe_boundary_merge/01_demonstrate_merge.py
uv run python investigations/bpe_boundary_merge/02_boundary_comparison.py
uv run python investigations/bpe_boundary_merge/03_quantify_distortion.py
uv run python investigations/bpe_boundary_merge/04_separator_logprob.py
```

See [REPORT.md](REPORT.md) for full findings.
