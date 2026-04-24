# Open Research Questions

Open research threads that do not belong in the paper body. Things that belong in the paper's Future Work section are there; things that are diagnostic / anomaly-investigation / longer-term-exploration live here.

## The k=5 jump

**Observation.** The mean $a_k$ curve shows a pronounced jump at $k=5$ in several settings, particularly on low-diversity response sets that were labeled high-diversity by the benchmark (i.e., the confounded cases — see Appendix "Dataset Construction Confounds in McDiv").

**Why it matters.** The jump is disproportionately visible in cases where the metric and the human label disagree. If it is a property of the response distribution, it is a signal worth exploiting. If it is a property of $\theta$'s in-context learning behavior — e.g., a context-length / positional artifact — it is a confound we want to understand and ideally neutralize.

**Hypotheses to test.**
1. **Data-side:** does the jump at $k=5$ correspond to a structural feature of the response sets (e.g., McDiv sets contain exactly 5 responses and workers produced them in a specific order)? Check whether reshuffling the response order materially changes the jump.
2. **Model-side:** does the jump persist in other base models and other tokenizers (GPT-2, Qwen2.5-3B, Qwen3-30B-A3B-Base) on the same response sets? Cross-model check was done at a high level but not specifically focused on whether the $k=5$ jump persists or shifts.
3. **Context-length:** is $k=5$ the point where the context-plus-prompt hits a boundary in $\theta$'s attention (e.g., sliding-window, RoPE-scaling, or token-count threshold)? Instrument the mean context length at each $k$ and check for a corresponding discontinuity.

**Status:** observation only. No experimental investigation yet.

## Anything else that comes up

Future open-question entries go here — one H2 per question, with the same Observation / Why it matters / Hypotheses / Status structure.
