#!/usr/bin/env python3
"""Forensic audit: response length vs D trend in OLMo-2-7B RLHF experiment.

Answers: can the monotone D drop across base -> SFT -> DPO -> Instruct be
explained by response-length shifts across stages?

Run from repo root:
    uv run python scripts/audit_response_lengths.py

Reads:
    results/rlhf_experiment/generations/{base,sft,dpo,instruct}_{alpacaeval,nbcurated}.jsonl

Reports per-(stage, prompt_set):
    - Mean / median response byte length (UTF-8)
    - Mean / median response token length (cl100k_base, as a stable proxy)
    - Fraction of responses hitting the 100-token cap (finish_reason == "length")
    - Coefficient of variation of byte length (std / mean)

Then flags whether any length metric shows a stage-monotone trend.
"""
from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "rlhf_experiment" / "generations"
STAGE_ORDER = ["base", "sft", "dpo", "instruct"]
PROMPT_SETS = ["alpacaeval", "nbcurated"]


def main() -> None:
    enc = tiktoken.get_encoding("cl100k_base")

    records: list[dict] = []
    for f in sorted(glob.glob(str(DATA_DIR / "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["stage"], r["prompt_set"])].append(r)

    print(
        f"{'Stage':<10} {'PromptSet':<14} {'N':>5} | "
        f"{'MeanByte':>9} {'MedByte':>8} | "
        f"{'MeanTok':>8} {'MedTok':>7} | "
        f"{'FracLen':>8} | {'CV_byte':>8}"
    )
    print("-" * 100)

    results: dict[tuple[str, str], dict] = {}
    for stage in STAGE_ORDER:
        for ps in PROMPT_SETS:
            recs = groups.get((stage, ps), [])
            if not recs:
                continue
            byte_lens = [len(r["response"].encode("utf-8")) for r in recs]
            tok_lens = [len(enc.encode(r["response"])) for r in recs]
            frac_len = sum(1 for r in recs if r["finish_reason"] == "length") / len(recs)
            mean_b = statistics.mean(byte_lens)
            cv_b = statistics.stdev(byte_lens) / mean_b if mean_b else float("nan")
            results[(stage, ps)] = dict(
                n=len(recs),
                mean_byte=mean_b,
                med_byte=statistics.median(byte_lens),
                mean_tok=statistics.mean(tok_lens),
                med_tok=statistics.median(tok_lens),
                frac_length=frac_len,
                cv_byte=cv_b,
            )
            v = results[(stage, ps)]
            print(
                f"{stage:<10} {ps:<14} {v['n']:>5} | "
                f"{v['mean_byte']:>9.1f} {v['med_byte']:>8.1f} | "
                f"{v['mean_tok']:>8.1f} {v['med_tok']:>7.1f} | "
                f"{v['frac_length']:>8.3f} | {v['cv_byte']:>8.3f}"
            )

    for ps in PROMPT_SETS:
        print(f"\n--- Monotone trend check ({ps}) ---")
        for metric in ("mean_byte", "frac_length", "cv_byte"):
            vals = [results[(s, ps)][metric] for s in STAGE_ORDER if (s, ps) in results]
            diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            mono_dec = all(d < 0 for d in diffs)
            mono_inc = all(d > 0 for d in diffs)
            print(
                f"  {metric}: {[f'{v:.2f}' for v in vals]}  "
                f"monotone_dec={mono_dec}  monotone_inc={mono_inc}"
            )


if __name__ == "__main__":
    main()
