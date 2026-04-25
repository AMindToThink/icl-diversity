"""Produce a length-matched copy of the RLHF generations.

For each (prompt_set, prompt_id), find the minimum UTF-8 byte length across all
40 responses (4 stages x 10 samples) for that prompt and truncate every response
to exactly that length. UTF-8 safety: if the byte cut falls inside a multi-byte
character, back off to the largest valid prefix.

Inputs
------
results/rlhf_experiment/generations/{stage}_{set}.jsonl

Outputs
-------
results/rlhf_experiment/generations_length_matched/{stage}_{set}.jsonl
    Same fields as input plus `original_bytes` (int) for traceability.
results/rlhf_experiment/generations_length_matched/length_match_report.json
    Per-prompt L_prompt and which stage/sample_idx defined the min, plus
    distribution stats.

Sanity checks
-------------
* Every prompt has exactly 40 responses (4 stages, 10 samples each); script
  aborts if violated.
* Prompts with L_prompt < --min-bytes (default 50; ~10 tokens of English) are
  DROPPED rather than truncated to a meaningless fragment, and the dropped
  fraction is logged. Set --min-bytes 0 to keep everything.

Run
---
    uv run python scripts/rlhf_experiment/3a_truncate_length_matched.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "results" / "rlhf_experiment" / "generations"
DST_DIR = REPO_ROOT / "results" / "rlhf_experiment" / "generations_length_matched"

STAGES = ["base", "sft", "dpo", "instruct"]
PROMPT_SETS = ["alpacaeval", "nbcurated"]
SAMPLES_PER_STAGE = 10
EXPECTED_RESPONSES_PER_PROMPT = len(STAGES) * SAMPLES_PER_STAGE  # 40


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def truncate_utf8(text: str, n_bytes: int) -> str:
    """Return the longest valid UTF-8 prefix of `text` whose encoded length
    is <= n_bytes."""
    if n_bytes < 0:
        raise ValueError(f"n_bytes must be non-negative, got {n_bytes}")
    encoded = text.encode("utf-8")
    if len(encoded) <= n_bytes:
        return text
    # decode with errors="ignore" drops a partial trailing multibyte sequence.
    return encoded[:n_bytes].decode("utf-8", errors="ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--min-bytes",
        type=int,
        default=50,
        help=(
            "Floor on per-prompt L_prompt. Prompts whose 40-response common "
            "minimum falls below this are dropped from the length-matched set "
            "(the truncation would yield text fragments too short to score "
            "meaningfully). Set to 0 to keep everything."
        ),
    )
    ap.add_argument(
        "--src-dir",
        type=Path,
        default=SRC_DIR,
    )
    ap.add_argument(
        "--dst-dir",
        type=Path,
        default=DST_DIR,
    )
    args = ap.parse_args()

    # --- Load all records, group by (prompt_set, prompt_id) ---
    by_prompt: dict[tuple[str, str], list[dict]] = defaultdict(list)
    files_loaded = 0
    for stage in STAGES:
        for ps in PROMPT_SETS:
            path = args.src_dir / f"{stage}_{ps}.jsonl"
            rows = read_jsonl(path)
            files_loaded += 1
            for r in rows:
                if r["stage"] != stage or r["prompt_set"] != ps:
                    raise ValueError(
                        f"record stage/set mismatch in {path}: got "
                        f"{r['stage']}/{r['prompt_set']}"
                    )
                by_prompt[(ps, r["prompt_id"])].append(r)
    print(f"[truncate] loaded {files_loaded} files, "
          f"{sum(len(v) for v in by_prompt.values())} records, "
          f"{len(by_prompt)} prompts", flush=True)

    # --- Validate completeness ---
    for key, recs in by_prompt.items():
        if len(recs) != EXPECTED_RESPONSES_PER_PROMPT:
            raise ValueError(
                f"prompt {key} has {len(recs)} responses, expected "
                f"{EXPECTED_RESPONSES_PER_PROMPT}"
            )
        # Within a prompt, must have exactly 10 samples per stage
        per_stage_count: dict[str, int] = defaultdict(int)
        for r in recs:
            per_stage_count[r["stage"]] += 1
        for s in STAGES:
            if per_stage_count[s] != SAMPLES_PER_STAGE:
                raise ValueError(
                    f"prompt {key} has {per_stage_count[s]} samples for stage "
                    f"{s}, expected {SAMPLES_PER_STAGE}"
                )

    # --- Per-prompt minimum bytes ---
    L_prompts: dict[tuple[str, str], int] = {}
    min_origin: dict[tuple[str, str], dict] = {}
    for key, recs in by_prompt.items():
        byte_lens = [(len(r["response"].encode("utf-8")), r) for r in recs]
        L, origin = min(byte_lens, key=lambda t: t[0])
        L_prompts[key] = L
        min_origin[key] = {
            "stage": origin["stage"],
            "sample_idx": origin["sample_idx"],
        }

    # --- Sanity stats ---
    Ls_all = list(L_prompts.values())
    Ls_by_set: dict[str, list[int]] = defaultdict(list)
    for (ps, _pid), L in L_prompts.items():
        Ls_by_set[ps].append(L)

    print("\n[truncate] L_prompt (min UTF-8 bytes across the 40 responses) summary:")
    print(f"  overall: n={len(Ls_all)}, min={min(Ls_all)}, "
          f"p10={int(statistics.quantiles(Ls_all, n=10)[0])}, "
          f"median={int(statistics.median(Ls_all))}, "
          f"mean={statistics.mean(Ls_all):.1f}, "
          f"max={max(Ls_all)}")
    for ps in PROMPT_SETS:
        Ls = Ls_by_set[ps]
        print(f"  {ps:<12}: n={len(Ls)}, min={min(Ls)}, "
              f"p10={int(statistics.quantiles(Ls, n=10)[0])}, "
              f"median={int(statistics.median(Ls))}, "
              f"mean={statistics.mean(Ls):.1f}, "
              f"max={max(Ls)}")

    # Which stage drove the minimum, by prompt_set?
    print("\n[truncate] which stage drove L_prompt (min byte response):")
    for ps in PROMPT_SETS:
        stage_counts: dict[str, int] = defaultdict(int)
        for (ps2, _pid), origin in min_origin.items():
            if ps2 == ps:
                stage_counts[origin["stage"]] += 1
        total = sum(stage_counts.values())
        ordered = sorted(stage_counts.items(), key=lambda t: -t[1])
        line = ", ".join(f"{s}={c}/{total}" for s, c in ordered)
        print(f"  {ps:<12}: {line}")

    # --- Drop prompts below the floor (length-matching them would yield empty
    # or near-empty text — not a meaningful diversity comparison). ---
    dropped: dict[tuple[str, str], int] = {
        k: L for k, L in L_prompts.items() if L < args.min_bytes
    }
    if dropped:
        print(
            f"\n[truncate] DROPPING {len(dropped)} / {len(L_prompts)} prompts "
            f"with L_prompt < {args.min_bytes} bytes "
            f"(length-matching would compare empty/near-empty fragments).",
            flush=True,
        )
        per_set_drop: dict[str, int] = defaultdict(int)
        for (ps, _pid), _L in dropped.items():
            per_set_drop[ps] += 1
        for ps in PROMPT_SETS:
            kept = sum(1 for (ps2, _pid) in L_prompts if ps2 == ps) - per_set_drop[ps]
            total = sum(1 for (ps2, _pid) in L_prompts if ps2 == ps)
            print(f"  {ps:<12}: dropped {per_set_drop[ps]}/{total}, kept {kept}")
        # Show a few extreme cases so the log is self-explanatory
        print("  worst 10 (smallest L_prompt) being dropped:")
        for k, L in sorted(dropped.items(), key=lambda t: t[1])[:10]:
            print(f"    {k} -> {L} bytes (min from "
                  f"{min_origin[k]['stage']}#{min_origin[k]['sample_idx']})")
    kept_prompts: dict[tuple[str, str], int] = {
        k: L for k, L in L_prompts.items() if k not in dropped
    }
    if not kept_prompts:
        print("[truncate] ABORT: no prompts survive the floor", file=sys.stderr)
        sys.exit(2)

    # --- Truncate and write (only kept prompts) ---
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_truncated = 0  # responses where text changed
    n_skipped_dropped = 0
    for stage in STAGES:
        for ps in PROMPT_SETS:
            out_rows: list[dict] = []
            src_path = args.src_dir / f"{stage}_{ps}.jsonl"
            for r in read_jsonl(src_path):
                key = (ps, r["prompt_id"])
                if key not in kept_prompts:
                    n_skipped_dropped += 1
                    continue
                L = kept_prompts[key]
                original = r["response"]
                truncated = truncate_utf8(original, L)
                if truncated != original:
                    n_truncated += 1
                new_r = dict(r)
                new_r["response"] = truncated
                new_r["original_bytes"] = len(original.encode("utf-8"))
                new_r["truncation_bytes"] = L
                out_rows.append(new_r)
            out_path = args.dst_dir / f"{stage}_{ps}.jsonl"
            write_jsonl(out_path, out_rows)
            n_written += len(out_rows)
            print(f"[truncate] wrote {out_path} ({len(out_rows)} rows)")

    print(f"\n[truncate] total: wrote {n_written} records "
          f"(dropped {n_skipped_dropped} from below-floor prompts), "
          f"{n_truncated} truncated ({n_truncated / max(1, n_written):.1%})")

    # --- Report file ---
    Ls_kept = list(kept_prompts.values())
    Ls_kept_by_set: dict[str, list[int]] = defaultdict(list)
    for (ps, _pid), L in kept_prompts.items():
        Ls_kept_by_set[ps].append(L)
    report_path = args.dst_dir / "length_match_report.json"
    report = {
        "min_bytes_floor": args.min_bytes,
        "n_prompts_total": len(L_prompts),
        "n_prompts_kept": len(kept_prompts),
        "n_prompts_dropped": len(dropped),
        "n_records_written": n_written,
        "n_records_truncated": n_truncated,
        "L_prompt_stats_all": {
            "min": min(Ls_all),
            "median": int(statistics.median(Ls_all)),
            "mean": statistics.mean(Ls_all),
            "max": max(Ls_all),
            "p10": int(statistics.quantiles(Ls_all, n=10)[0]),
            "p90": int(statistics.quantiles(Ls_all, n=10)[8]),
        },
        "L_prompt_stats_kept": {
            "min": min(Ls_kept),
            "median": int(statistics.median(Ls_kept)),
            "mean": statistics.mean(Ls_kept),
            "max": max(Ls_kept),
        } if Ls_kept else {},
        "L_prompt_stats_by_set_all": {
            ps: {
                "min": min(Ls_by_set[ps]),
                "median": int(statistics.median(Ls_by_set[ps])),
                "mean": statistics.mean(Ls_by_set[ps]),
                "max": max(Ls_by_set[ps]),
                "n": len(Ls_by_set[ps]),
            }
            for ps in PROMPT_SETS
        },
        "L_prompt_stats_by_set_kept": {
            ps: ({
                "min": min(Ls_kept_by_set[ps]),
                "median": int(statistics.median(Ls_kept_by_set[ps])),
                "mean": statistics.mean(Ls_kept_by_set[ps]),
                "max": max(Ls_kept_by_set[ps]),
                "n": len(Ls_kept_by_set[ps]),
            } if Ls_kept_by_set[ps] else {"n": 0})
            for ps in PROMPT_SETS
        },
        "dropped_prompts": [
            {
                "prompt_set": ps,
                "prompt_id": pid,
                "L_bytes": L,
                "min_origin_stage": min_origin[(ps, pid)]["stage"],
                "min_origin_sample_idx": min_origin[(ps, pid)]["sample_idx"],
            }
            for (ps, pid), L in sorted(dropped.items(), key=lambda t: t[1])
        ],
        "L_prompt_per_prompt": {
            f"{ps}::{pid}": {
                "L_bytes": L,
                "min_origin_stage": min_origin[(ps, pid)]["stage"],
                "min_origin_sample_idx": min_origin[(ps, pid)]["sample_idx"],
                "kept": (ps, pid) in kept_prompts,
            }
            for (ps, pid), L in L_prompts.items()
        },
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[truncate] wrote {report_path}")


if __name__ == "__main__":
    main()
