"""Compare aggregate a_k curve statistics between Qwen2.5-3B and GPT-2.

Prints mean a_k curves (total bits and per-byte), drop magnitudes,
drop distributions at m=10, and unconditional surprise variance.
"""

import json
from collections import defaultdict

import numpy as np


def analyze_model(path: str, label: str) -> None:
    with open(path) as f:
        data = json.load(f)

    by_m: dict[int, list[dict]] = defaultdict(list)
    for r in data["runs"]:
        by_m[r["m"]].append(r)

    n_responses = data["n_responses"]

    # --- Mean a_k curves (total bits) ---
    print(f"=== {label} (n={n_responses}) ===")
    print("  m     a_1      a_2      a_3      a_4      a_5      a_6    drop12   drop23   pct_drop12")
    for m in sorted(by_m):
        curves = np.array([r["a_k_curve"] for r in by_m[m]])
        mean = curves.mean(axis=0)
        d12 = mean[0] - mean[1]
        d23 = mean[1] - mean[2]
        pct = d12 / mean[0] * 100
        print(
            f"{m:>3}  {mean[0]:>7.1f}  {mean[1]:>7.1f}  {mean[2]:>7.1f}  "
            f"{mean[3]:>7.1f}  {mean[4]:>7.1f}  {mean[5]:>7.1f}  "
            f"{d12:>7.1f}  {d23:>7.1f}  {pct:>9.1f}%"
        )
    print()

    # --- Per-byte rates ---
    print(f"{label}: per-byte a_k curves (first 6)")
    print("  m    a1/b    a2/b    a3/b    a4/b    a5/b    a6/b   drop12/b")
    for m in sorted(by_m):
        curves = np.array([r["a_k_curve_per_byte"] for r in by_m[m]])
        mean = curves.mean(axis=0)
        d12 = mean[0] - mean[1]
        print(
            f"{m:>3}  {mean[0]:>6.3f}  {mean[1]:>6.3f}  {mean[2]:>6.3f}  "
            f"{mean[3]:>6.3f}  {mean[4]:>6.3f}  {mean[5]:>6.3f}  {d12:>7.3f}"
        )
    print()

    # --- Drop distribution at m=10 ---
    m10 = by_m[10]
    drops = [r["a_k_curve"][0] - r["a_k_curve"][1] for r in m10]
    drops_arr = np.array(drops)
    print(f"{label} m=10: drop from a_1 to a_2")
    print(f"  mean: {np.mean(drops_arr):.1f} bits")
    print(f"  median: {np.median(drops_arr):.1f} bits")
    print(f"  fraction positive: {np.mean(drops_arr > 0):.1%}")
    print(f"  min: {np.min(drops_arr):.1f}, max: {np.max(drops_arr):.1f}")
    print()

    # --- Unconditional surprise variance ---
    for m in [1, 5, 10]:
        runs = by_m[m]
        uncond_stds = [np.std(r["unconditional_total_bits"]) for r in runs]
        byte_stds = [np.std(r["a_k_byte_counts"]) for r in runs]
        print(
            f"{label} m={m}: uncond_std={np.mean(uncond_stds):.1f} bits, "
            f"byte_count_std={np.mean(byte_stds):.1f}"
        )
    print()

    # --- Byte counts by position at m=10 ---
    byte_counts = np.array([r["a_k_byte_counts"] for r in m10])
    mean_bytes = byte_counts.mean(axis=0)
    print(f"{label} m=10: mean byte counts by position")
    for k in [0, 1, 2, 9, n_responses - 1]:
        if k < len(mean_bytes):
            print(f"  position {k+1}: {mean_bytes[k]:.1f}")
    print()


if __name__ == "__main__":
    analyze_model("results/mode_count/qwen2.5-3b_1k_draws.json", "Qwen2.5-3B")
    analyze_model("results/mode_count/gpt2_1k_draws.json", "GPT-2")
