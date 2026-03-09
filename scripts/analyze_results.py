"""Analyze and summarize metrics from scenario result JSON files."""

import json


def fmt(v, w=10):
    if v is None:
        return " " * w
    return f"{v:>{w}.4f}"


def analyze_file(label: str, path: str) -> None:
    with open(path) as f:
        data = json.load(f)

    print(f"=== {label} ===")
    print(f"  base_model:      {data.get('base_model', 'N/A')}")
    print(f"  n_permutations:  {data.get('n_permutations', 'N/A')}")
    print(f"  n_responses:     {data.get('n_responses', 'N/A')}")
    print(f"  seed:            {data.get('seed', 'N/A')}")
    print()

    # Summary table
    hdr = (
        f"  {'prompt_label':<30} {'E(bits)':>10} {'E_rate':>10} "
        f"{'C':>10} {'D':>10} {'D_rate':>10} "
        f"{'sigma':>10} {'mono':>5}"
    )
    print(hdr)
    sep = f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 5}"
    print(sep)

    scenario_order = [
        "pure_noise",
        "multi_incoherent",
        "multi_mode",
        "one_mode",
        "mixed",
    ]

    for sname in scenario_order:
        prompts = data["scenarios"].get(sname, [])
        for r in prompts:
            pl = r.get("prompt_label", sname)
            E = r.get("excess_entropy_E")
            Er = r.get("excess_entropy_E_rate")
            C = r.get("coherence_C")
            D = r.get("diversity_score_D")
            Dr = r.get("diversity_score_D_rate")
            sig = r.get("coherence_spread_sigma", r.get("sigma"))
            mono = r.get("is_monotone")

            print(
                f"  {pl:<30} {fmt(E)} {fmt(Er)} "
                f"{fmt(C)} {fmt(D)} {fmt(Dr)} "
                f"{fmt(sig)} {str(mono):>5}"
            )

    print()

    # a_k curves
    print("  a_k curves (first 10 values):")
    for sname in scenario_order:
        prompts = data["scenarios"].get(sname, [])
        for r in prompts:
            pl = r.get("prompt_label", sname)
            ak_pb = r.get("a_k_curve_per_byte", [])
            ak_tot = r.get("a_k_curve", [])
            print(f"    {pl}:")
            vals_tot = ", ".join(f"{v:.2f}" for v in ak_tot[:10])
            vals_pb = ", ".join(f"{v:.4f}" for v in ak_pb[:10])
            print(f"      total bits: [{vals_tot}]")
            print(f"      bits/byte:  [{vals_pb}]")

    print()
    print("=" * 115)
    print()


def main():
    files = {
        "GPT-2 (3 perm)": "results/scenario_metrics_v2_3perm.json",
        "GPT-2 (100 perm)": "results/scenario_metrics_v2_100perm.json",
        "Qwen2.5-32B (3 perm)": "results/scenario_metrics_v2_qwen_3perm.json",
    }

    for label, path in files.items():
        try:
            analyze_file(label, path)
        except FileNotFoundError:
            print(f"=== {label} === FILE NOT FOUND: {path}")
            print()


if __name__ == "__main__":
    main()
