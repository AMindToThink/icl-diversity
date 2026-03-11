"""Demonstrate the BPE boundary merge: Qwen merges '.' + '\n\n' into '.\n\n'.

Shows that the full tokenization and truncated (prefix) tokenization produce
different tokens at response/separator boundaries, even though token counts
happen to match.
"""

from transformers import AutoTokenizer

PROMPT = "Write a short piece about rain."
RESPONSES = ["Rain falls gently.", "The drops patter.", "Water from above."]


def main() -> None:
    for model_id in ["Qwen/Qwen2.5-3B", "gpt2"]:
        tok = AutoTokenizer.from_pretrained(model_id)
        print(f"\n{'='*60}")
        print(f"Model: {model_id}")
        print(f"{'='*60}")

        # Build full text
        parts = [PROMPT]
        for i, r in enumerate(RESPONSES):
            label = chr(ord("A") + i)
            parts.append(f"\n\nResponse {label}: {r}")
        full_text = "".join(parts)
        full_ids = tok.encode(full_text, add_special_tokens=False)

        # Compare prefix tokenization vs full tokenization at each boundary
        running = PROMPT + "\n\nResponse A: "
        for k in range(len(RESPONSES)):
            running += RESPONSES[k]
            prefix_ids = tok.encode(running, add_special_tokens=False)
            n = len(prefix_ids)

            # Check if last token differs
            if prefix_ids[-1] != full_ids[n - 1]:
                print(
                    f"\n  After resp {k}: MERGE at boundary"
                    f"\n    prefix token [{n-1}] = {tok.decode([prefix_ids[-1]])!r}"
                    f"\n    full   token [{n-1}] = {tok.decode([full_ids[n-1]])!r}"
                )
            else:
                print(f"\n  After resp {k}: no merge (tokens match)")

            if k < len(RESPONSES) - 1:
                label = chr(ord("A") + k + 1)
                running += f"\n\nResponse {label}: "

        # Show full tokenization around first boundary
        n_prefix = len(
            tok.encode(
                PROMPT + "\n\nResponse A: " + RESPONSES[0], add_special_tokens=False
            )
        )
        print(f"\n  Tokens around first boundary (position {n_prefix}):")
        for i in range(max(0, n_prefix - 2), min(len(full_ids), n_prefix + 3)):
            print(f"    [{i}] {tok.decode([full_ids[i]])!r}")


if __name__ == "__main__":
    main()
