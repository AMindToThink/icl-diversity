"""Compare old (progressive) vs new (offset-mapping) boundary detection.

Shows that both approaches produce identical boundaries for all tested cases,
because the BPE merge preserves token counts even though individual tokens
differ.  The offset-mapping approach is more principled (derived directly from
the actual full tokenization) even when outputs match.
"""

from transformers import AutoTokenizer

from icl_diversity.core import _find_response_boundaries, _response_label

PROMPT = "Write a short piece about rain."

TEST_CASES: list[list[str]] = [
    # Standard responses ending in periods
    ["Rain falls gently.", "The drops patter."],
    ["Rain falls gently.", "The drops patter.", "Water from above."],
    # Various punctuation
    ["Hello world!", "Goodbye world."],
    ["Data: 42%", "Result: OK"],
    ["End)", "Start("],
    ["rain...", "more rain."],
    ['(rain)', '"rain"'],
]


def old_progressive_boundaries(
    tok: object, prompt: str, responses: list[str]
) -> list[tuple[int, int]]:
    """The old progressive tokenization approach (pre-fix)."""
    boundaries: list[tuple[int, int]] = []
    running = prompt + f"\n\nResponse {_response_label(0)}: "
    for k in range(len(responses)):
        n_prefix = len(tok.encode(running, add_special_tokens=False))  # type: ignore[union-attr]
        running += responses[k]
        n_with = len(tok.encode(running, add_special_tokens=False))  # type: ignore[union-attr]
        boundaries.append((n_prefix, n_with))
        if k < len(responses) - 1:
            running += f"\n\nResponse {_response_label(k + 1)}: "
    return boundaries


def main() -> None:
    for model_id in ["Qwen/Qwen2.5-3B", "gpt2"]:
        tok = AutoTokenizer.from_pretrained(model_id)
        print(f"\n{'='*60}")
        print(f"Model: {model_id}")
        print(f"{'='*60}")

        n_differ = 0
        for responses in TEST_CASES:
            full_ids, new_b = _find_response_boundaries(tok, PROMPT, responses)
            old_b = old_progressive_boundaries(tok, PROMPT, responses)

            if old_b != list(new_b):
                print(f"\n  DIFFER for {responses}:")
                print(f"    OLD: {old_b}")
                print(f"    NEW: {list(new_b)}")
                for k in range(len(responses)):
                    if old_b[k] != new_b[k]:
                        os, oe = old_b[k]
                        ns, ne = new_b[k]
                        print(
                            f"    resp {k}: old=[{tok.decode(full_ids[os:oe])!r}]"
                            f" new=[{tok.decode(full_ids[ns:ne])!r}]"
                        )
                n_differ += 1
            else:
                print(f"  Same for {responses}: {old_b}")

        print(f"\n  {n_differ}/{len(TEST_CASES)} cases differ")


if __name__ == "__main__":
    main()
