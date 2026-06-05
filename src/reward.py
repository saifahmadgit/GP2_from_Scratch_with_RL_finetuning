"""
"Dialogify" reward for PPO fine-tuning.

The behavior we want to induce: the model should write *dialogue* — characters
talking inside quotation marks — instead of plain narration.

This module is the programmatic stand-in for a learned reward model. In real
RLHF a neural net trained on human preferences would score the text; here we
score it with a transparent function instead. The PPO loop is identical either
way (this is the "RLVR" / verifiable-reward style of RL fine-tuning).

Core metric
-----------
    dialogue_fraction = (chars inside quotes) / (total chars)

PPO needs a *per-token* reward (for credit assignment via GAE), so we walk the
generated tokens left-to-right with a quote-state machine: every `"` flips an
`in_quote` flag, and each token's reward is the fraction of its characters that
land inside an open quote.

Anti-reward-hacking shaping (added at the final token):
  + RL_PAIR_BONUS         per *balanced* "…" pair  → rewards real exchanges
  - RL_UNBALANCED_PENALTY if a quote is left open  → forces the model to close them

The KL-to-reference penalty is NOT added here — it's applied in the PPO loop,
because it depends on the policy and reference log-probs, not just the text.

Run `python src/reward.py` to see the reward on a few hand-written examples.
"""

from dataclasses import dataclass

import tiktoken

# Quote delimiters. Curly quotes are DIRECTIONAL — “ always opens, ” always
# closes — so we must not treat them as plain toggles, or a stray closing ”
# (common when the model emits `,”` mid-stream) flips the parity and inverts the
# whole reward. The straight " is ambiguous, so it toggles.
_OPEN_QUOTE     = "“"   # U+201C
_CLOSE_QUOTE    = "”"   # U+201D
_STRAIGHT_QUOTE = '"'   # U+0022


@dataclass
class RewardInfo:
    """Breakdown of the reward for one completion (for logging / debugging)."""
    per_token: list[float]      # one reward per generated token (pre-KL)
    dialogue_fraction: float    # chars-in-quotes / total chars  (the headline metric)
    num_pairs: int              # number of balanced "…" pairs
    unbalanced: bool            # True if a quote was left open
    total: float                # sum of per_token (completion-level scalar)


def dialogue_reward(
    gen_token_ids: list[int],
    enc: tiktoken.Encoding,
    pair_bonus: float = 0.5,
    unbalanced_penalty: float = 1.0,
) -> RewardInfo:
    """Score a list of *generated* token ids for dialogue-ness.

    Returns per-token rewards (aligned to ``gen_token_ids``) plus a breakdown.
    Only pass the generated tokens — not the prompt — so the prompt isn't scored.
    """
    if not gen_token_ids:
        return RewardInfo([], 0.0, 0, False, 0.0)

    # IMPORTANT: decode the FULL completion at once. Curly quotes (“ ”) and many
    # other characters are multi-byte UTF-8 — decoding token-by-token would split
    # those bytes and lose the quote characters entirely. We then map each
    # character back to the token whose span it falls in, via cumulative offsets.
    G = len(gen_token_ids)
    offsets = [0] * (G + 1)                    # offsets[i] = char index where token i starts
    for i in range(1, G + 1):
        offsets[i] = len(enc.decode(gen_token_ids[:i]))
    full = enc.decode(gen_token_ids)
    total_chars = len(full)

    # Walk the full string with a directional quote-state machine; flag which
    # chars are inside an open quote.
    inside_flags = [False] * total_chars
    in_quote = False
    valid_pairs = 0
    span_content = 0
    MIN_SPAN = 3             # a quoted span must hold ≥ this many chars to count as a real line

    def close_quote():       # record a content-bearing pair on close
        nonlocal in_quote, valid_pairs
        if in_quote and span_content >= MIN_SPAN:
            valid_pairs += 1
        in_quote = False

    for pos, ch in enumerate(full):
        if ch == _OPEN_QUOTE:
            in_quote = True
            span_content = 0
        elif ch == _CLOSE_QUOTE:
            close_quote()
        elif ch == _STRAIGHT_QUOTE:           # ambiguous → toggle
            if not in_quote:
                in_quote = True
                span_content = 0
            else:
                close_quote()
        elif in_quote:
            inside_flags[pos] = True
            span_content += 1

    # Per-token reward = fraction of that token's characters that are inside quotes.
    per_token: list[float] = []
    for i in range(G):
        a, b = offsets[i], offsets[i + 1]
        span = b - a
        per_token.append((sum(inside_flags[a:b]) / span) if span > 0 else 0.0)

    inside_chars = sum(inside_flags)
    num_pairs = valid_pairs
    unbalanced = in_quote                     # a quote was left open at the end

    # Fold the completion-level shaping into the LAST generated token's reward,
    # so it flows through GAE like any other reward.
    if per_token:
        per_token[-1] += pair_bonus * num_pairs
        if unbalanced:
            per_token[-1] -= unbalanced_penalty

    return RewardInfo(
        per_token=per_token,
        dialogue_fraction=(inside_chars / total_chars) if total_chars else 0.0,
        num_pairs=num_pairs,
        unbalanced=unbalanced,
        total=sum(per_token),
    )


# ---------------------------------------------------------------------------
# Standalone sanity test:  python src/reward.py
# ---------------------------------------------------------------------------
def _selftest() -> None:
    enc = tiktoken.get_encoding("gpt2")

    examples = {
        "pure narration":
            "He walked into the dim room and looked around carefully.",
        "real dialogue":
            'He said, "Hello there, Watson!" and smiled warmly at us.',
        "lots of dialogue":
            '"Who did it?" she cried. "I cannot say," he replied softly.',
        "curly quotes (like the corpus)":
            "“Who did it?” she cried. “I cannot say,” he replied softly.",
        "stray leading close-quote (was inverting)":
            "blood,” said Watson. “The body was cold. “Indeed,” Holmes murmured.",
        "HACK: one giant unclosed quote":
            'He said "and on and on and on and on and on forever and ever',
        "HACK: quote spam":
            '" " " " " " " " " " " " " " " " " " " "',
    }

    print(f"{'example':<38} {'frac':>6} {'pairs':>6} {'unbal':>6} {'total':>8}")
    print("-" * 70)
    for name, text in examples.items():
        ids = enc.encode_ordinary(text)
        info = dialogue_reward(ids, enc)
        print(f"{name:<38} {info.dialogue_fraction:>6.2f} "
              f"{info.num_pairs:>6d} {str(info.unbalanced):>6} {info.total:>8.2f}")

    print("\nExpected: narration ≈ 0 reward; dialogue high; the unclosed-quote")
    print("hack is dragged down by the unbalanced penalty; quote-spam earns ~0")
    print("content reward. (The KL penalty in PPO further suppresses both hacks.)")


if __name__ == "__main__":
    _selftest()
