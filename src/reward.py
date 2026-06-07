"""
"Dialogify" reward for PPO fine-tuning.

The behavior we want to induce: the model should write *dialogue* — characters
talking inside quotation marks — instead of plain narration.

This module is the programmatic stand-in for a learned reward model. In real
RLHF a neural net trained on human preferences would score the text; here we
score it with a transparent function instead. The PPO loop is identical either
way (this is the "RLVR" / verifiable-reward style of RL fine-tuning).


Normalised reward (so the numbers are easy to read while tuning)
----------------------------------------------------------------
The completion-level score is a weighted sum of THREE sub-scores, each a
fraction in [0, 1], so the whole reward sits in roughly [-W_UNBALANCED, +1]:

    score =  W_DIALOGUE   * dialogue_fraction     # 0..1  is the text real dialogue?
           + W_PAIRS      * pair_score            # 0..1  enough complete exchanges?
           - W_UNBALANCED * stranded_fraction     # 0..1  text stuck in an open quote

With W_DIALOGUE + W_PAIRS = 1, a perfect completion scores ~1.0, pure narration
scores 0.0, and the "open a quote and ramble forever" hack goes NEGATIVE. You
can read the reward like a percentage, which makes weights (and the KL_COEFF
that competes with them) much easier to tune.

What counts as "real dialogue" (the anti-hack rule)
---------------------------------------------------
A quoted span only counts if it is *closed* AND holds at least MIN_SPAN
characters. Text inside a quote that never closes earns ZERO dialogue credit and
instead drives the `stranded_fraction` penalty. This removes the degenerate
optimum where the model emits one `"` and then spews tokens forever.

Per-token rewards
-----------------
PPO needs a per-token reward for credit assignment (GAE). We give the dense
dialogue credit per token (each token gets its share of `dialogue_fraction`) and
attach the two completion-level shaping terms (pair bonus, stranded penalty) to
the LAST token, where GAE propagates them backward. By construction
`sum(per_token) == total`, so the per-token rewards and the headline score agree.

The KL-to-reference penalty is NOT added here — it's applied in the PPO loop,
because it depends on the policy and reference log-probs, not just the text.

Run `python src/reward.py` to see the reward on a few hand-written examples.
"""

from dataclasses import dataclass

import tiktoken


# TUNABLE REWARD PARAMETERS  
# Weights are on a common [0, 1] scale. Keep W_DIALOGUE + W_PAIRS ≈ 1 so that a
# perfect completion scores ~1.0 and can read the reward like a percentage.

W_DIALOGUE   = 0.7   # weight on the main signal: fraction of text that is real (closed) dialogue
W_PAIRS      = 0.3   # weight on having enough complete "…" exchanges
W_UNBALANCED = 1.0   # penalty weight for text left stranded inside an unclosed quote (anti-hack)

TARGET_PAIRS = 2     # how many closed quote-pairs counts as "full marks" for the pair bonus
MIN_SPAN     = 3     # a quoted span must hold at least this many chars to count as real dialogue

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
    per_token: list[float]      # one reward per generated token (pre-KL); sums to `total`
    dialogue_fraction: float    # chars inside VALID (closed, ≥MIN_SPAN) quotes / total chars  [0..1]
    pair_score: float           # min(num_pairs / TARGET_PAIRS, 1)                              [0..1]
    stranded_fraction: float    # chars stuck inside an UNCLOSED quote / total chars            [0..1]
    num_pairs: int              # number of valid closed "…" pairs
    unbalanced: bool            # True if a quote was left open at the end
    total: float                # the normalised completion score (== sum of per_token)


def dialogue_reward(
    gen_token_ids: list[int],
    enc: tiktoken.Encoding,
    *,
    w_dialogue: float = W_DIALOGUE,
    w_pairs: float = W_PAIRS,
    w_unbalanced: float = W_UNBALANCED,
    target_pairs: int = TARGET_PAIRS,
    min_span: int = MIN_SPAN,
) -> RewardInfo:
    """Score a list of *generated* token ids for dialogue-ness.

    Returns per-token rewards (aligned to ``gen_token_ids``) plus a breakdown.
    Only pass the generated tokens — not the prompt — so the prompt isn't scored.
    All weights default to the TUNABLE block at the top of this file; override
    them per-call if you want to sweep values without editing the module.
    """
    if not gen_token_ids:
        return RewardInfo([], 0.0, 0.0, 0.0, 0, False, 0.0)

    # IMPORTANT: decode the FULL completion at once. Curly quotes (“ ”) and many
    # other characters are multi-byte UTF-8 — decoding token-by-token would split
    # those bytes and lose the quote characters entirely. We then map each
    # character back to the token whose span it falls in, via cumulative offsets.
    G = len(gen_token_ids)
    offsets = [0] * (G + 1)                     # offsets[i] = char index where token i starts
    for i in range(1, G + 1):
        offsets[i] = len(enc.decode(gen_token_ids[:i]))
    full = enc.decode(gen_token_ids)
    total_chars = len(full)
    if total_chars == 0:
        return RewardInfo([0.0] * G, 0.0, 0.0, 0.0, 0, False, 0.0)

    # Walk the full string with a directional quote-state machine. A character
    # only counts as "dialogue" once its span has been CLOSED with ≥ min_span
    # content chars, so we buffer each open span's positions and commit them on a
    # valid close. Anything still open at the end is "stranded" (the hack).
    valid_flags = [False] * total_chars         # char is inside a VALID closed span
    in_quote = False
    span_positions: list[int] = []              # content-char indices of the current open span
    valid_pairs = 0
    stranded_chars = 0

    def open_quote():
        nonlocal in_quote, span_positions
        in_quote = True
        span_positions = []

    def close_quote():
        nonlocal in_quote, valid_pairs, span_positions
        if in_quote and len(span_positions) >= min_span:
            valid_pairs += 1
            for p in span_positions:            # commit this span's chars as real dialogue
                valid_flags[p] = True
        in_quote = False
        span_positions = []

    for pos, ch in enumerate(full):
        if ch == _OPEN_QUOTE:
            open_quote()
        elif ch == _CLOSE_QUOTE:
            close_quote()
        elif ch == _STRAIGHT_QUOTE:             # ambiguous → toggle
            close_quote() if in_quote else open_quote()
        elif in_quote:
            span_positions.append(pos)          # content char inside the (still open) span

    if in_quote:                                # quote never closed → these chars are stranded
        stranded_chars = len(span_positions)

    # ---- the three normalised [0, 1] sub-scores ----
    dialogue_fraction = sum(valid_flags) / total_chars
    pair_score        = min(valid_pairs / target_pairs, 1.0) if target_pairs > 0 else 0.0
    stranded_fraction = stranded_chars / total_chars

    # ---- per-token dense reward: each token's share of the dialogue credit ----
    # Summed over tokens this equals w_dialogue * dialogue_fraction.
    per_token: list[float] = []
    for i in range(G):
        a, b = offsets[i], offsets[i + 1]
        per_token.append(w_dialogue * sum(valid_flags[a:b]) / total_chars)

    # ---- completion-level shaping → attached to the last token (GAE spreads it) ----
    per_token[-1] += w_pairs * pair_score
    per_token[-1] -= w_unbalanced * stranded_fraction

    return RewardInfo(
        per_token=per_token,
        dialogue_fraction=dialogue_fraction,
        pair_score=pair_score,
        stranded_fraction=stranded_fraction,
        num_pairs=valid_pairs,
        unbalanced=in_quote,
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

    print(f"weights:  W_DIALOGUE={W_DIALOGUE}  W_PAIRS={W_PAIRS}  "
          f"W_UNBALANCED={W_UNBALANCED}  TARGET_PAIRS={TARGET_PAIRS}  MIN_SPAN={MIN_SPAN}\n")
    print(f"{'example':<42} {'dlg':>5} {'pair':>5} {'strand':>7} {'pairs':>6} {'total':>7}")
    print("-" * 76)
    for name, text in examples.items():
        ids = enc.encode_ordinary(text)
        info = dialogue_reward(ids, enc)
        print(f"{name:<42} {info.dialogue_fraction:>5.2f} {info.pair_score:>5.2f} "
              f"{info.stranded_fraction:>7.2f} {info.num_pairs:>6d} {info.total:>7.2f}")

    print("\nExpected with default weights: narration ≈ 0.00; dialogue positive and")
    print("rising with more closed exchanges; BOTH hacks now score ≤ 0 (the unclosed")
    print("quote goes negative via the stranded penalty, quote-spam earns 0 because")
    print("its spans are shorter than MIN_SPAN). The KL penalty in PPO suppresses")
    print("them further. Scores are normalised: ~1.0 is a perfect dialogue completion.")


if __name__ == "__main__":
    _selftest()
