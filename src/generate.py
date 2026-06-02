"""
Command-line text generation for the from-scratch GPT-2.

Loads a checkpoint and autoregressively samples text from a prompt.

Examples:
    python src/generate.py --prompt "Sherlock Holmes turned to me and said"
    python src/generate.py -p "It was a dark night" --tokens 200 --temperature 0.7 --top-k 40
    python src/generate.py -p "Once upon a time" --checkpoint checkpoints/ckpt_01000.pt

The model architecture is read from the checkpoint itself (the GPTConfig saved at
train time), so generation works even if config.py has since changed.
"""

import argparse
import glob
import os
import sys

import torch
import tiktoken

# Make sure src/ is on the path when running as `python src/generate.py`
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import CHECKPOINT_DIR
from model import GPT, GPTConfig


def find_latest_checkpoint() -> str | None:
    """Return the most-recently modified .pt checkpoint, or None."""
    candidates = glob.glob(str(CHECKPOINT_DIR / "**" / "*.pt"), recursive=True)
    return max(candidates, key=os.path.getmtime) if candidates else None


def load_model(ckpt_path: str, device: str) -> GPT:
    """Build the model from the checkpoint's own saved config and load weights."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Use the config stored in the checkpoint so the architecture always matches
    # the trained weights (falls back to config.py defaults for older checkpoints).
    cfg = state.get("config", GPTConfig())
    model = GPT(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with the from-scratch GPT-2.")
    parser.add_argument("-p", "--prompt", required=True, help="Prompt text to continue.")
    parser.add_argument("-n", "--tokens", type=int, default=100,
                        help="Number of new tokens to generate (default: 100).")
    parser.add_argument("-t", "--temperature", type=float, default=0.8,
                        help="Sampling temperature; lower = more focused (default: 0.8).")
    parser.add_argument("-k", "--top-k", type=int, default=50,
                        help="Top-k sampling; 0 disables it (default: 50).")
    parser.add_argument("-c", "--checkpoint", default=None,
                        help="Path to a .pt checkpoint (default: most recent in checkpoints/).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible output.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = args.checkpoint or find_latest_checkpoint()
    if ckpt is None:
        sys.exit(f"No checkpoint found in {CHECKPOINT_DIR}. Train the model first.")
    if not os.path.exists(ckpt):
        sys.exit(f"Checkpoint not found: {ckpt}")

    print(f"Loading {ckpt} on {device} ...", file=sys.stderr)
    model = load_model(ckpt, device)

    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = enc.encode(args.prompt)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out = model.generate(
        ids,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
    )
    text = enc.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
