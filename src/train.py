"""
GPT-2 training script.

  - tiktoken BPE tokenizer (50257-token GPT-2 vocab)
  - Data loaded from data/train.txt & data/val.txt; cached as .bin after first run
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - Checkpoint saved every eval_interval steps

NOTE — dataset size vs. model size:
  Sherlock Holmes corpus is ~130k BPE tokens.  GPT-2 small has ~117M params and
  was trained on 40 GB of text.  Expect heavy overfitting at this scale.
  For experimentation on this corpus, consider a smaller config:
      n_layer=4, n_head=4, n_embd=128  (~1.5M params)
  Change the constants in config.py to try a smaller model.
"""

import math
import sys
import os

import numpy as np
import torch
import tiktoken

# Make sure src/ is on the path when running as `python src/train.py`
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import (
    BLOCK_SIZE,
    BATCH_SIZE, MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS,
    LEARNING_RATE, MIN_LR, WARMUP_ITERS, GRAD_CLIP,
    DATA_DIR, CHECKPOINT_DIR,
    WANDB_LOG, WANDB_PROJECT, WANDB_RUN_NAME,
)
from model import GPT, GPTConfig

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")


def _load_split(split: str) -> np.ndarray:
    """Tokenize a text split once and cache it as a binary file."""
    bin_path = DATA_DIR / f"{split}.bin"
    if not bin_path.exists():
        txt_path = DATA_DIR / f"{split}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(
                f"{txt_path} not found — run `python src/prepare_data.py` first."
            )
        text = txt_path.read_text(encoding="utf-8")
        ids = enc.encode_ordinary(text)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(bin_path)
        print(f"Tokenized {split}: {len(arr):,} tokens  →  {bin_path}")
    return np.fromfile(bin_path, dtype=np.uint16)


train_data = _load_split("train")
val_data   = _load_split("val")
print(f"Train tokens: {len(train_data):,}   Val tokens: {len(val_data):,}")


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([
        torch.from_numpy(data[i : i + BLOCK_SIZE].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix
    ])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------
def get_lr(step: int) -> float:
    if step < WARMUP_ITERS:
        return LEARNING_RATE * step / WARMUP_ITERS
    if step >= MAX_ITERS:
        return MIN_LR
    decay = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
config = GPTConfig()  # defaults sourced from config.py
model = GPT(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params / 1e6:.1f}M parameters")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Weights & Biases (optional)
# ---------------------------------------------------------------------------
use_wandb = WANDB_LOG
if use_wandb:
    try:
        import wandb
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "n_layer": config.n_layer, "n_head": config.n_head,
                "n_embd": config.n_embd, "block_size": config.block_size,
                "vocab_size": config.vocab_size, "dropout": config.dropout,
                "n_params": n_params, "batch_size": BATCH_SIZE,
                "max_iters": MAX_ITERS, "learning_rate": LEARNING_RATE,
                "min_lr": MIN_LR, "warmup_iters": WARMUP_ITERS,
                "weight_decay": 0.1, "train_tokens": len(train_data),
            },
        )
        print(f"wandb: logging to project '{WANDB_PROJECT}'")
    except Exception as e:
        use_wandb = False
        print(f"wandb disabled ({e}). Continuing without logging.")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for step in range(MAX_ITERS):

    # Update LR
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Periodic evaluation + checkpoint
    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        losses = estimate_loss(model)
        print(
            f"step {step:5d} | lr {lr:.2e} | "
            f"train loss {losses['train']:.4f} | val loss {losses['val']:.4f}"
        )
        ckpt_path = CHECKPOINT_DIR / f"ckpt_{step:05d}.pt"
        torch.save({"step": step, "model": model.state_dict(), "config": config}, ckpt_path)
        print(f"         saved → {ckpt_path.name}")
        if use_wandb:
            wandb.log(
                {"train/loss": losses["train"], "val/loss": losses["val"]},
                step=step,
            )

    # Forward + backward
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    if use_wandb:
        wandb.log({"train/batch_loss": loss.item(), "lr": lr}, step=step)

print("Training complete.")
if use_wandb:
    wandb.finish()
