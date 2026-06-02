from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR        = Path(__file__).parent
PROJECT_DIR    = SRC_DIR.parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
DATA_DIR       = PROJECT_DIR / "data"

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
# Sized down to match a small (~140k-token) dataset and avoid heavy overfitting.
# Scale these back up (e.g. 12 / 12 / 768 / 1024) if you train on far more data.
BLOCK_SIZE = 256    # context window (tokens)
VOCAB_SIZE = 50257  # GPT-2 BPE vocab
N_LAYER    = 4      # transformer blocks
N_HEAD     = 4      # attention heads per block
N_EMBD     = 256    # embedding dimension
DROPOUT    = 0.2    # higher regularization for the small dataset

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE    = 8       # sequences per step — lower if you run OOM
MAX_ITERS     = 5000
EVAL_INTERVAL = 500
EVAL_ITERS    = 100     # mini-batches to average for loss estimate
LEARNING_RATE = 6e-4
MIN_LR        = 6e-5    # cosine decay floor
WARMUP_ITERS  = 100
GRAD_CLIP     = 1.0

# ---------------------------------------------------------------------------
# Logging (Weights & Biases)
# ---------------------------------------------------------------------------
WANDB_LOG      = True                 # set False to train without wandb
WANDB_PROJECT  = "gpt2-from-scratch"  # project name in your wandb account
WANDB_RUN_NAME = None                 # None → wandb auto-generates a run name
