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
BLOCK_SIZE = 1024   # context window (tokens)
VOCAB_SIZE = 50257  # GPT-2 BPE vocab
N_LAYER    = 12     # transformer blocks
N_HEAD     = 12     # attention heads per block
N_EMBD     = 768    # embedding dimension
DROPOUT    = 0.1

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
