from pathlib import Path

# Paths
SRC_DIR       = Path(__file__).parent
PROJECT_DIR   = SRC_DIR.parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
DATA_DIR      = PROJECT_DIR / "data"

# GPT-2 small hyperparameters (wire these into your model)
BLOCK_SIZE = 1024   # context window
VOCAB_SIZE = 50257  # GPT-2 BPE vocab
N_LAYER    = 12
N_HEAD     = 12
N_EMBD     = 768
DROPOUT    = 0.1
