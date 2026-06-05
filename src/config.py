from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR           = Path(__file__).parent
PROJECT_DIR       = SRC_DIR.parent
CHECKPOINT_DIR    = PROJECT_DIR / "checkpoints"
RL_CHECKPOINT_DIR = PROJECT_DIR / "checkpoints_RL"   # PPO fine-tuned checkpoints
DATA_DIR          = PROJECT_DIR / "data"

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
MAX_ITERS     = 12000
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

# ---------------------------------------------------------------------------
# RL fine-tuning (PPO) — "dialogify" the model
# ---------------------------------------------------------------------------
# The actor/policy is the pretrained GPT-2 (weights updated). A frozen copy of
# it is the reference model (KL leash). A new Linear(n_embd, 1) value head is the
# critic. Reward = fraction of generated tokens that sit inside quotation marks.

# Rollout ("simulation") settings
RL_ITERS         = 200    # number of PPO iterations
RL_BATCH_SIZE    = 16     # completions sampled per iteration
RL_PROMPT_LEN    = 16     # tokens of prompt (sampled from train data)
RL_GEN_LEN       = 64     # new tokens generated per completion
RL_TEMPERATURE   = 1.0    # sampling temperature during rollouts (exploration)
RL_TOP_K         = 0      # 0 = full distribution (more exploration); >0 to restrict

# PPO update settings
RL_LEARNING_RATE = 1e-5   # much lower than pretraining — RL updates are delicate
PPO_EPOCHS       = 4      # optimization passes over each batch of rollouts
PPO_CLIP         = 0.2    # PPO probability-ratio clip epsilon
PPO_MINIBATCH    = 8      # completions per minibatch in the update
GAE_GAMMA        = 1.0    # discount (1.0 is standard for short text episodes)
GAE_LAMBDA       = 0.95   # GAE smoothing
VALUE_COEFF      = 0.5    # weight on the critic (value) loss
ENTROPY_COEFF    = 0.01   # entropy bonus (encourages exploration)
KL_COEFF         = 0.1    # β — strength of the KL leash to the reference model
RL_GRAD_CLIP     = 1.0
RL_SAVE_INTERVAL = 20     # save a checkpoint every N PPO iterations

# Reward shaping (see reward.py)
RL_PAIR_BONUS         = 0.5   # bonus per *balanced* "…" quote pair (rewards real exchanges)
RL_UNBALANCED_PENALTY = 1.0   # penalty if a completion leaves a quote unclosed

# Logging
RL_WANDB_PROJECT = "gpt2-rl-dialogify"
