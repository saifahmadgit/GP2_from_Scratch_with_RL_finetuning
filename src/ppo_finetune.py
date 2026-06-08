"""
PPO fine-tuning ("dialogify") for the from-scratch GPT-2.

This is the RL stage of an RLHF-style pipeline, with the human-trained reward
model replaced by a programmatic reward (see reward.py) — i.e. the "RLVR" /
verifiable-reward style. The RL machinery (PPO + KL penalty) is identical to
what ChatGPT's final stage uses.

Cast of characters
------------------
  actor / policy  : the pretrained GPT-2 (lm_head). ITS WEIGHTS GET UPDATED.
  critic / value  : a new Linear(n_embd, 1) on the SHARED trunk. Trained fresh.
  reference       : a FROZEN copy of the base model. Only used for the KL leash.
  environment     : "append the sampled token"; reward = dialogue_reward(text).

Usage
-----
    # fine-tune the latest checkpoint in checkpoints/
    uv run python src/ppo_finetune.py

    # fine-tune a specific base checkpoint, for fewer iters
    uv run python src/ppo_finetune.py -c checkpoints/ckpt_11999.pt --iters 100

Checkpoints are saved to checkpoints_RL/ in the SAME format train.py uses, so
generate.py can load them directly:
    uv run python src/generate.py -p "The inspector" -c checkpoints_RL/rl_00200.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# Make sure src/ is on the path when running as `python src/ppo_finetune.py`
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import (
    DATA_DIR, CHECKPOINT_DIR, RL_CHECKPOINT_DIR,
    RL_ITERS, RL_BATCH_SIZE, RL_PROMPT_LEN, RL_GEN_LEN,
    RL_TEMPERATURE, RL_TOP_K,
    RL_LEARNING_RATE, PPO_EPOCHS, PPO_CLIP, PPO_MINIBATCH,
    GAE_GAMMA, GAE_LAMBDA, VALUE_COEFF, ENTROPY_COEFF, KL_COEFF,
    KL_ADAPTIVE, KL_TARGET, KL_HORIZON,
    RL_GRAD_CLIP, RL_SAVE_INTERVAL,
    WANDB_LOG, RL_WANDB_PROJECT,
)
from model import GPT, GPTConfig
from generate import find_latest_checkpoint
from reward import dialogue_reward, W_DIALOGUE, W_PAIRS, W_UNBALANCED

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")


# ---------------------------------------------------------------------------
# Adaptive KL controller (InstructGPT / Ziegler et al. 2019; same as TRL)
# ---------------------------------------------------------------------------
class AdaptiveKLController:
    """Auto-tunes the KL coefficient β to hold the measured KL near a target.

    Robotics analogy: a P-controller on the KL "leash". The error is how far the
    measured per-completion KL is above/below the target; β is the actuator. The
    error is clipped to ±20% so β never jumps more than that fraction per step,
    and `horizon` sets how gently it corrects. `value` is the live β.
    """
    def __init__(self, init_coef: float, target: float, horizon: float):
        self.value = init_coef
        self.target = target
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        proportional_error = float(np.clip(current_kl / self.target - 1.0, -0.2, 0.2))
        self.value *= 1.0 + proportional_error * n_steps / self.horizon


# ---------------------------------------------------------------------------
# Prompts: random windows sampled from the training corpus
# ---------------------------------------------------------------------------
def load_prompt_pool() -> np.ndarray:
    bin_path = DATA_DIR / "train.bin"
    if not bin_path.exists():
        sys.exit(f"{bin_path} not found — run train.py once to create it.")
    return np.fromfile(bin_path, dtype=np.uint16)


def sample_prompts(pool: np.ndarray, batch: int, prompt_len: int) -> torch.Tensor:
    ix = torch.randint(len(pool) - prompt_len, (batch,))
    out = torch.stack([
        torch.from_numpy(pool[i: i + prompt_len].astype(np.int64)) for i in ix
    ])
    return out.to(device)


# ---------------------------------------------------------------------------
# Rollout: generate completions from the current policy (no grad)
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout(policy: GPT, prompts: torch.Tensor) -> torch.Tensor:
    """Autoregressively sample RL_GEN_LEN tokens. Returns (B, prompt_len+gen_len)."""
    idx = prompts
    for _ in range(RL_GEN_LEN):
        idx_cond = idx[:, -policy.config.block_size:]
        logits, _ = policy(idx_cond)
        logits = logits[:, -1, :] / RL_TEMPERATURE
        if RL_TOP_K and RL_TOP_K > 0:
            v, _ = torch.topk(logits, min(RL_TOP_K, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, nxt), dim=1)
    return idx


# ---------------------------------------------------------------------------
# Teacher-forced pass: log-probs, values and entropy for the GENERATED tokens
# ---------------------------------------------------------------------------
def eval_sequences(
    model: GPT, value_head: nn.Module, seq: torch.Tensor, prompt_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (logp, values, entropy), each (B, gen_len), for the generated tokens.

    For a generated token at index i, the distribution that produced it comes
    from the logits at position i-1 (the state = tokens before it). So we slice
    positions [prompt_len-1 : L-1].
    """
    L = seq.size(1)
    x = model.trunk(seq)                       # (B, L, n_embd) — shared trunk
    logits = model.lm_head(x)                  # actor head    (B, L, V)
    values = value_head(x).squeeze(-1)         # critic head   (B, L)

    gen_logits = logits[:, prompt_len - 1: L - 1, :]   # (B, G, V)
    actions = seq[:, prompt_len:L]                     # (B, G)
    logp_all = F.log_softmax(gen_logits, dim=-1)
    logp = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)   # (B, G)
    entropy = -(logp_all.exp() * logp_all).sum(-1)                  # (B, G)
    gen_values = values[:, prompt_len - 1: L - 1]                   # (B, G)
    return logp, gen_values, entropy


def compute_rewards(seq: torch.Tensor, prompt_len: int) -> tuple[torch.Tensor, list]:
    """Per-token dialogue reward for the generated tokens (pre-KL). (B, G)."""
    rewards = torch.zeros(seq.size(0), RL_GEN_LEN, device=device)
    infos = []
    for b in range(seq.size(0)):
        gen_ids = seq[b, prompt_len:].tolist()
        info = dialogue_reward(gen_ids, enc)   # weights come from reward.py's tunable block
        rewards[b] = torch.tensor(info.per_token, device=device)
        infos.append(info)
    return rewards, infos


def compute_gae(rewards: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation. rewards/values: (B, G). Episode is
    truncated at G tokens, so we bootstrap the final step with 0 (terminal)."""
    B, G = rewards.shape
    advantages = torch.zeros_like(rewards)
    last = torch.zeros(B, device=device)
    for t in reversed(range(G)):
        next_value = values[:, t + 1] if t < G - 1 else torch.zeros(B, device=device)
        delta = rewards[:, t] + GAE_GAMMA * next_value - values[:, t]
        last = delta + GAE_GAMMA * GAE_LAMBDA * last
        advantages[:, t] = last
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="PPO 'dialogify' fine-tuning of the from-scratch GPT-2.")
    ap.add_argument("-c", "--checkpoint", default=None,
                    help="Base checkpoint to fine-tune (default: latest in checkpoints/).")
    ap.add_argument("--iters", type=int, default=RL_ITERS, help="Number of PPO iterations.")
    ap.add_argument("--name", default="exp_12", help="wandb run name (default: exp_12).")
    ap.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    args = ap.parse_args()

    base_ckpt = args.checkpoint or find_latest_checkpoint()
    if base_ckpt is None or not os.path.exists(base_ckpt):
        sys.exit(f"Base checkpoint not found: {base_ckpt}. Train with train.py first.")
    print(f"Device: {device}")
    print(f"Fine-tuning base checkpoint: {base_ckpt}")

    state = torch.load(base_ckpt, map_location=device, weights_only=False)
    cfg: GPTConfig = state.get("config", GPTConfig())

    # actor / policy — weights WILL be updated
    policy = GPT(cfg).to(device)
    policy.load_state_dict(state["model"])
    policy.eval()                      # dropout OFF for stable log-probs during RL

    # reference — FROZEN copy for the KL leash
    reference = GPT(cfg).to(device)
    reference.load_state_dict(state["model"])
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    # critic — a NEW value head on the shared trunk
    value_head = nn.Linear(cfg.n_embd, 1).to(device)

    optimizer = torch.optim.AdamW(
        list(policy.parameters()) + list(value_head.parameters()),
        lr=RL_LEARNING_RATE, betas=(0.9, 0.95),
    )

    # KL leash: adaptive β (InstructGPT-style) or a fixed constant.
    kl_ctl = AdaptiveKLController(KL_COEFF, KL_TARGET, KL_HORIZON)
    print(f"KL leash: {'adaptive' if KL_ADAPTIVE else 'fixed'} β, "
          f"start={KL_COEFF}" + (f", target={KL_TARGET} nats" if KL_ADAPTIVE else ""))

    pool = load_prompt_pool()
    RL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    use_wandb = WANDB_LOG and not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=RL_WANDB_PROJECT, name=args.name, config={
                "base_ckpt": os.path.basename(base_ckpt), "iters": args.iters,
                "batch": RL_BATCH_SIZE, "gen_len": RL_GEN_LEN, "lr": RL_LEARNING_RATE,
                "kl_coeff": KL_COEFF, "ppo_clip": PPO_CLIP, "ppo_epochs": PPO_EPOCHS,
                "kl_adaptive": KL_ADAPTIVE, "kl_target": KL_TARGET, "kl_horizon": KL_HORIZON,
            })
            print(f"wandb: logging to '{RL_WANDB_PROJECT}' as run '{args.name}'")
        except Exception as e:
            use_wandb = False
            print(f"wandb disabled ({e}).")

    prompt_len = RL_PROMPT_LEN
    for it in range(args.iters):
        # ---- 1. ROLLOUT (the "simulation") ----
        prompts = sample_prompts(pool, RL_BATCH_SIZE, prompt_len)
        seq = rollout(policy, prompts)                       # (B, P+G)

        # ---- 2. REWARD ----
        with torch.no_grad():
            old_logp, old_values, _ = eval_sequences(policy, value_head, seq, prompt_len)
            ref_logp, _, _ = eval_sequences(reference, value_head, seq, prompt_len)
        task_rewards, infos = compute_rewards(seq, prompt_len)
        kl = old_logp - ref_logp                             # per-token KL estimate
        rewards = task_rewards - kl_ctl.value * kl           # KL leash folded into reward (live β)

        # ---- 3. ADVANTAGE ----
        advantages, returns = compute_gae(rewards, old_values)
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- 4. PPO UPDATE ----
        B = seq.size(0)
        pg_losses, v_losses, ent_vals = [], [], []
        clip_fracs, approx_kls, grad_norms = [], [], []
        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(B)
            for s in range(0, B, PPO_MINIBATCH):
                mb = perm[s: s + PPO_MINIBATCH]
                new_logp, new_values, entropy = eval_sequences(policy, value_head, seq[mb], prompt_len)
                logratio = new_logp - old_logp[mb]
                ratio = torch.exp(logratio)
                a = adv_norm[mb]
                pg = -torch.min(ratio * a,
                                torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * a).mean()
                v_loss = 0.5 * (new_values - returns[mb]).pow(2).mean()
                ent = entropy.mean()
                loss = pg + VALUE_COEFF * v_loss - ENTROPY_COEFF * ent

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value_head.parameters()), RL_GRAD_CLIP)
                optimizer.step()
                pg_losses.append(pg.item()); v_losses.append(v_loss.item()); ent_vals.append(ent.item())
                with torch.no_grad():
                    # diagnostics: how much THIS update moved the policy (vs. the
                    # reference-KL logged elsewhere) and whether the clip is binding.
                    clip_fracs.append(((ratio - 1.0).abs() > PPO_CLIP).float().mean().item())
                    approx_kls.append(((ratio - 1.0) - logratio).mean().item())  # Schulman k3
                    grad_norms.append(float(gnorm))

        # ---- logging: full reward decomposition so you can see WHICH term dominates ----
        # raw sub-scores (each in [0, 1]), averaged over the batch
        mean_dlg_frac   = float(np.mean([i.dialogue_fraction for i in infos]))
        mean_pair_score = float(np.mean([i.pair_score for i in infos]))
        mean_strand     = float(np.mean([i.stranded_fraction for i in infos]))
        mean_pairs      = float(np.mean([i.num_pairs for i in infos]))
        mean_ent        = float(np.mean(ent_vals))
        # PPO health diagnostics
        mean_clipfrac   = float(np.mean(clip_fracs))    # how often the clip binds
        mean_approx_kl  = float(np.mean(approx_kls))    # policy movement per update
        mean_gradnorm   = float(np.mean(grad_norms))    # pre-clip grad magnitude
        # critic quality: fraction of return variance the value head explains
        with torch.no_grad():
            y = returns.flatten(); var_y = y.var()
            explained_var = float(1.0 - (y - old_values.flatten()).var() / (var_y + 1e-8)) \
                if var_y > 0 else 0.0
        # collapse / reward-hacking detector: distinct generated tokens per completion
        gen = seq[:, prompt_len:]
        mean_distinct   = float(np.mean(
            [len(set(gen[b].tolist())) / gen.size(1) for b in range(gen.size(0))]))
        # weighted contributions — these EXACTLY sum to the task reward total
        dlg_term    = W_DIALOGUE   * mean_dlg_frac      # +
        pair_term   = W_PAIRS      * mean_pair_score    # +
        strand_term = W_UNBALANCED * mean_strand        # − (penalty)
        task_total  = dlg_term + pair_term - strand_term
        # KL leash (computed in the loop; its weight is the live adaptive β)
        mean_kl  = kl.sum(dim=1).mean().item()           # per-completion KL (nats)
        kl_coef  = kl_ctl.value                          # β actually used this iter
        kl_pen   = kl_coef * mean_kl                     # − (penalty)
        net      = task_total - kl_pen                   # the reward PPO actually optimizes

        print(f"iter {it:4d} | total {net:+.3f} = dlg {dlg_term:+.3f} + pair {pair_term:+.3f} "
              f"- strand {strand_term:.3f} - kl {kl_pen:.3f}  |  "
              f"raw[dlg_frac {mean_dlg_frac:.2f} pair {mean_pair_score:.2f} strand {mean_strand:.2f} "
              f"pairs {mean_pairs:.2f} KL {mean_kl:6.2f} β {kl_coef:.4f}]  |  "
              f"pg {np.mean(pg_losses):+.3f} v {np.mean(v_losses):.3f} ent {mean_ent:.2f} "
              f"ev {explained_var:+.2f} clip {mean_clipfrac:.2f} akl {mean_approx_kl:+.4f} "
              f"gn {mean_gradnorm:.2f} dist {mean_distinct:.2f}")
        if use_wandb:
            # Plot titles are the keys below. Rewards and penalties are labelled
            # so each panel is self-explanatory:  total reward = (reward terms) −
            # (penalty terms). Penalties are logged as positive magnitudes.
            wandb.log({
                # headline
                "total reward": net,
                # reward components (positive contributions)
                "reward - dialogue": dlg_term,
                "reward - pairs":    pair_term,
                # penalty components (magnitudes;  total = rewards − penalties)
                "penalty - stranded": strand_term,
                "penalty - kl":       kl_pen,
                # optimization
                "value loss":  float(np.mean(v_losses)),
                "policy loss": float(np.mean(pg_losses)),
                "entropy":     mean_ent,
                # PPO health diagnostics (grouped under "ppo/")
                "ppo/explained_variance": explained_var,  # critic quality (→1 good)
                "ppo/clip_fraction":      mean_clipfrac,   # is PPO_CLIP binding?
                "ppo/approx_kl":          mean_approx_kl,  # policy move per update
                "ppo/grad_norm":          mean_gradnorm,   # pre-clip grad magnitude
                "ppo/distinct_frac":      mean_distinct,   # ↓ = repetition/collapse
                # raw unweighted sub-scores (grouped under a "raw" section)
                "raw/dialogue_fraction": mean_dlg_frac,
                "raw/pair_score":        mean_pair_score,
                "raw/stranded_fraction": mean_strand,
                "raw/valid_pairs":       mean_pairs,
                "raw/kl_nats":           mean_kl,
                "raw/kl_coeff":          kl_coef,
            }, step=it)

        # ---- adaptive KL: nudge β toward holding mean_kl at KL_TARGET ----
        if KL_ADAPTIVE:
            kl_ctl.update(mean_kl, RL_BATCH_SIZE)

        # ---- periodic sample completions → wandb table (eyeball reward hacking) ----
        if use_wandb and ((it + 1) % RL_SAVE_INTERVAL == 0 or it == args.iters - 1):
            comp_rewards = task_rewards.sum(dim=1)
            tbl = wandb.Table(columns=["iter", "task_reward", "completion"])
            for b in range(min(4, seq.size(0))):
                tbl.add_data(it + 1, round(comp_rewards[b].item(), 3),
                             enc.decode(seq[b, prompt_len:].tolist()))
            wandb.log({"samples": tbl}, step=it)

        # ---- checkpoint ----
        if (it + 1) % RL_SAVE_INTERVAL == 0 or it == args.iters - 1:
            ckpt_path = RL_CHECKPOINT_DIR / f"rl_{it + 1:05d}.pt"
            torch.save({
                "step": it + 1, "model": policy.state_dict(),
                "config": cfg, "value_head": value_head.state_dict(),
            }, ckpt_path)
            sample = enc.decode(seq[0].tolist())
            print(f"         saved → {ckpt_path.name}")
            print(f"         sample: {sample!r}")

    print("RL fine-tuning complete.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
