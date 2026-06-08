"""
Gradio demo: base GPT-2 vs. RL-fine-tuned GPT-2, side by side.

Loads two checkpoints automatically (the highest-step one in each folder):
  • LEFT  — base GPT-2 trained on the detective-novel corpus  (checkpoints/)
  • RIGHT — the PPO "dialogify" fine-tune                      (checkpoints_RL/)

Type one prompt and both models complete it with the SAME sampling knobs and the
SAME random seed, so any difference you see is the RL fine-tuning, not luck.
Models are loaded lazily on first generation and cached.
"""

import glob
import os
import re
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import gradio as gr  # noqa: E402
import tiktoken  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from config import CHECKPOINT_DIR, RL_CHECKPOINT_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Tokeniser + device
# ---------------------------------------------------------------------------
_enc = tiktoken.get_encoding("gpt2")
_device = "cuda" if torch.cuda.is_available() else "cpu"

# Cached models, keyed by the checkpoint path they were loaded from.
_cache: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Checkpoint discovery — pick the HIGHEST-step checkpoint in a folder
# ---------------------------------------------------------------------------
def _latest_checkpoint(folder) -> str | None:
    """Return the .pt file with the largest trailing step number (e.g.
    ckpt_11999.pt, rl_02000.pt). Falls back to most-recently modified."""
    candidates = glob.glob(str(folder / "**" / "*.pt"), recursive=True)
    if not candidates:
        return None

    def step_of(path: str) -> int:
        m = re.search(r"(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    best = max(candidates, key=step_of)
    if step_of(best) < 0:                       # no numbered files → use mtime
        best = max(candidates, key=os.path.getmtime)
    return best


def _load_model(ckpt_path: str):
    """Instantiate GPT and load weights, using the config saved in the ckpt."""
    from model import GPT, GPTConfig
    state = torch.load(ckpt_path, map_location=_device, weights_only=False)
    cfg = state.get("config", GPTConfig())      # checkpoints store their own config
    m = GPT(cfg).to(_device)
    m.load_state_dict(state["model"])
    m.eval()
    return m


def _get_model(ckpt_path: str):
    if ckpt_path not in _cache:
        print(f"[app] Loading checkpoint: {ckpt_path}")
        _cache[ckpt_path] = _load_model(ckpt_path)
        print(f"[app] Ready on {_device}.")
    return _cache[ckpt_path]


# ---------------------------------------------------------------------------
# Sampling — temperature / top-k / top-p (nucleus) / seed
# ---------------------------------------------------------------------------
def _sample(model, prompt_ids, max_new_tokens, temperature, top_k, top_p, seed):
    if seed is not None and seed >= 0:
        torch.manual_seed(int(seed))           # same seed → identical noise for both models
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=_device)
    temperature = max(float(temperature), 1e-5)
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            idx_cond = idx[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if top_p and 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = sorted_probs.cumsum(dim=-1)
                # drop tokens once the cumulative mass (before this token) passes top_p,
                # but always keep at least the single most-likely token
                drop = (cum - sorted_probs) > top_p
                sorted_probs[drop] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                nxt = torch.multinomial(sorted_probs, num_samples=1)
                idx_next = sorted_idx.gather(-1, nxt)
            else:
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx[0].tolist()


# ---------------------------------------------------------------------------
# Gradio callback — run BOTH models on the same prompt/knobs/seed
# ---------------------------------------------------------------------------
def generate_both(prompt, max_new_tokens, temperature, seed):
    if not prompt.strip():
        msg = "(Enter a prompt above.)"
        return msg, msg

    base_ckpt = _latest_checkpoint(CHECKPOINT_DIR)
    rl_ckpt = _latest_checkpoint(RL_CHECKPOINT_DIR)
    prompt_ids = _enc.encode(prompt)

    def run(ckpt, missing_hint):
        if ckpt is None:
            return missing_hint
        try:
            model = _get_model(ckpt)
            out = _sample(model, prompt_ids, max_new_tokens, temperature,
                          top_k=0, top_p=1.0, seed=seed)
            return _enc.decode(out[len(prompt_ids):])
        except Exception as exc:                # keep the UI alive on errors
            return f"Error: {exc}"

    base_text = run(base_ckpt, f"No base checkpoint in {CHECKPOINT_DIR}. Run train.py first.")
    rl_text = run(rl_ckpt, f"No RL checkpoint in {RL_CHECKPOINT_DIR}. Run ppo_finetune.py first.")
    return base_text, rl_text


def get_status() -> str:
    base = _latest_checkpoint(CHECKPOINT_DIR)
    rl = _latest_checkpoint(RL_CHECKPOINT_DIR)
    base_s = f"`{os.path.basename(base)}`" if base else "_none_"
    rl_s = f"`{os.path.basename(rl)}`" if rl else "_none_"
    return f"Base: {base_s}  |  RL: {rl_s}  |  Device: `{_device}`"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="GPT-2: Base vs. RL-Dialogify") as app:
    gr.Markdown("# GPT-2 (from scratch): Base vs. RL-Fine-Tuned")
    gr.Markdown(
        "One prompt → two completions. **Left** is the base model (trained on "
        "*The Adventures of Sherlock Holmes*); **right** is the PPO fine-tune that "
        "was rewarded for writing dialogue. Both use the same knobs and seed."
    )

    status_box = gr.Textbox(value=get_status, label="Loaded checkpoints",
                            interactive=False, every=10)

    prompt_box = gr.Textbox(
        lines=3,
        placeholder='e.g. "It was a cold foggy morning when Sherlock Holmes"',
        label="Prompt",
    )

    with gr.Row():
        max_tokens_slider = gr.Slider(10, 500, value=120, step=10, label="Max new tokens")
        temperature_slider = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
        seed_box = gr.Number(value=1337, precision=0, label="Seed  (-1 = random)")

    generate_btn = gr.Button("Generate", variant="primary")

    with gr.Row():
        base_out = gr.Textbox(lines=12, label="Base GPT-2 (detective novels)", interactive=False)
        rl_out = gr.Textbox(lines=12, label="RL fine-tuned GPT-2 (dialogue)", interactive=False)

    generate_btn.click(
        fn=generate_both,
        inputs=[prompt_box, max_tokens_slider, temperature_slider, seed_box],
        outputs=[base_out, rl_out],
    )

    gr.Markdown(
        "_Temperature: 1.0 = diverse, < 0.5 = focused. "
        "Fix the seed to compare the two models fairly._"
    )


if __name__ == "__main__":
    app.launch(share=True)
