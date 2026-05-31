"""
Gradio demo for GPT-2 (from-scratch) text generation.

The model is loaded lazily on first inference.  While no checkpoint exists the
UI stays functional — it just returns a clear status message so you can test
the plumbing before training is done.

To plug in your model:
  1. Import it here:   from model import GPT, GPTConfig
  2. Fill in _load_model() to instantiate and load weights.
  3. Fill in _generate() to call model.generate() and decode tokens.
"""

import glob
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import gradio as gr  # noqa: E402
import tiktoken  # noqa: E402
import torch  # noqa: E402

from config import CHECKPOINT_DIR, BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD, VOCAB_SIZE, DROPOUT  # noqa: E402

# ---------------------------------------------------------------------------
# Tokeniser (GPT-2 BPE — same encoding your model will use)
# ---------------------------------------------------------------------------
_enc = tiktoken.get_encoding("gpt2")

# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _find_latest_checkpoint() -> str | None:
    """Return the most-recently modified .pt checkpoint, or None."""
    pattern = str(CHECKPOINT_DIR / "**" / "*.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_model(ckpt_path: str):
    """Instantiate GPT and load weights from *ckpt_path*."""
    from model import GPT, GPTConfig
    cfg = GPTConfig(
        block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT,
    )
    m = GPT(cfg).to(_device)
    state = torch.load(ckpt_path, map_location=_device, weights_only=False)
    m.load_state_dict(state["model"])
    m.eval()
    return m


def _get_model():
    """Load model once and cache globally."""
    global _model
    if _model is None:
        ckpt = _find_latest_checkpoint()
        if ckpt is None:
            return None  # no checkpoint yet — caller handles this
        print(f"[GPT-2] Loading checkpoint: {ckpt}")
        _model = _load_model(ckpt)
        print(f"[GPT-2] Model ready on {_device}.")
    return _model


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> list[int]:
    """Token-by-token autoregressive generation via model.generate()."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=_device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens, temperature=temperature,
                             top_k=top_k if top_k > 0 else None)
    return out[0].tolist()


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------

def generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    if not prompt.strip():
        return "(Enter a prompt above.)"

    # Check for checkpoint
    ckpt = _find_latest_checkpoint()
    if ckpt is None:
        return (
            "No checkpoint found in checkpoints/\n\n"
            "Train the model first, then come back and generate!"
        )

    # Try loading model (catches NotImplementedError during development)
    try:
        model = _get_model()
    except NotImplementedError:
        return (
            "[STUB MODE]  Model class not yet implemented.\n\n"
            f"Checkpoint detected:  {ckpt}\n"
            f"Prompt tokens:        {_enc.encode(prompt)}\n\n"
            "Fill in _load_model() and _generate() in src/app.py to enable real inference."
        )
    except Exception as exc:
        return f"Error loading model: {exc}"

    # Run generation
    try:
        prompt_ids = _enc.encode(prompt)
        output_ids = _generate(model, prompt_ids, max_new_tokens, temperature, top_k)
        new_ids    = output_ids[len(prompt_ids):]
        return _enc.decode(new_ids)
    except NotImplementedError:
        return (
            "[STUB MODE]  Generation loop not yet implemented.\n\n"
            "Model loaded successfully — fill in _generate() in src/app.py."
        )
    except Exception as exc:
        return f"Generation error: {exc}"


def get_status() -> str:
    ckpt = _find_latest_checkpoint()
    if ckpt is None:
        return f"No checkpoint found in `{CHECKPOINT_DIR}`"
    rel = os.path.relpath(ckpt, CHECKPOINT_DIR)
    return f"Checkpoint ready: `checkpoints/{rel}`   |   Device: `{_device}`"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="GPT-2 Text Generation") as app:
    gr.Markdown("# GPT-2 (from scratch) — Text Generation")
    gr.Markdown(
        "Trained on *The Adventures of Sherlock Holmes* (Project Gutenberg). "
        "Enter a prompt and tune the sampling knobs below."
    )

    status_box = gr.Textbox(
        value=get_status,
        label="Model status",
        interactive=False,
        every=10,
    )

    prompt_box = gr.Textbox(
        lines=4,
        placeholder='e.g. "It was a cold foggy morning when Sherlock Holmes..."',
        label="Prompt",
    )

    with gr.Row():
        max_tokens_slider = gr.Slider(
            minimum=10, maximum=500, value=100, step=10,
            label="Max new tokens",
        )
        temperature_slider = gr.Slider(
            minimum=0.1, maximum=2.0, value=0.8, step=0.05,
            label="Temperature",
        )
        top_k_slider = gr.Slider(
            minimum=0, maximum=200, value=50, step=5,
            label="Top-k  (0 = disabled)",
        )

    generate_btn = gr.Button("Generate", variant="primary")

    output_box = gr.Textbox(
        lines=10,
        label="Generated text",
        interactive=False,
    )

    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_box, max_tokens_slider, temperature_slider, top_k_slider],
        outputs=output_box,
    )

    gr.Markdown(
        "_Tip: Temperature → 1.0 = diverse, < 0.5 = focused. "
        "Top-k = 0 samples from the full distribution._"
    )


if __name__ == "__main__":
    app.launch(share=True)
