# GPT-2 from Scratch, with RL Fine-Tuning for Dialogue

A decoder-only GPT-2 implemented from scratch in PyTorch — tokeniser, attention,
training loop, and sampling — trained on *The Adventures of Sherlock Holmes*
(Project Gutenberg). The base model learns to write in the style of the detective
novels. A second stage then **RL fine-tunes** that model with PPO, in an
RLHF-style pipeline where the human preference model is replaced by a transparent,
programmatic reward (the "RLVR" / verifiable-reward setup). The reward pushes the
model to write **dialogue** — characters talking inside quotes — rather than plain
narration. The result is two models you can compare side by side: the base
storyteller and the dialogue-tuned fine-tune.

## Installation & Usage

1. Clone the repository and enter it:

   ```bash
   git clone git@github.com:saifahmadgit/GPT2_from_Scratch_with_RL_finetuning.git
   cd GPT2_from_Scratch_with_RL_finetuning
   ```

2. Install the dependencies. This project uses [`uv`](https://docs.astral.sh/uv/);
   it creates the virtual environment and pulls a CUDA 12.4 build of PyTorch:

   ```bash
   uv sync
   ```

3. Prepare the dataset. This downloads the corpus, strips the Project Gutenberg
   boilerplate, and writes the tokenised `train.bin` / `val.bin` files into `data/`:

   ```bash
   uv run python src/prepare_data.py
   ```

4. Train the base GPT-2. Checkpoints are written to `checkpoints/` and metrics are
   logged to Weights & Biases:

   ```bash
   uv run python src/train.py
   ```

   If you run out of GPU memory, lower `BATCH_SIZE` in `src/config.py`.

5. Generate text from a checkpoint. Pass a prompt and pick a checkpoint:

   ```bash
   uv run python src/generate.py -p "It was a cold foggy morning when Sherlock Holmes" -c checkpoints/ckpt_11999.pt
   ```

   Knobs: `-n` (tokens), `-t` (temperature), `-k` (top-k), `--seed`.

6. RL fine-tune the model with PPO to "dialogify" it. Fine-tuned checkpoints are
   written to `checkpoints_RL/`:

   ```bash
   uv run python src/ppo_finetune.py
   ```

   Pass `-c` to choose the base checkpoint and `--iters` for the number of PPO
   iterations. The reward weights and KL settings live in `src/config.py` and
   `src/reward.py`.

## Results

The base model was trained for 12,000 iterations on the Sherlock Holmes corpus,
then RL fine-tuned for 2,000 PPO iterations. The plot below shows the base model's
training loss converging over the run:

![training loss](assets/training_loss.png)

During RL fine-tuning, the total reward rises as the model learns to produce
real (closed) dialogue, while the adaptive KL controller holds the policy close
to the base model so it doesn't forget how to write coherent English:

![rl reward](assets/rl_reward.png)

The effect is clearest when both models complete the *same* prompt with the same
seed. The base model narrates; the fine-tune breaks into dialogue:

![example output](assets/example_output.png)

## Extra: Reinforcement Learning Fine-Tuning

The extra pursuit for this project is the second training stage: taking the
finished base model and improving a *behaviour* with reinforcement learning. The
base model narrates competently but rarely writes dialogue, and there is no easy
supervised dataset for "more dialogue, please" — but we **can** score how much
dialogue a completion contains. That makes it a natural fit for RL: optimise a
reward we can measure rather than a target we can label.

The pipeline is the same one used for RLHF, with the learned human-preference
model swapped for a transparent, programmatic reward (the "RLVR" /
verifiable-reward style). It uses **PPO** with Generalized Advantage Estimation:

- **Actor / policy** — the base GPT-2; its weights are updated.
- **Critic / value head** — a small head on the shared trunk, trained fresh to
  estimate returns for the advantage calculation.
- **Reference** — a frozen copy of the base model, used only for the KL leash.
- **Reward** — defined in `src/reward.py`: a normalised score that rewards text
  inside *closed* quotes and complete back-and-forth exchanges, and penalises
  text stranded inside an unclosed quote (the anti-hacking term).

On top of the standard PPO objective (clipped policy ratio, value loss, and an
entropy bonus for exploration), the run logs a full reward decomposition and a
set of PPO health metrics to Weights & Biases — explained variance of the critic,
clip fraction, per-update KL, gradient norm, and a distinct-token fraction to
catch repetition/collapse — plus periodic sample completions. The two failure
modes that actually came up, and how they were fixed, are described in
[Difficulties & Solutions](#difficulties--solutions) below.

## Web-based GUI

A [Gradio](https://www.gradio.app/) interface lets you compare both models on one
prompt without touching the command line. It automatically loads the latest base
checkpoint and the latest RL checkpoint:

```bash
uv run python src/app.py
```

```text
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxx.gradio.live
```

- The **local URL** works on the machine running the app.
- The **public URL** is a temporary share link you can open from any device.

Type a prompt, set the temperature and seed, and hit **Generate** — the base
GPT-2 and the RL fine-tune complete it side by side, using identical sampling so
any difference is the fine-tuning, not luck.

> On a remote/headless server, use the public `gradio.live` link to reach the UI
> from your own browser.

## Difficulties & Solutions

The hardest part of the RL stage was stopping the model from *gaming* the reward.
An early version rewarded any text inside quotes, so the policy discovered it could
open a single quote and ramble forever to farm reward. The fix was to only count a
quoted span once it is properly **closed** and holds a few characters — unclosed
quotes now drive a penalty instead, turning the cheat into a loss.

The second issue was the KL leash. With a fixed KL coefficient, the policy slowly
drifted away from the base model for the entire run (its KL crept upward without
ever stabilising), degrading fluency. Replacing the fixed coefficient with an
**adaptive KL controller** — which nudges the penalty up when the policy strays
too far and relaxes it when it stays close — holds the model at a target distance
from the base, keeping the text coherent while it learns to write dialogue.
