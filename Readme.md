# GPT-2 from Scratch with RL Finetuning

A ground-up implementation of GPT-2, pretrained on the Sherlock Holmes corpus, then finetuned with reinforcement learning.

## Project roadmap

| Phase | Status |
|---|---|
| Data collection & preprocessing | Done |
| GPT-2 architecture implementation | Pending |
| Pretraining | Pending |
| RL finetuning | Pending |

## Dataset

**Sherlock Holmes** — *The Adventures of Sherlock Holmes* (Project Gutenberg, [pg1661](https://www.gutenberg.org/ebooks/1661)).

Split: 90% train / 10% validation, snapped to the nearest newline.

## Project structure

```
.
├── data/
│   ├── train.txt          # ~90% of corpus
│   └── val.txt            # ~10% of corpus
├── src/
│   └── prepare_data.py    # Download & preprocess script
├── checkpoints/           # Saved model weights (git-ignored)
└── Readme.md
```

## Setup

```bash
pip install torch
```

## Usage

### 1. Prepare data

Downloads the corpus, strips Gutenberg boilerplate, and writes `data/train.txt` and `data/val.txt`.

```bash
python src/prepare_data.py
```
