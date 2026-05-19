"""
Download and preprocess the Sherlock Holmes corpus from Project Gutenberg.
Strips boilerplate, then splits into train (first ~90%) and val (last ~10%).
"""

import re
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
DATA_DIR = Path(__file__).parent.parent / "data"

# Gutenberg boilerplate boundary markers
START_MARKER = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK", re.IGNORECASE)
END_MARKER   = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK",   re.IGNORECASE)


def download(url: str) -> str:
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def strip_boilerplate(raw: str) -> str:
    lines = raw.splitlines()

    start_idx = next(
        (i for i, l in enumerate(lines) if START_MARKER.search(l)), None
    )
    end_idx = next(
        (i for i, l in enumerate(lines) if END_MARKER.search(l)), None
    )

    if start_idx is None or end_idx is None:
        raise ValueError("Could not locate Gutenberg boilerplate markers.")

    # Content lives between the two markers
    body_lines = lines[start_idx + 1 : end_idx]
    return "\n".join(body_lines).strip()


def split_train_val(text: str, val_fraction: float = 0.10):
    chars = len(text)
    split = int(chars * (1 - val_fraction))
    # Snap to the nearest newline so we don't cut mid-sentence
    snap = text.rfind("\n", 0, split)
    if snap == -1:
        snap = split
    return text[:snap], text[snap + 1:]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw = download(URL)
    text = strip_boilerplate(raw)

    train, val = split_train_val(text, val_fraction=0.10)

    train_path = DATA_DIR / "train.txt"
    val_path   = DATA_DIR / "val.txt"

    train_path.write_text(train, encoding="utf-8")
    val_path.write_text(val,   encoding="utf-8")

    print(f"Train: {len(train):,} chars  →  {train_path}")
    print(f"Val:   {len(val):,} chars   →  {val_path}")
    print(f"Val fraction: {len(val) / (len(train) + len(val)):.1%}")


if __name__ == "__main__":
    main()
