"""
Download and preprocess a corpus of classic detective / mystery novels from
Project Gutenberg.  Each book's boilerplate is stripped, then everything is
concatenated and split into train (first ~90%) and val (last ~10%).

Downloading is fault-tolerant: any book that fails (404, network error, missing
boilerplate markers) is skipped with a warning rather than aborting the run.

Run:  python src/prepare_data.py
"""

import re
import time
import urllib.request
import urllib.error
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Public-domain detective / mystery fiction (Project Gutenberg ebook IDs).
# (id, "Title — Author")
BOOKS = [
    (1661, "The Adventures of Sherlock Holmes — Doyle"),
    (2097, "The Sign of the Four — Doyle"),
    (244,  "A Study in Scarlet — Doyle"),
    (834,  "The Memoirs of Sherlock Holmes — Doyle"),
    (2350, "The Hound of the Baskervilles — Doyle"),
    (108,  "The Return of Sherlock Holmes — Doyle"),
    (3289, "The Valley of Fear — Doyle"),
    (155,  "The Moonstone — Wilkie Collins"),
    (583,  "The Woman in White — Wilkie Collins"),
    (204,  "The Innocence of Father Brown — Chesterton"),
    (223,  "The Wisdom of Father Brown — Chesterton"),
    (863,  "The Mysterious Affair at Styles — Christie"),
    (434,  "The Circular Staircase — Mary Roberts Rinehart"),
    (4047, "The Leavenworth Case — Anna Katharine Green"),
    (1685, "The Mystery of the Yellow Room — Gaston Leroux"),
    (706,  "The Amateur Cracksman (Raffles) — E. W. Hornung"),
    (6133, "The Extraordinary Adventures of Arsène Lupin — Leblanc"),
]

# Gutenberg boilerplate boundary markers (handle both "THE" and "THIS" wording).
START_MARKER = re.compile(r"\*\*\* START OF TH(E|IS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
END_MARKER   = re.compile(r"\*\*\* END OF TH(E|IS) PROJECT GUTENBERG EBOOK",   re.IGNORECASE)


def download(book_id: int) -> str | None:
    """Fetch a book's plain-text from Gutenberg; return None on failure."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (data-prep script)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  ! download failed for {book_id}: {e}")
        return None


def strip_boilerplate(raw: str) -> str | None:
    """Return the text between the Gutenberg START/END markers, or None."""
    lines = raw.splitlines()
    start_idx = next((i for i, l in enumerate(lines) if START_MARKER.search(l)), None)
    end_idx   = next((i for i, l in enumerate(lines) if END_MARKER.search(l)), None)
    if start_idx is None or end_idx is None or end_idx <= start_idx:
        return None
    return "\n".join(lines[start_idx + 1 : end_idx]).strip()


def split_train_val(text: str, val_fraction: float = 0.10):
    split = int(len(text) * (1 - val_fraction))
    snap = text.rfind("\n", 0, split)  # snap to a newline so we don't cut mid-line
    if snap == -1:
        snap = split
    return text[:snap], text[snap + 1:]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    texts = []
    ok, skipped = 0, 0
    for book_id, title in BOOKS:
        print(f"Downloading [{book_id}] {title} ...")
        raw = download(book_id)
        if raw is None:
            skipped += 1
            continue
        body = strip_boilerplate(raw)
        if not body:
            print(f"  ! could not find boilerplate markers for {book_id}, skipping")
            skipped += 1
            continue
        texts.append(body)
        ok += 1
        print(f"  ok — {len(body):,} chars")
        time.sleep(1)  # be polite to Gutenberg's servers

    if not texts:
        raise SystemExit("No books downloaded — check your network connection.")

    corpus = "\n\n".join(texts)
    train, val = split_train_val(corpus, val_fraction=0.10)

    (DATA_DIR / "train.txt").write_text(train, encoding="utf-8")
    (DATA_DIR / "val.txt").write_text(val, encoding="utf-8")

    # Remove stale tokenized caches so train.py re-tokenizes the new corpus.
    for stale in ("train.bin", "val.bin"):
        p = DATA_DIR / stale
        if p.exists():
            p.unlink()
            print(f"Removed stale cache: {p.name}")

    print()
    print(f"Books downloaded: {ok}   skipped: {skipped}")
    print(f"Corpus: {len(corpus):,} chars")
    print(f"Train:  {len(train):,} chars  →  {DATA_DIR / 'train.txt'}")
    print(f"Val:    {len(val):,} chars   →  {DATA_DIR / 'val.txt'}")
    print(f"Val fraction: {len(val) / len(corpus):.1%}")


if __name__ == "__main__":
    main()
