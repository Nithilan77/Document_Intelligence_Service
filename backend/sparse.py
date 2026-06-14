"""BM25 sparse index (Phase 3, + toggleable stopword filtering).

Keyword retrieval to complement dense embeddings. BM25 excels where dense is
weak: exact terms, defined phrases, numbers, named entities ("interchange",
"cardmember", "Item 7A").

Stopword filtering is toggleable. The smoke test suggested BM25 was matching
on non-distinctive words ("the company faces risks"), so we filter a small
English stopword set to push weight onto distinctive terms. Whether this
actually helps is left for the Phase 5 eval to MEASURE, not assumed -- the
build_bm25(use_stopwords=...) flag lets us index both ways and compare.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BM25_PATH = DATA_DIR / "bm25.pkl"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Compact English stopword set. Deliberately small and standard -- we are not
# hand-picking domain words (that would be overfitting to our questions).
_STOPWORDS = frozenset("""
a an and are as at be by for from has have in is it its of on or that the to
was were will with we our us you your they their this these those then than
been being do does did but not no nor so such can could would should may might
must shall into over under about above below between through during company
""".split())


def tokenize(text: str, use_stopwords: bool = True) -> list[str]:
    """Lowercase, split on non-alphanumerics, optionally drop stopwords.

    Numbers are kept (they matter in 10-Ks). Stopword removal is applied
    identically at index and query time so the two stay consistent.
    """
    toks = _TOKEN_RE.findall(text.lower())
    if use_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


def build_bm25(data_dir: Path = DATA_DIR, use_stopwords: bool = True) -> None:
    chunks = json.loads((data_dir / "chunks.json").read_text())
    print(f"Loaded {len(chunks)} chunks (stopwords={'on' if use_stopwords else 'off'})")
    corpus_tokens = [tokenize(c["text"], use_stopwords) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "use_stopwords": use_stopwords,
        }, f)
    print(f"Built BM25 over {len(corpus_tokens)} docs -> {BM25_PATH.name}")


if __name__ == "__main__":
    import sys
    use_sw = not (len(sys.argv) > 1 and sys.argv[1] == "--no-stopwords")
    build_bm25(use_stopwords=use_sw)