"""Ingestion: parse EDGAR HTML 10-K -> text -> overlapping chunks with stable IDs.

Phase 1 (HTML variant). EDGAR files 10-Ks as HTML with a real text layer, so
we parse that directly rather than image-PDFs. The job: turn filings into
clean, inspectable chunks carrying enough metadata to (a) cite back to a
source document and section, and (b) serve as stable ground-truth targets for
chunk-level retrieval eval in Phase 5.

Chunk IDs are deterministic: same input + same chunking params always produce
the same IDs, so the eval QA set can reference chunks by ID across re-ingests.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from bs4 import BeautifulSoup

# 10-K item headings we track so each chunk can name its section in a citation.
_ITEM_RE = re.compile(
    r"\bItem\s+(\d{1,2}[A-Z]?)\.?\s*([A-Z][^\n]{0,60})?", re.IGNORECASE
)


@dataclass
class Chunk:
    chunk_id: str          # stable hash-based id
    doc_id: str            # source document stem (e.g. "V_10K")
    text: str
    char_count: int
    word_count: int
    section: str           # nearest preceding 10-K item heading, best-effort

    def to_dict(self) -> dict:
        return asdict(self)


def extract_text(html_path: Path) -> str:
    """Extract clean text from an EDGAR HTML filing.

    Strips script/style, converts non-breaking spaces, drops tags. Tables are
    flattened to whitespace-separated cell text (good enough for retrieval;
    we are not trying to preserve tabular structure here).
    """
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return text


def _normalize(text: str) -> str:
    """Collapse whitespace and strip artifacts that hurt chunking."""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _section_index(text: str) -> list[tuple[int, str]]:
    """Find (char_offset, section_label) for each 10-K item heading in text."""
    marks: list[tuple[int, str]] = []
    for m in _ITEM_RE.finditer(text):
        num = m.group(1).upper()
        title = (m.group(2) or "").strip().rstrip(".")
        label = f"Item {num}" + (f". {title}" if title else "")
        marks.append((m.start(), label))
    return marks


def _section_for(offset: int, marks: list[tuple[int, str]]) -> str:
    """Nearest preceding heading for a given char offset."""
    current = "(front matter)"
    for pos, label in marks:
        if pos <= offset:
            current = label
        else:
            break
    return current


def _chunk_id(doc_id: str, ordinal: int, text: str) -> str:
    """Deterministic id: stable across re-ingestion of the same input."""
    h = hashlib.sha1(f"{doc_id}|{ordinal}|{text}".encode()).hexdigest()
    return f"{doc_id}::{ordinal:04d}::{h[:8]}"


def chunk_document(
    html_path: Path,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    """Parse an HTML filing and split into overlapping character chunks.

    Chunk on characters with a word-boundary backoff so we never split
    mid-word. 1000/150 chars suits dense filing prose and MiniLM's context.
    """
    doc_id = html_path.stem
    stream = _normalize(extract_text(html_path))
    if not stream:
        return []

    marks = _section_index(stream)

    chunks: list[Chunk] = []
    start = 0
    ordinal = 0
    n = len(stream)
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    while start < n:
        end = min(start + chunk_size, n)
        if end < n and not stream[end].isspace():
            boundary = stream.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        text = stream[start:end].strip()
        if text:
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc_id, ordinal, text),
                    doc_id=doc_id,
                    text=text,
                    char_count=len(text),
                    word_count=len(text.split()),
                    section=_section_for(start, marks),
                )
            )
            ordinal += 1
        start += step

    return chunks


def ingest_dir(data_dir: Path, **chunk_kwargs) -> list[Chunk]:
    """Ingest every .htm/.html file in a directory."""
    all_chunks: list[Chunk] = []
    files = sorted(data_dir.glob("*.htm")) + sorted(data_dir.glob("*.html"))
    if not files:
        raise FileNotFoundError(
            f"No .htm/.html files found in {data_dir.resolve()}"
        )
    for f in files:
        doc_chunks = chunk_document(f)
        all_chunks.extend(doc_chunks)
        print(f"  {f.name}: {len(doc_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    import json
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    print(f"Ingesting HTML filings from {data_dir.resolve()}")
    chunks = ingest_dir(data_dir)
    print(f"\nTotal: {len(chunks)} chunks across all documents")

    if chunks:
        words = [c.word_count for c in chunks]
        print(f"Words/chunk  min={min(words)} max={max(words)} "
              f"mean={sum(words) / len(words):.0f}")
        out = data_dir / "chunks.json"
        out.write_text(json.dumps([c.to_dict() for c in chunks], indent=2))
        print(f"Wrote {out.resolve()}")
        print("\n--- sample chunk ---")
        print(json.dumps(chunks[len(chunks) // 2].to_dict(), indent=2)[:800])