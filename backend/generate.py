"""Grounded answer generation (Phase 4).

Takes a question, retrieves chunks (hybrid by default), and asks Gemini
Flash-Lite to answer USING ONLY those chunks, with inline citations [n] that
map back to source chunks. If the chunks don't contain the answer, the model
is instructed to say so rather than invent one -- the anti-hallucination
property that makes RAG defensible.

The LLM call is isolated here; retrieval is reused unchanged from retriever.py.
API key comes from the GEMINI_API_KEY env var (never hard-coded).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

from retriever import search

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

_SYSTEM = """You are a financial-filings assistant. Answer the user's question \
USING ONLY the numbered context passages provided. Rules:
- Ground every factual claim in the passages and cite the passage number(s) \
inline like [1] or [2][3].
- If the passages do not contain enough information to answer, say exactly: \
"The retrieved documents do not contain enough information to answer this." \
Do not use outside knowledge.
- Be concise and specific. Quote figures exactly as they appear.
- Do not cite passages you did not use."""


@dataclass
class Source:
    n: int
    chunk_id: str
    doc_id: str
    section: str
    score: float


@dataclass
class GroundedAnswer:
    question: str
    answer: str
    sources: list
    mode: str

    def to_dict(self):
        d = asdict(self)
        return d


def _build_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[{i}] (doc: {r['doc_id']}, {r['section']})\n{r['text']}")
    return "\n\n".join(blocks)


def answer(question: str, k: int = 5, mode: str = "hybrid") -> GroundedAnswer:
    from google import genai
    from google.genai import types

    results = search(question, k=k, mode=mode)
    context = _build_context(results)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = f"Context passages:\n{context}\n\nQuestion: {question}"
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            temperature=0.0,
        ),
    )
    text = (resp.text or "").strip()

    sources = [
        Source(n=i, chunk_id=r["chunk_id"], doc_id=r["doc_id"],
               section=r["section"], score=r["score"]).__dict__
        for i, r in enumerate(results, 1)
    ]
    return GroundedAnswer(question=question, answer=text, sources=sources, mode=mode)


if __name__ == "__main__":
    import json
    import sys
    q = " ".join(sys.argv[1:]) or "What capital ratios must American Express maintain?"
    a = answer(q)
    print(json.dumps(a.to_dict(), indent=2))