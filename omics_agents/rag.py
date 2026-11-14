"""Tiny retrieval-augmented generation (RAG) backend: TF-IDF over curated markdown notes.

Real systems (BioMaster, Biomni, GeneAgent) retrieve from large databases / literature with
embeddings. A TF-IDF index over ~15 notes is enough to show the mechanism: the agent must
*retrieve* evidence and *cite* it instead of relying on (possibly hallucinated) parametric memory.
"""
from __future__ import annotations

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import KNOWLEDGE_DIR

MIN_SCORE = 0.08  # below this we say "nothing found" rather than return noise


def _normalise(text: str) -> str:
    # make "hsa-mir-17", "miR-17" and "mir17" all produce the token "mir17"
    t = text.lower().replace("hsa-", "")
    t = t.replace("mir-", "mir").replace("let-", "let")
    return t


@lru_cache(maxsize=1)
def _index():
    docs = sorted(KNOWLEDGE_DIR.glob("*.md"))
    texts = [p.read_text(encoding="utf-8") for p in docs]
    vec = TfidfVectorizer(
        preprocessor=_normalise,
        token_pattern=r"[a-z0-9][a-z0-9_\-]*",
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(texts)
    return docs, texts, vec, matrix


def search_knowledge(query: str, top_k: int = 2) -> dict:
    """Retrieve the most relevant knowledge-base notes for a query."""
    docs, texts, vec, matrix = _index()
    scores = cosine_similarity(vec.transform([query]), matrix).ravel()
    order = scores.argsort()[::-1][: max(1, min(int(top_k), 4))]
    hits = [
        {"source": docs[i].stem, "score": round(float(scores[i]), 3), "text": texts[i]}
        for i in order
        if scores[i] >= MIN_SCORE
    ]
    if not hits:
        return {"query": query, "results": [], "note": "No relevant entry in the knowledge base."}
    return {"query": query, "results": hits}


SEARCH_SCHEMA = {
    "description": "Search the curated breast-cancer / multi-omics knowledge base. Returns notes "
    "with a 'source' id you must cite as [source]. Returns no results if nothing relevant exists.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Gene, miRNA, protein or topic"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 4},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}
