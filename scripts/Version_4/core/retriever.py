# ─────────────────────────────────────────────────────────────
#  TARS v4 — Retriever
#  ChromaDB cosine + BM25 hibrit arama.
#  Confidence eşiği ile "bilmiyorum" kararı.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
from rank_bm25 import BM25Okapi

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TOP_K, FINAL_K, CONFIDENCE_THRESHOLD
from core import vector_store as vs

# ── BM25 indeksi (session boyunca bellekte tutulur) ───────────
_bm25: BM25Okapi | None = None
_bm25_chunks: list[dict] = []


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(chunks: list[dict]) -> None:
    """
    ingestion sonrası veya ilk retrieval sırasında çağrılır.
    chunks: [{"text": ..., "source": ..., "title": ...}, ...]
    """
    global _bm25, _bm25_chunks
    _bm25_chunks = chunks
    corpus = [_tokenize(c["text"]) for c in chunks]
    _bm25 = BM25Okapi(corpus)


def _bm25_scores(query: str, n: int) -> list[tuple[int, float]]:
    """BM25 ile top-n sonuç döndür: [(indeks, normalize_skor), ...]"""
    if _bm25 is None or not _bm25_chunks:
        return []
    raw = _bm25.get_scores(_tokenize(query))
    top_idx = np.argsort(raw)[::-1][:n]
    max_score = raw.max() + 1e-9
    return [(int(i), float(raw[i]) / max_score) for i in top_idx if raw[i] > 0]


def retrieve(
    query: str,
    top_k: int = TOP_K,
    final_k: int = FINAL_K,
    bm25_weight: float = 0.25,
) -> tuple[list[dict], float]:
    """
    Hibrit retrieval: ChromaDB cosine + BM25.

    Returns:
        (chunks, best_score)
        chunks    : LLM'e gönderilecek metin parçaları
        best_score: En yüksek benzerlik skoru (0–1 arası)
                    Bu değer CONFIDENCE_THRESHOLD ile karşılaştırılır.
    """
    # ── 1. ChromaDB araması (bilgi tabanı + kullanıcı hafızası) ──
    knowledge_hits = vs.query_knowledge(query, n_results=top_k)
    memory_hits    = vs.query_memory(query, n_results=3)

    # Tüm adayları birleştir, text'e göre tekrarları çıkar
    seen_texts: set[str] = set()
    candidates: list[dict] = []
    for hit in knowledge_hits + memory_hits:
        key = hit["text"][:80]
        if key not in seen_texts:
            seen_texts.add(key)
            candidates.append(hit)

    if not candidates:
        return [], 0.0

    # ── 2. BM25 skorunu ekle ──────────────────────────────────────
    bm25_results = _bm25_scores(query, n=top_k)
    bm25_map: dict[str, float] = {}
    for idx, score in bm25_results:
        if idx < len(_bm25_chunks):
            key = _bm25_chunks[idx]["text"][:80]
            bm25_map[key] = score

    # ── 3. Hibrit skor hesapla ────────────────────────────────────
    for c in candidates:
        key = c["text"][:80]
        bm25_s = bm25_map.get(key, 0.0)
        dense_s = c.get("score", 0.0)
        c["hybrid_score"] = round(
            (1 - bm25_weight) * dense_s + bm25_weight * bm25_s, 4
        )

    # ── 4. Hibrit skora göre sırala, top final_k al ───────────────
    ranked = sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)[:final_k]

    best_score = ranked[0]["hybrid_score"] if ranked else 0.0
    return ranked, best_score


def is_confident(score: float, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """Puanın eşiği geçip geçmediğini kontrol et."""
    return score >= threshold
