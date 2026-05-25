# ─────────────────────────────────────────────────────────────
#  TARS v4 — Vector Store (ChromaDB)
#  Kalıcı vektör veritabanı. İki koleksiyon:
#    - tars_knowledge   : belgeler + JSONL (statik)
#    - tars_user_memory : kullanıcıdan öğrenilen (dinamik)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CHROMA_PATH, COLLECTION_KNOWLEDGE, COLLECTION_MEMORY
from core.embedder import embed_one


# ── Singleton client ──────────────────────────────────────────
_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _col(name: str):
    """Koleksiyonu al, yoksa oluştur."""
    return _get_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ── Bilgi tabanı işlemleri ────────────────────────────────────

def add_documents(
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    """
    Toplu belge ekle. Zaten var olan ID'ler güncellenir (upsert).
    """
    col = _col(COLLECTION_KNOWLEDGE)
    embeddings = []
    from core.embedder import embed
    import numpy as np
    vecs = embed(texts)
    embeddings = [v.tolist() for v in vecs]

    col.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def delete_documents(ids: list[str]) -> None:
    """Silinmiş dosyaların chunk'larını kaldır."""
    col = _col(COLLECTION_KNOWLEDGE)
    existing = col.get(ids=ids)
    found_ids = existing["ids"]
    if found_ids:
        col.delete(ids=found_ids)


def query_knowledge(
    query_text: str,
    n_results: int = 8,
) -> list[dict]:
    """
    Bilgi tabanında anlamsal arama.

    Returns:
        [{"text": ..., "source": ..., "title": ..., "distance": ...}, ...]
        distance: 0 = mükemmel eşleşme, 2 = hiç alakasız (cosine uzaklığı)
    """
    col = _col(COLLECTION_KNOWLEDGE)
    count = col.count()
    if count == 0:
        return []

    n = min(n_results, count)
    q_emb = embed_one(query_text)

    results = col.query(
        query_embeddings=[q_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "source":   meta.get("source", ""),
            "title":    meta.get("title", ""),
            "distance": round(float(dist), 4),
            "score":    round(1 - float(dist) / 2, 4),  # cosine similarity'ye çevir
        })
    return chunks


def get_all_doc_ids() -> set[str]:
    """Bilgi tabanındaki tüm ID'leri döndür (artımlı güncelleme için)."""
    col = _col(COLLECTION_KNOWLEDGE)
    if col.count() == 0:
        return set()
    result = col.get(include=[])
    return set(result["ids"])


# ── Kullanıcı hafızası işlemleri ──────────────────────────────

def add_user_memory(text: str, source_hint: str = "") -> None:
    """
    Kullanıcının öğrettiği bir bilgiyi hafızaya ekle.
    """
    col = _col(COLLECTION_MEMORY)
    # Benzersiz ID: içerik hash'i
    import hashlib
    uid = "mem_" + hashlib.md5(text.encode()).hexdigest()[:12]
    emb = embed_one(text)
    col.upsert(
        ids=[uid],
        documents=[text],
        embeddings=[emb],
        metadatas=[{
            "source":     "kullanıcı",
            "hint":       source_hint,
            "tarih":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        }],
    )


def query_memory(query_text: str, n_results: int = 3) -> list[dict]:
    """Kullanıcı hafızasında arama."""
    col = _col(COLLECTION_MEMORY)
    count = col.count()
    if count == 0:
        return []

    n = min(n_results, count)
    q_emb = embed_one(query_text)

    results = col.query(
        query_embeddings=[q_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":   doc,
            "source": "kullanıcı hafızası",
            "title":  meta.get("hint", ""),
            "tarih":  meta.get("tarih", ""),
            "score":  round(1 - float(dist) / 2, 4),
        })
    return chunks


def list_user_memories() -> list[dict]:
    """Tüm kullanıcı hafıza kayıtlarını listele."""
    col = _col(COLLECTION_MEMORY)
    if col.count() == 0:
        return []
    result = col.get(include=["documents", "metadatas"])
    items = []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        items.append({
            "text":  doc,
            "tarih": meta.get("tarih", ""),
        })
    return items


def clear_user_memories() -> int:
    """Tüm kullanıcı hafıza kayıtlarını sil. Silinen sayısını döndür."""
    col = _col(COLLECTION_MEMORY)
    count = col.count()
    if count > 0:
        all_ids = col.get(include=[])["ids"]
        col.delete(ids=all_ids)
    return count


# ── İstatistikler ─────────────────────────────────────────────

def get_stats() -> dict[str, Any]:
    return {
        "bilgi_tabanı": _col(COLLECTION_KNOWLEDGE).count(),
        "kullanıcı_hafızası": _col(COLLECTION_MEMORY).count(),
        "chroma_dizini": str(CHROMA_PATH),
    }
