# ─────────────────────────────────────────────────────────────
#  TARS v4 — Embedder
#  Metin → vektör dönüşümü. Model tek seferinde yüklenir.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EMBED_MODEL


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Model ilk çağrıda yüklenir, sonraki çağrılarda önbellekten gelir."""
    print(f"  [Embedder] Model yükleniyor: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"  [Embedder] Hazır.")
    return model


def embed(texts: str | list[str], batch_size: int = 64) -> np.ndarray:
    """
    Tek metin veya liste → normalize float32 numpy dizisi.

    Args:
        texts: Embed edilecek metin(ler)
        batch_size: Toplu işlem için batch boyutu

    Returns:
        shape (n, dim) numpy array — cosine similarity için normalize edilmiş
    """
    model = _get_model()
    if isinstance(texts, str):
        texts = [texts]
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)


def embed_one(text: str) -> list[float]:
    """Tek metin → Python list (ChromaDB'ye doğrudan verilebilir)."""
    return embed(text)[0].tolist()
