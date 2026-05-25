# ─────────────────────────────────────────────────────────────
#  TARS v4 — Chunker
#  Cümle sınırlarına saygı duyan semantik metin bölücü.
#  V3'ten alındı, daha temiz hale getirildi.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def semantic_chunk(
    text: str,
    max_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_length: int = 30,
) -> list[str]:
    """
    Metni anlamsal olarak böler. Cümle asla ortadan kesilmez.

    Adımlar:
    1. Metni cümlelere böl (nokta / soru işareti / ünlem ile biter)
    2. Cümleleri max_size'ı aşmadan birleştir
    3. Örtüşme için önceki bloğun son cümlesini yeni bloğa taşı
    4. Çok kısa parçaları at

    Args:
        text      : Bölünecek metin
        max_size  : Maksimum chunk boyutu (karakter)
        overlap   : Örtüşme miktarı (karakter)
        min_length: Bu kadar karakterden kısa chunk'ları at

    Returns:
        Metin parçalarının listesi
    """
    # Satır sonlarını ve fazla boşlukları normalize et
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text).strip()

    # Cümlelere böl
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        # Tek cümle bile max_size'ı aşıyorsa zorla böl
        if sent_len > max_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            # Uzun cümleyi kelime bazında böl
            words = sent.split()
            sub: list[str] = []
            sub_len = 0
            for word in words:
                if sub_len + len(word) + 1 > max_size and sub:
                    chunks.append(" ".join(sub))
                    sub, sub_len = [], 0
                sub.append(word)
                sub_len += len(word) + 1
            if sub:
                chunks.append(" ".join(sub))
            continue

        if current_len + sent_len + 1 <= max_size:
            current.append(sent)
            current_len += sent_len + 1
        else:
            if current:
                chunks.append(" ".join(current))

            # Örtüşme: önceki bloğun son cümlesi yeni bloğa girer
            if current and len(current[-1]) <= overlap:
                carry = current[-1]
                current = [carry, sent]
                current_len = len(carry) + sent_len + 2
            else:
                current = [sent]
                current_len = sent_len

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c) >= min_length]


def chunk_document(doc: dict) -> list[dict]:
    """
    Belge dict'ini chunk'lara böl.

    Args:
        doc: {"title": ..., "text": ..., "source": ...}

    Returns:
        [{"text": ..., "title": ..., "source": ...}, ...]
    """
    parts = semantic_chunk(doc["text"])
    return [
        {
            "text":   part,
            "title":  doc["title"],
            "source": doc["source"],
        }
        for part in parts
    ]
