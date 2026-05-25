# ─────────────────────────────────────────────────────────────
#  TARS v4 — Memory (Kullanıcı Hafızası)
#  Kullanıcının öğrettiği bilgileri yönetir.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TEACHING_TRIGGERS
from core import vector_store as vs


def is_teaching(text: str) -> bool:
    """Kullanıcının bir bilgi öğretip öğretmediğini kontrol et."""
    lower = text.lower()
    return any(lower.startswith(t) or t in lower for t in TEACHING_TRIGGERS)


def extract_fact(text: str) -> str:
    """
    Öğretme ifadesinden asıl bilgiyi çıkar.
    Örn: "bunu öğren: UART sinyali 3.3V'ta çalışır" → "UART sinyali 3.3V'ta çalışır"
    """
    lower = text.lower()
    for trigger in sorted(TEACHING_TRIGGERS, key=len, reverse=True):
        if trigger in lower:
            idx = lower.index(trigger) + len(trigger)
            return text[idx:].strip(" :").strip()
    return text.strip()


def save(raw_input: str) -> str:
    """
    Kullanıcının söylediği bilgiyi hafızaya kaydet.
    Returns: Kullanıcıya gösterilecek onay mesajı.
    """
    fact = extract_fact(raw_input)
    if not fact:
        return "Kaydedilecek bir bilgi bulamadım. 'bunu öğren: [bilgi]' formatını deneyin."

    vs.add_user_memory(fact)
    return f"Öğrendim ve hafızama kaydettim:\n  → \"{fact}\""


def list_memories() -> str:
    """Kayıtlı hafızaları okunabilir formatta döndür."""
    items = vs.list_user_memories()
    if not items:
        return "Henüz öğrendiğim bir bilgi yok."

    lines = [f"Hafızamda {len(items)} kayıt var:\n"]
    for i, item in enumerate(items, 1):
        tarih = f"  [{item['tarih']}]" if item.get("tarih") else ""
        lines.append(f"  {i}. {item['text']}{tarih}")
    return "\n".join(lines)


def clear_memories() -> str:
    """Tüm kullanıcı hafızasını temizle."""
    count = vs.clear_user_memories()
    if count == 0:
        return "Temizlenecek hafıza kaydı yok."
    return f"{count} hafıza kaydı silindi."
