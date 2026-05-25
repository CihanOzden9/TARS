# ─────────────────────────────────────────────────────────────
#  TARS v4 — LLM (Ollama)
#  Ollama REST API üzerinden yerel model çağrısı.
#  Streaming destekli — kullanıcı cevabın geldiğini görür.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, SYSTEM_PROMPT, MAX_HISTORY

try:
    import ollama as _ollama
    _HAS_OLLAMA_LIB = True
except ImportError:
    _HAS_OLLAMA_LIB = False

import requests
import json


# ── Bağlantı kontrolü ─────────────────────────────────────────

def check_ollama() -> bool:
    """Ollama servisinin çalışıp çalışmadığını kontrol et."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    """Ollama'da yüklü modelleri listele."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ── Prompt oluşturma ──────────────────────────────────────────

def _build_messages(
    query: str,
    chunks: list[dict],
    history: list[dict],
) -> list[dict]:
    """
    Ollama chat formatında mesaj listesi oluştur.
    history: [{"role": "user"/"assistant", "content": "..."}, ...]
    """
    # Kaynak metinlerini hazırla
    if chunks:
        context_parts = []
        for i, c in enumerate(chunks, 1):
            src = c.get("source", "bilinmeyen")
            context_parts.append(f"[Kaynak {i} — {src}]\n{c['text']}")
        context = "\n\n".join(context_parts)
        system_with_context = (
            SYSTEM_PROMPT.strip()
            + f"\n\nKULLANILACAK KAYNAKLAR:\n{context}"
        )
    else:
        system_with_context = SYSTEM_PROMPT.strip()

    messages = [{"role": "system", "content": system_with_context}]

    # Sohbet geçmişini ekle (son MAX_HISTORY tur)
    messages.extend(history[-MAX_HISTORY * 2:])

    # Kullanıcının mevcut sorusunu ekle
    messages.append({"role": "user", "content": query})

    return messages


# ── Streaming üretim ──────────────────────────────────────────

def generate(
    query: str,
    chunks: list[dict],
    history: list[dict],
    stream: bool = True,
) -> str:
    """
    Ollama ile cevap üret.

    Args:
        query   : Kullanıcı sorusu
        chunks  : Retrieval'dan gelen kaynak parçaları
        history : Sohbet geçmişi
        stream  : True = cevap karakter karakter yazdırılır

    Returns:
        Tam cevap metni
    """
    messages = _build_messages(query, chunks, history)

    if _HAS_OLLAMA_LIB:
        return _generate_with_lib(messages, stream)
    else:
        return _generate_with_requests(messages, stream)


def _generate_with_lib(messages: list[dict], stream: bool) -> str:
    """ollama Python kütüphanesi ile üretim."""
    client = _ollama.Client(host=OLLAMA_BASE_URL)
    full_response = ""

    if stream:
        print("\nTARS: ", end="", flush=True)
        for chunk in client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
            options={"temperature": 0.3, "repeat_penalty": 1.1},
        ):
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response += token
        print()  # Satır sonu
    else:
        resp = client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=False,
            options={"temperature": 0.3, "repeat_penalty": 1.1},
        )
        full_response = resp["message"]["content"]

    return full_response.strip()


def _generate_with_requests(messages: list[dict], stream: bool) -> str:
    """Fallback: requests ile doğrudan REST API çağrısı."""
    payload = {
        "model":   OLLAMA_MODEL,
        "messages": messages,
        "stream":  stream,
        "options": {"temperature": 0.3, "repeat_penalty": 1.1},
    }

    full_response = ""

    if stream:
        print("\nTARS: ", end="", flush=True)
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    print(token, end="", flush=True)
                    full_response += token
                    if data.get("done"):
                        break
        print()
    else:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={**payload, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        full_response = r.json()["message"]["content"]

    return full_response.strip()
