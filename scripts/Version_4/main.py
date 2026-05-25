#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  TARS v4 — Ana Giriş Noktası
#
#  Başlatma:
#    python main.py
#    python main.py --model llama3.2:3b   (farklı model)
#    python main.py --no-stream           (streaming kapalı)
#
#  Dahili komutlar:
#    /ingest          → Yeni belgeleri tara ve ekle
#    /ingest --full   → Veri tabanını sıfırla, baştan oluştur
#    /hafiza          → Öğrenilen bilgileri listele
#    /temizle         → Kullanıcı hafızasını sil
#    /stats           → Veritabanı istatistikleri
#    /model <isim>    → Ollama modelini değiştir
#    /yardim          → Bu listeyi göster
#    /cikis           → Programı kapat
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import sys
import argparse
from pathlib import Path

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from core import llm, memory, retriever, vector_store as vs
from ingestion.ingest import run_ingestion, _rebuild_bm25


# ─────────────────────────────────────────────────────────────
#  Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────

HELP_TEXT = """
┌─────────────────────────────────────────────────────┐
│  TARS v4 — Komutlar                                 │
├─────────────────────────────────────────────────────┤
│  /ingest            Yeni belge varsa ekle           │
│  /ingest --full     Veri tabanını sıfırla           │
│  /ingest --clean    Silinmiş dosyaları temizle      │
│  /hafiza            Öğrendiğim bilgileri listele    │
│  /temizle           Kullanıcı hafızamı sıfırla      │
│  /stats             Veri tabanı istatistikleri      │
│  /model <isim>      Ollama modelini değiştir        │
│  /gecmis            Bu oturumun sohbet geçmişi      │
│  /yardim            Bu listeyi göster               │
│  /cikis             Programı kapat                  │
├─────────────────────────────────────────────────────┤
│  Öğretmek için:                                     │
│    "bunu öğren: [bilgi]"                            │
│    "hatırla: [bilgi]"                               │
│    "not et: [bilgi]"                                │
└─────────────────────────────────────────────────────┘
"""

BANNER = """
╔══════════════════════════════════════════════════════╗
║   ████████╗ █████╗ ██████╗ ███████╗                 ║
║      ██╔══╝██╔══██╗██╔══██╗██╔════╝                 ║
║      ██║   ███████║██████╔╝███████╗                 ║
║      ██║   ██╔══██║██╔══██╗╚════██║                 ║
║      ██║   ██║  ██║██║  ██║███████║  v4             ║
║      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝                ║
║                                                      ║
║   Teknik Asistan — Ollama + RAG + Kalıcı Hafıza     ║
╚══════════════════════════════════════════════════════╝
"""


def startup_checks() -> bool:
    """Başlangıç kontrolleri. False dönerse çıkış yapılır."""

    # 1. Ollama bağlantısı
    print("  Ollama kontrol ediliyor...", end=" ")
    if not llm.check_ollama():
        print("❌")
        print(f"""
  HATA: Ollama servisine bağlanılamadı ({cfg.OLLAMA_BASE_URL})

  Çözüm:
    1. https://ollama.com adresinden Ollama'yı kur
    2. Terminalde çalıştır: ollama serve
    3. Modeli indir: ollama pull {cfg.OLLAMA_MODEL}
""")
        return False
    print("✅")

    # 2. Model kontrolü
    models = llm.list_models()
    model_short = cfg.OLLAMA_MODEL.split(":")[0]
    model_found = any(model_short in m for m in models)

    if not model_found:
        print(f"  ⚠️  Model '{cfg.OLLAMA_MODEL}' bulunamadı.")
        if models:
            print(f"     Mevcut modeller: {', '.join(models[:5])}")
        print(f"     İndirmek için: ollama pull {cfg.OLLAMA_MODEL}")
        ans = input("  Yine de devam et? (e/h): ").strip().lower()
        if ans != "e":
            return False
    else:
        print(f"  Model: {cfg.OLLAMA_MODEL} ✅")

    # 3. ChromaDB kontrolü
    stats = vs.get_stats()
    chunk_count = stats["bilgi_tabanı"]
    mem_count   = stats["kullanıcı_hafızası"]

    if chunk_count == 0:
        print("\n  ⚠️  Bilgi tabanı boş!")
        print("  İlk çalıştırmada veriyi yüklemeniz gerekiyor.")
        ans = input("  Şimdi veri tabanını oluşturayım mı? (e/h): ").strip().lower()
        if ans == "e":
            run_ingestion(full=False)
        else:
            print("  Daha sonra '/ingest' komutuyla başlatabilirsiniz.")
    else:
        print(f"  Bilgi tabanı: {chunk_count:,} chunk ✅")
        print(f"  Kullanıcı hafızası: {mem_count} kayıt")
        # BM25 indeksini hafızaya yükle
        _rebuild_bm25()

    return True


def handle_command(cmd: str, history: list[dict]) -> bool:
    """
    Dahili komutu işle.

    Returns:
        True → komut işlendi, sohbet döngüsüne dön
        False → çıkış komutu, döngüyü kır
    """
    parts = cmd.strip().split()
    base  = parts[0].lower()

    if base in ("/cikis", "/exit", "/quit", "q"):
        return False

    elif base == "/ingest":
        full  = "--full"  in parts
        clean = "--clean" in parts
        run_ingestion(full=full, clean_deleted=clean)
        _rebuild_bm25()

    elif base == "/hafiza":
        print("\n" + memory.list_memories())

    elif base == "/temizle":
        ans = input("  Tüm öğrendiğim bilgiler silinecek. Emin misin? (e/h): ").strip().lower()
        if ans == "e":
            print("  " + memory.clear_memories())

    elif base == "/stats":
        stats = vs.get_stats()
        print(f"""
  Veri Tabanı Durumu:
    Bilgi tabanı     : {stats['bilgi_tabanı']:,} chunk
    Kullanıcı hafızası: {stats['kullanıcı_hafızası']} kayıt
    ChromaDB dizini  : {stats['chroma_dizini']}
    Aktif model      : {cfg.OLLAMA_MODEL}
""")

    elif base == "/model":
        if len(parts) < 2:
            print(f"  Mevcut model: {cfg.OLLAMA_MODEL}")
            print(f"  Kullanım: /model <model_adı>")
            models = llm.list_models()
            if models:
                print(f"  Yüklü modeller: {', '.join(models)}")
        else:
            new_model = parts[1]
            cfg.OLLAMA_MODEL = new_model
            print(f"  ✅ Model değiştirildi: {new_model}")

    elif base == "/gecmis":
        if not history:
            print("  Bu oturumda henüz sohbet yok.")
        else:
            print(f"\n  Bu oturumda {len(history) // 2} tur konuşma var:")
            for i, msg in enumerate(history):
                role = "Sen" if msg["role"] == "user" else "TARS"
                print(f"  [{role}] {msg['content'][:80]}...")

    elif base == "/yardim":
        print(HELP_TEXT)

    else:
        print(f"  Bilinmeyen komut: {base}. '/yardim' yazarak komutları görebilirsin.")

    return True


# ─────────────────────────────────────────────────────────────
#  Ana sohbet döngüsü
# ─────────────────────────────────────────────────────────────

def chat_loop(stream: bool = True) -> None:
    history: list[dict] = []

    print(HELP_TEXT)
    print("  Hazır! Sorunuzu yazın veya '/yardim' ile komutları görün.\n")

    while True:
        try:
            turn = (len(history) // 2) + 1
            user_input = input(f"[{turn}] Siz: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Görüşmek üzere!")
            break

        if not user_input:
            continue

        # ── Dahili komut ──────────────────────────────────────
        if user_input.startswith("/") or user_input.lower() in ("q", "quit", "exit"):
            should_continue = handle_command(user_input, history)
            if not should_continue:
                print("\n  Görüşmek üzere!")
                break
            continue

        # ── Öğretme komutu ────────────────────────────────────
        if memory.is_teaching(user_input):
            result = memory.save(user_input)
            print(f"\n  ✅ {result}\n")
            continue

        # ── Normal soru → RAG + LLM ───────────────────────────
        chunks, best_score = retriever.retrieve(user_input)

        # Güven kontrolü
        if not retriever.is_confident(best_score):
            print(f"\nTARS: Bu konuda bilgim yok.")
            print(f"      (Benzerlik skoru: {best_score:.2f} < eşik: {cfg.CONFIDENCE_THRESHOLD})")
            print(f"      'bunu öğren: [bilgi]' ile bana öğretebilirsin.\n")
            # Geçmişe "bilmiyorum" yanıtını da kaydet
            history.append({"role": "user",      "content": user_input})
            history.append({"role": "assistant",  "content": "Bu konuda bilgim yok."})
            continue

        # Kaynak bilgisini göster
        sources = list({c["source"] for c in chunks})
        print(f"\n  [📚 Kaynaklar: {', '.join(sources[:3])} | Skor: {best_score:.2f}]")

        # LLM ile cevap üret
        try:
            answer = llm.generate(
                query=user_input,
                chunks=chunks,
                history=history,
                stream=stream,
            )
        except Exception as e:
            print(f"\n  ❌ LLM hatası: {e}")
            print(f"  Ollama çalışıyor mu? 'ollama serve' komutunu kontrol et.\n")
            continue

        # Geçmişe ekle
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant",  "content": answer})

        # Geçmişi kırp (token limitini aşmamak için)
        max_msgs = cfg.MAX_HISTORY * 2
        if len(history) > max_msgs:
            history = history[-max_msgs:]

        print()  # Boş satır


# ─────────────────────────────────────────────────────────────
#  Giriş noktası
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TARS v4 — Teknik Asistan")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Ollama model adı (varsayılan: {cfg.OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Streaming'i kapat (cevap bir bütün olarak gelir)",
    )
    args = parser.parse_args()

    if args.model:
        cfg.OLLAMA_MODEL = args.model

    print(BANNER)
    print(f"  Model: {cfg.OLLAMA_MODEL}")
    print(f"  Eşik : {cfg.CONFIDENCE_THRESHOLD} (bu değerin altında 'bilmiyorum' derim)\n")

    if not startup_checks():
        sys.exit(1)

    print()
    chat_loop(stream=not args.no_stream)


if __name__ == "__main__":
    main()
