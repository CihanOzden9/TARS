# ─────────────────────────────────────────────────────────────
#  TARS v4 — Merkezi Ayar Dosyası
#  Tüm parametreleri buradan değiştirebilirsin.
# ─────────────────────────────────────────────────────────────

from pathlib import Path

# ── Dizinler ──────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent.parent   # TARS/
DOCS_DIR       = BASE_DIR / "data_docs"
JSONL_DIR      = BASE_DIR / "JSONL"
SYNTHETIC_DIR  = BASE_DIR / "Synthetic_JSONL"
CHROMA_PATH    = Path(__file__).parent / "chroma_db"   # Version_4/chroma_db/

# ── Ollama ────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:7b"   # Değiştirmek için: "llama3.2:3b", "mistral:7b"

# ── Embedding ─────────────────────────────────────────────────
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ── ChromaDB Koleksiyonları ───────────────────────────────────
COLLECTION_KNOWLEDGE = "tars_knowledge"    # Statik bilgi tabanı (belgeler + JSONL)
COLLECTION_MEMORY    = "tars_user_memory"  # Kullanıcıdan öğrenilen bilgiler

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE    = 500   # Maksimum karakter
CHUNK_OVERLAP = 80    # Örtüşme miktarı

# ── Retrieval ─────────────────────────────────────────────────
TOP_K               = 8    # ChromaDB + BM25'ten kaç aday alınsın
FINAL_K             = 4    # LLM'e kaç chunk gönderilsin
CONFIDENCE_THRESHOLD = 0.40  # Bu değerin altındaysa "bilmiyorum" de
                              # 0.0–1.0 arası; düşürürsen daha fazla cevap,
                              # yükseltirsen daha az ama daha güvenli cevap

# ── Sohbet ────────────────────────────────────────────────────
MAX_HISTORY = 6   # Bellekte tutulacak maksimum konuşma turu

# ── Sistem Promptu ────────────────────────────────────────────
SYSTEM_PROMPT = """Sen TARS adlı teknik bir asistansın. Gömülü sistemler, roket aviyoniği ve elektronik konularında uzmanlaşmışsın.

KURALLARIN:
1. SADECE sana verilen kaynak metinlerine dayanarak cevap ver.
2. Kaynaklarda olmayan bir bilgiyi kesinlikle uydurma.
3. Eğer cevap kaynaklarda yoksa doğrudan "Bu konuda bilgim yok." de.
4. Teknik terimler (UART, SPI, PWM, vb.) dışında tamamen Türkçe konuş.
5. Cevapların kısa, net ve teknik olsun — gereksiz dolgu kelimeler kullanma.
"""

# ── Öğrenme Tetikleyicileri ───────────────────────────────────
TEACHING_TRIGGERS = [
    "bunu öğren:",
    "şunu kaydet:",
    "bilgi ekle:",
    "not et:",
    "hatırla:",
    "öğren ki",
    "bil ki",
    "bunu bil:",
    "kaydet:",
]
