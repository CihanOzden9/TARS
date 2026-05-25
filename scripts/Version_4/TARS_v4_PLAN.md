# TARS v4 — Geliştirme Planı
## Ollama Tabanlı Yerel RAG + Dinamik Hafıza Sistemi

---

## V3'ten V4'e: Neden Değişiyoruz?

| Sorun (V3) | Çözüm (V4) |
|---|---|
| Qwen 1.5B modeli ince ayarlı olsa da teknik sorularda zayıf kalıyor | Ollama ile 7B+ model (yerel, güçlü) |
| Google Colab GPU gereksinimi — her seferinde eğitim döngüsü | Tamamen yerel, GPU opsiyonel (CPU'da da çalışır) |
| Fine-tuning saatler sürüyor, her değişiklikte yeniden gerekli | Eğitim yok — RAG doğrudan bilgi tabanından çeker |
| Kullanıcıdan öğrenme yok | Yeni bilgi öğretildiğinde vektöre dönüştürüp saklar |
| Hallucination kontrolü ROUGE-L ile kaba ölçüm | Cosine similarity eşiği ile net "bilmiyorum" kararı |
| Notebook formatı — modüler değil | Modüler Python paket yapısı |

---

## V4 Mimarisi

```
Kullanıcı Sorusu
       │
       ▼
┌─────────────────────┐
│   Query Embedding   │  ← sentence-transformers (multilingual)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│           ChromaDB (Kalıcı)             │
│  ┌──────────────────┐  ┌─────────────┐ │
│  │ knowledge_base   │  │ user_memory │ │
│  │ (JSONL + Docs)   │  │ (öğrenilen) │ │
│  └──────────────────┘  └─────────────┘ │
└─────────┬───────────────────────────────┘
          │  Top-K chunk + similarity score
          ▼
┌─────────────────────┐
│  Confidence Check   │  score < threshold → "Bilmiyorum"
└─────────┬───────────┘
          │  score >= threshold
          ▼
┌─────────────────────┐
│  Ollama LLM         │  ← qwen2.5:7b veya llama3.2:3b
│  (Yerel Inference)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Cevap + Kaynak     │
└─────────────────────┘
          │
          ▼ (Eğer kullanıcı yeni bilgi öğrettiyse)
┌─────────────────────┐
│  user_memory'e      │
│  Vektörleyip Kaydet │
└─────────────────────┘
```

---

## Dosya Yapısı

```
Version_4/
├── TARS_v4_PLAN.md          ← Bu dosya
├── main.py                  ← Ana giriş noktası (sohbet döngüsü)
├── config.py                ← Tüm ayarlar tek yerde
├── requirements.txt         ← Bağımlılıklar
│
├── core/
│   ├── __init__.py
│   ├── embedder.py          ← Embedding işlemleri
│   ├── vector_store.py      ← ChromaDB CRUD + arama
│   ├── retriever.py         ← Hibrit arama + confidence scoring
│   ├── llm.py               ← Ollama API çağrısı
│   └── memory.py            ← Kullanıcı öğretme / hafıza yönetimi
│
├── ingestion/
│   ├── __init__.py
│   ├── loader.py            ← PDF, DOCX, JSONL okuma
│   ├── chunker.py           ← Semantik chunking (V3'ten iyileştirilmiş)
│   └── ingest.py            ← Veri tabanını ilk kez oluştur/güncelle
│
└── chroma_db/               ← ChromaDB kalıcı depolama (git ignore)
```

---

## Modül Detayları

### 1. `config.py`
```python
# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:7b"        # Ana model
OLLAMA_EMBED    = "nomic-embed-text"  # Opsiyonel: Ollama embedding

# Embedding (sentence-transformers)
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"

# ChromaDB
CHROMA_PATH     = "./chroma_db"
COLLECTION_KNOWLEDGE = "tars_knowledge"   # Statik bilgi tabanı
COLLECTION_MEMORY    = "tars_user_memory" # Kullanıcıdan öğrenilen

# Retrieval
TOP_K           = 6
FINAL_K         = 3
CONFIDENCE_THRESHOLD = 0.45  # Bunun altındaysa "bilmiyorum" de
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 80

# Sohbet
MAX_HISTORY     = 6  # Kaç tur geçmiş tutulsun
```

---

### 2. `core/embedder.py`
- `SentenceTransformer` ile embedding üretir
- Model tek seferinde yüklenir, `@lru_cache` ile tekrar yüklenmez
- Input: `str | list[str]` → Output: `np.ndarray`

---

### 3. `core/vector_store.py`
- `chromadb.PersistentClient` ile kalıcı veritabanı
- İki koleksiyon: `knowledge_base` + `user_memory`
- Metodlar:
  - `add_documents(texts, metadatas, ids)` → toplu ekleme
  - `query(embedding, n_results, collection)` → benzerlik arama
  - `add_user_fact(text, metadata)` → kullanıcı öğretme
  - `get_stats()` → kaç belge var

---

### 4. `core/retriever.py`
**Retrieval stratejisi — V3'ten fark:**
- V3: FAISS + BM25 + CrossEncoder (3 ayrı model, ağır)
- V4: ChromaDB cosine + BM25 hibrit (daha hafif, Ollama'ya VRAM bırak)
- CrossEncoder kaldırılıyor (Ollama 7B zaten daha iyi anlama yapar)

```python
def retrieve(query: str) -> tuple[list[dict], float]:
    """
    Returns: (chunks, best_similarity_score)
    """
    # 1. Her iki koleksiyonu da sorgula
    # 2. Sonuçları birleştir, tekrar edenleri çıkar
    # 3. En yüksek similarity score'u döndür
    # 4. Bu skor confidence check için kullanılır
```

**Confidence Logic:**
```python
if best_score < CONFIDENCE_THRESHOLD:
    return "Bu konuda bilgim yok. Bana öğretmek ister misin?"
```

---

### 5. `core/llm.py`
- Ollama REST API (`http://localhost:11434/api/chat`)
- Streaming desteği (kullanıcı cevabın geldiğini görür)
- Türkçe sistem promptu:
```
Sen TARS adlı teknik bir asistansın.
SADECE verilen kaynaklar temelinde cevap ver.
Kaynaklarda yoksa doğrudan "Bu bilgi elimde yok" de, asla uydurma.
Teknik terimler dışında Türkçe konuş.
```

---

### 6. `core/memory.py`
**Kullanıcı öğretme akışı:**

```
Kullanıcı: "Bunu öğren: [bilgi]"
    ↓
Bilgiyi embed et
    ↓
user_memory koleksiyonuna kaydet (metadata: kaynak="kullanıcı", tarih=now)
    ↓
"Öğrendim! Bu bilgiyi hatırlayacağım." yanıtı
```

**Öğrenme tetikleyicileri (anahtar kelimeler):**
- "bunu öğren:", "şunu kaydet:", "bilgi ekle:", "not et:"
- "hatırla:", "öğren ki", "bil ki"

**Bellek listesi:**
- `memory.py` içinde `list_memories()` → ne öğrenildiğini göster
- `memory.py` içinde `clear_memories()` → kullanıcı belleğini sıfırla

---

### 7. `ingestion/ingest.py` — Artımlı & Otomatik İngestion

**Nasıl çalışır?**

ChromaDB'de her chunk için bir `doc_id` saklanır. Bu ID, dosya adı + içerik hash'inden (MD5) üretilir. Bu sayede hangi dosyanın hangi versiyonunun işlendiği bilinir.

```
doc_id = MD5(dosya_adı + dosya_boyutu + son_değiştirilme_tarihi)
```

**Yeni belge eklendiğinde ne olur?**

```
python -m ingestion.ingest          ← her çalıştırmada otomatik tarar
         │
         ▼
data_docs/ ve JSONL/ klasörlerini tara
         │
         ▼
Her dosya için doc_id üret
         │
         ├── ChromaDB'de var mı? → ATLA (tekrar işleme, hız kazanımı)
         │
         └── YOK → Oku → Chunk'la → Embed et → ChromaDB'ye ekle
                   "✅ yeni_dosya.pdf → 47 chunk eklendi"
```

**Sonuç:** `data_docs/` klasörüne yeni bir PDF veya DOCX bırakıp `python -m ingestion.ingest` komutunu çalıştırman yeterli. Sistem sadece yeni dosyayı işler, var olanları tekrar okumaz.

**Silinen belge ne olur?**
`--clean-deleted` flag'i ile: Klasörde artık olmayan dosyaların chunk'ları ChromaDB'den silinir.

**main.py içinden de tetiklenebilir:**
```
/ingest            → Yeni belgeleri tara ve ekle
/ingest --full     → Tüm veri tabanını sıfırlayıp baştan oluştur
/stats             → Kaç belge, kaç chunk var göster
```

**Desteklenen formatlar:**
- `.pdf` → pypdf ile metin çıkarımı
- `.docx` → python-docx ile metin çıkarımı  
- `.jsonl` → instruction/input/output formatı (mevcut JSONL'ler)

---

### 8. `main.py` — Sohbet Döngüsü

```python
# Pseudo-kod
while True:
    user_input = input("[TARS] > ")

    # Öğretme komutu mu?
    if is_teaching_command(user_input):
        memory.save(user_input)
        print("Öğrendim!")
        continue

    # Özel komutlar
    if user_input == "/hafiza":
        memory.list_memories()
        continue
    if user_input == "/temizle":
        memory.clear_memories()
        continue
    if user_input == "/stats":
        vector_store.get_stats()
        continue

    # Normal soru
    chunks, best_score = retriever.retrieve(user_input)

    if best_score < CONFIDENCE_THRESHOLD:
        print("Bu konuda bilgim yok. Bana öğretmek ister misin?")
        continue

    answer = llm.generate(user_input, chunks, history)
    history.append(user_input, answer)
    print(answer)
```

---

## Kurulum Adımları (Kullanıcı için)

### 1. Ollama Kurulumu
```bash
# https://ollama.com adresinden indir ve kur
ollama pull qwen2.5:7b        # Ana model (~4.7GB)
# veya daha hafif:
ollama pull llama3.2:3b       # (~2GB)
```

### 2. Python Bağımlılıkları
```bash
pip install chromadb sentence-transformers rank-bm25
pip install ollama python-docx pypdf requests
```

### 3. Veri Tabanı Oluşturma (tek seferlik)
```bash
cd scripts/Version_4
python -m ingestion.ingest
# → "✅ 3278 chunk ChromaDB'ye yüklendi"
```

### 4. TARS'ı Başlat
```bash
python main.py
```

---

## Ollama Model Seçim Rehberi

| Model | Boyut | VRAM / RAM | Türkçe | Önerilen Kullanım |
|---|---|---|---|---|
| `qwen2.5:7b` | ~4.7GB | 8GB RAM | İyi | **Önerilen** — güçlü teknik anlama |
| `qwen2.5:3b` | ~2GB | 4GB RAM | Orta | Düşük RAM sistemler |
| `llama3.2:3b` | ~2GB | 4GB RAM | Orta | Alternatif |
| `mistral:7b` | ~4.1GB | 8GB RAM | Orta | İngilizce ağırlıklı kullanım |

> **Not:** Ollama CPU'da da çalışır ama 7B model yavaş olabilir. RAM 8GB+ önerilir.

---

## V3 → V4 Bileşen Karşılaştırması

| Bileşen | V3 | V4 |
|---|---|---|
| LLM | Qwen 1.5B (fine-tuned, GPU gerekli) | Ollama qwen2.5:7b (yerel, CPU/GPU) |
| Fine-tuning | QLoRA, saatler sürer | **Yok** — sadece RAG |
| Vektör DB | FAISS (bellekte, kalıcı değil) | ChromaDB (diske kalıcı) |
| Retrieval | FAISS + BM25 + CrossEncoder | ChromaDB cosine + BM25 (daha hafif) |
| Bilmiyorum | ROUGE-L ile kaba kontrol | Cosine threshold ile net karar |
| Kullanıcı hafızası | Yok | `user_memory` koleksiyonu (kalıcı) |
| Yapı | Tek Jupyter Notebook | Modüler Python paketi |
| Çalışma ortamı | Google Colab | Tamamen yerel |

---

## Geliştirme Aşamaları (Onay Sonrası)

```
Aşama 1 — Altyapı (config + embedder + vector_store)
Aşama 2 — Veri İngestion (loader + chunker + ingest)
Aşama 3 — Retrieval + Confidence Check
Aşama 4 — Ollama LLM entegrasyonu
Aşama 5 — Hafıza / Öğrenme sistemi
Aşama 6 — main.py sohbet döngüsü + özel komutlar
Aşama 7 — Test & ince ayar (threshold, chunk boyutu)
```

---

## Riskler ve Önlemler

| Risk | Önlem |
|---|---|
| Ollama kurulu değilse çalışmaz | `llm.py` başlangıçta bağlantı testi yapar, anlaşılır hata mesajı verir |
| 7B model çok yavaş | Config'den kolayca 3B'ye geçilebilir |
| ChromaDB ilk ingestion uzun sürer | Progress bar + artımlı güncelleme |
| Threshold yanlış ayarlanırsa çok "bilmiyorum" der | Config'den kolayca ayarlanabilir, test scripti eklenecek |
| Türkçe embedding kalitesi | Zaten V3'te kullanılan multilingual model devam ediyor |
