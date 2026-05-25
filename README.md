# TARS — Teknik Asistan

Gömülü sistemler, roket aviyoniği ve elektronik konularında uzmanlaşmış yerel yapay zeka teknik asistanı. **Ollama** üzerinde çalışan 7B+ dil modeli, **ChromaDB** tabanlı kalıcı vektör veritabanı ve **hibrit RAG** (Retrieval-Augmented Generation) mimarisiyle donatılmıştır.

---

## Sürüm Geçmişi

| Sürüm | Mimari | Konum |
|---|---|---|
| v2 | QLoRA fine-tuning (Qwen2.5-0.5B) + FAISS RAG | `scripts/Version_2/` |
| v3 | QLoRA + FAISS + BM25 + CrossEncoder RAG | `scripts/Version_3/` |
| **v4** | **Ollama + ChromaDB + Hibrit RAG + Kalıcı Hafıza** | `scripts/Version_4/` |

---

## Proje Yapısı

```
TARS/
├── data_docs/              ← PDF ve DOCX kaynak belgeler
├── JSONL/                  ← İnstruction-format eğitim/bilgi verisi
├── Synthetic_JSONL/        ← Sentetik veri setleri
├── Image/
│   └── TARS.png
├── scripts/
│   ├── Version_2/          ← v2 Jupyter notebook'ları
│   ├── Version_3/          ← v3 Jupyter notebook'u
│   └── Version_4/          ← Aktif sürüm (modüler Python paketi)
│       ├── main.py         ← Başlangıç noktası
│       ├── config.py       ← Tüm ayarlar
│       ├── requirements.txt
│       ├── core/
│       │   ├── embedder.py     ← sentence-transformers embedding
│       │   ├── vector_store.py ← ChromaDB CRUD
│       │   ├── retriever.py    ← Hibrit arama + confidence scoring
│       │   ├── llm.py          ← Ollama API (streaming destekli)
│       │   └── memory.py       ← Kullanıcı öğretme / hafıza yönetimi
│       └── ingestion/
│           ├── loader.py   ← PDF, DOCX, JSONL okuma
│           ├── chunker.py  ← Semantik chunking
│           └── ingest.py   ← Artımlı veri tabanı güncelleme
├── requirements.txt
└── .gitignore
```

---

## Mimari (v4)

```
Kullanıcı Sorusu
       │
       ▼
┌─────────────────────┐
│   Query Embedding   │  ← paraphrase-multilingual-MiniLM-L12-v2
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│           ChromaDB (Kalıcı)             │
│  ┌──────────────────┐  ┌─────────────┐ │
│  │ tars_knowledge   │  │ user_memory │ │
│  │ (JSONL + Docs)   │  │ (öğrenilen) │ │
│  └──────────────────┘  └─────────────┘ │
└─────────┬───────────────────────────────┘
          │  Hibrit: ChromaDB cosine + BM25
          ▼
┌─────────────────────┐
│  Confidence Check   │  skor < 0.40 → "Bilmiyorum"
└─────────┬───────────┘
          │  skor >= 0.40
          ▼
┌─────────────────────┐
│  Ollama LLM         │  ← qwen2.5:7b (streaming)
└─────────┬───────────┘
          │
          ▼
     Cevap + Kaynak Bilgisi
```

---

## Kurulum

### Gereksinimler

- Python 3.10+
- [Ollama](https://ollama.com) kurulu ve çalışıyor olmalı
- 8 GB+ RAM (7B model için)

### 1. Ollama Kurulumu

```bash
# https://ollama.com adresinden indirip kur, ardından:
ollama pull qwen2.5:7b        # Önerilen (~4.7 GB)
# veya daha hafif alternatifler:
ollama pull qwen2.5:3b        # (~2 GB)
ollama pull llama3.2:3b       # (~2 GB)
```

### 2. Python Bağımlılıkları

```bash
cd scripts/Version_4
pip install -r requirements.txt
```

### 3. Bilgi Tabanını Oluştur (İlk Çalıştırma)

```bash
python -m ingestion.ingest
# → "✅ 3278 chunk ChromaDB'ye yüklendi"
```

Bu komut `data_docs/`, `JSONL/` ve `Synthetic_JSONL/` klasörlerini tarar ve ChromaDB'ye yükler. Sonraki çalıştırmalarda yalnızca yeni dosyalar işlenir.

### 4. TARS'ı Başlat

```bash
python main.py
# Farklı model ile:
python main.py --model llama3.2:3b
# Streaming kapalı:
python main.py --no-stream
```

---

## Kullanım

### Sohbet

Programı başlattıktan sonra doğrudan sorunuzu yazın:

```
[1] Siz: UART ile SPI arasındaki fark nedir?

  [📚 Kaynaklar: STM32_rehberi.pdf | Skor: 0.76]

TARS: UART asenkron, SPI ise senkron bir seri iletişim protokolüdür...
```

Bilgi tabanında cevap yoksa:

```
[2] Siz: Kuantum bilgisayar nasıl çalışır?

TARS: Bu konuda bilgim yok.
      (Benzerlik skoru: 0.21 < eşik: 0.40)
      'bunu öğren: [bilgi]' ile bana öğretebilirsin.
```

### TARS'a Bilgi Öğretme

Aşağıdaki ifadelerden biriyle başlayan mesajlar kalıcı kullanıcı hafızasına kaydedilir:

```
bunu öğren: [bilgi]
şunu kaydet: [bilgi]
hatırla: [bilgi]
not et: [bilgi]
bilgi ekle: [bilgi]
bunu bil: [bilgi]
kaydet: [bilgi]
```

Örnek:

```
[3] Siz: bunu öğren: BMP280 sensörü -40°C ile +85°C arasında çalışır.

  ✅ Öğrendim! Bu bilgiyi kalıcı olarak hafızama kaydettim.
```

### Dahili Komutlar

| Komut | Açıklama |
|---|---|
| `/ingest` | Yeni belgeleri tara ve ekle |
| `/ingest --full` | Veri tabanını sıfırla, baştan oluştur |
| `/ingest --clean` | Silinmiş dosyaların chunk'larını temizle |
| `/hafiza` | Öğrendiğim bilgileri listele |
| `/temizle` | Kullanıcı hafızasını sıfırla |
| `/stats` | Veri tabanı istatistiklerini göster |
| `/model <isim>` | Ollama modelini değiştir (örn. `/model llama3.2:3b`) |
| `/gecmis` | Bu oturumun sohbet geçmişini göster |
| `/yardim` | Komut listesini göster |
| `/cikis` | Programı kapat |

---

## Yeni Belge Ekleme

`data_docs/` klasörüne PDF veya DOCX dosyası bırakın, ardından:

```bash
python -m ingestion.ingest
```

Desteklenen formatlar: `.pdf`, `.docx`, `.jsonl`

Sistem yalnızca yeni veya değişmiş dosyaları işler (MD5 hash karşılaştırması).

---

## Ayarlar (`config.py`)

Tüm parametreler `scripts/Version_4/config.py` dosyasından değiştirilebilir:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Kullanılacak Ollama modeli |
| `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding modeli |
| `CONFIDENCE_THRESHOLD` | `0.40` | Bu değerin altında "bilmiyorum" yanıtı |
| `TOP_K` | `8` | ChromaDB + BM25'ten alınacak aday sayısı |
| `FINAL_K` | `4` | LLM'e gönderilecek chunk sayısı |
| `CHUNK_SIZE` | `500` | Maksimum karakter / chunk |
| `CHUNK_OVERLAP` | `80` | Chunk örtüşme miktarı |
| `MAX_HISTORY` | `6` | Sohbet geçmişinde tutulacak tur sayısı |

---

## Ollama Model Seçim Rehberi

| Model | Boyut | RAM | Türkçe | Öneri |
|---|---|---|---|---|
| `qwen2.5:7b` | ~4.7 GB | 8 GB | İyi | **Önerilen** |
| `qwen2.5:3b` | ~2 GB | 4 GB | Orta | Düşük RAM |
| `llama3.2:3b` | ~2 GB | 4 GB | Orta | Alternatif |
| `mistral:7b` | ~4.1 GB | 8 GB | Orta | İngilizce ağırlıklı |

---

## Teknolojiler

- **LLM:** Ollama (qwen2.5:7b veya uyumlu model)
- **Embedding:** sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Vektör Veritabanı:** ChromaDB (kalıcı)
- **Sparse Retrieval:** BM25 (rank-bm25)
- **Belge Okuma:** pypdf, python-docx
- **Önceki Sürümler (v2/v3):** QLoRA (PEFT, TRL, bitsandbytes), FAISS, CrossEncoder
