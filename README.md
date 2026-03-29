# 🚀 TARS — Teknik Asistan (LoRA Fine-Tuning)

Bu proje, gömülü sistemler, roket aviyoniği ve elektronik konularında uzmanlaşmış bir yapay zeka teknik asistanı olan **TARS**'ı oluşturmak için geliştirilmiştir. Sistem, **Qwen2.5-0.5B-Instruct** temel modeline **LoRA (Low-Rank Adaptation)** yöntemi kullanılarak fine-tune (ince ayar) işlemi uygular.

## 📂 Proje Dizini

- **`TARS_LoRA_FineTuning.ipynb`**: Modelin veri seti yükleme, kuantizasyon (4-bit QLoRA), ince ayar ve test (sohbet) süreçlerinin yürütüldüğü ana çalışma dosyasıdır.
- **`JSONL/` & `Synthetic_JSONL/`**: Modele öğretilecek instruction, input ve output yapısına sahip Türkçe veri setlerini (JSONL formatında) içeren dizinler.
- **`docs/`**: İlerleyen süreçlerde RAG (Retrieval-Augmented Generation) entegrasyonu için kaynak belgeleri içeren klasör.

## 📦 Kurulum

Projeyi çalıştırmak için tercihen **NVIDIA GPU**'ya (örneğin T4, RTX serisi vb. minimum 16GB VRAM önerilir) sahip bir ortamda (veya Google Colab üzerinde) olmanız tavsiye edilir.

1. **Sanal Ortam Oluşturun ve Aktifleştirin**
   ```bash
   python -m venv .venv
   # Windows için
   .venv\Scripts\activate
   # Linux/Mac için
   # source .venv/bin/activate
   ```

2. **Bağımlılıkları Yükleyin**
   ```bash
   pip install -r requirements.txt
   ```
   *(Eğer yerel sistemde çalışıyorsanız, sisteminizle uyumlu PyTorch sürümünü öncelikle resmi web sitesinden yüklemeniz önerilir.)*

## 🚀 Kullanım Adımları

1. **Veri Hazırlığı:**
   `JSONL/` klasörü içerisine eğitim verilerinizi JSONL formatında ekleyin. Örnek format:
   ```json
   {"instruction": "Sensör nedir?", "input": "", "output": "Fiziksel ortam değişkenlerini ölçen cihaza denir."}
   ```
2. **Eğitim (Fine-Tuning):**
   `TARS_LoRA_FineTuning.ipynb` defterini baştan sona çalıştırın. İhtiyaca göre `EPOCHS`, `BATCH_SIZE`, `LORA_R` gibi hiperparametreleri değiştirebilirsiniz.
3. **Kayıt ve Sohbet:**
   Eğitim tamamlandığında kayıp (loss) grafiği kaydedilir ve LoRA adaptörü diske yazılır. Defterin sonundaki interaktif sohbet hücresinden model ile doğrudan mesajlaşarak sonuçları test edebilirsiniz.

## 🛠️ Teknolojiler ve Konfigürasyonlar

* **Model:** Qwen/Qwen2.5-0.5B-Instruct
* **Eğitim Yöntemi:** QLoRA (4-bit quantization, `nf4`, `bfloat16`)
* **LoRA Parametreleri:** `r=16`, `alpha=32`, Target Modules: `q, k, v, o, gate, up, down_proj`
* **Kütüphaneler:** PyTorch, Transformers, PEFT, TRL, Datasets, Bitsandbytes
