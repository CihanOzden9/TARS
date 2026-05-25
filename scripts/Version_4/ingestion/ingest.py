# TARS v4 - Ingestion Pipeline
# Artimli: sadece yeni/degisen dosyalari isler.
#
# Kullanim:
#   python -m ingestion.ingest
#   python -m ingestion.ingest --full
#   python -m ingestion.ingest --clean

from __future__ import annotations
import hashlib
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DOCS_DIR, JSONL_DIR, SYNTHETIC_DIR
from ingestion.loader import load_all_documents
from ingestion.chunker import chunk_document
from core import vector_store as vs


def _file_hash(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _chunk_id(file_hash: str, chunk_index: int) -> str:
    return f"{file_hash}_{chunk_index:04d}"


def _source_prefix(file_hash: str) -> str:
    return f"{file_hash}_"


def run_ingestion(full: bool = False, clean_deleted: bool = False) -> None:
    print("\n" + "=" * 55)
    print("  TARS v4 - Veri Tabani Guncelleme")
    print("=" * 55)

    if full:
        print("  [!] --full: Mevcut bilgi tabani temizleniyor...")
        existing_ids = vs.get_all_doc_ids()
        if existing_ids:
            vs.delete_documents(list(existing_ids))
        print(f"  Silindi: {len(existing_ids)} kayit")

    print(f"\n  Taranıyor:")
    print(f"    {DOCS_DIR}")
    print(f"    {JSONL_DIR}")
    print(f"    {SYNTHETIC_DIR}")

    all_docs = load_all_documents(
        docs_dir=DOCS_DIR,
        jsonl_dirs=[JSONL_DIR, SYNTHETIC_DIR],
    )

    if not all_docs:
        print("\n  [!] Hic belge bulunamadi.")
        return

    print(f"\n  Toplam kaynak belge: {len(all_docs)}")

    existing_ids = vs.get_all_doc_ids()
    new_chunks_total = 0
    skipped_docs = 0
    processed_prefixes: set = set()

    print("\n  Isleniyor...")

    for doc in all_docs:
        file_path = doc.get("file_path")

        if file_path and file_path.exists():
            fhash = _file_hash(file_path)
        else:
            fhash = hashlib.md5(doc["source"].encode()).hexdigest()[:12]

        prefix = _source_prefix(fhash)

        if not full and any(eid.startswith(prefix) for eid in existing_ids):
            skipped_docs += 1
            processed_prefixes.add(prefix)
            continue

        processed_prefixes.add(prefix)

        doc_clean = {k: v for k, v in doc.items() if k != "file_path"}
        chunks = chunk_document(doc_clean)
        if not chunks:
            continue

        texts     = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"], "title": c["title"]} for c in chunks]
        ids       = [_chunk_id(fhash, i) for i in range(len(chunks))]

        vs.add_documents(texts=texts, metadatas=metadatas, ids=ids)
        new_chunks_total += len(chunks)

    if clean_deleted:
        stale_ids = [
            eid for eid in existing_ids
            if not any(eid.startswith(p) for p in processed_prefixes)
        ]
        if stale_ids:
            vs.delete_documents(stale_ids)
            print(f"\n  Silinmis dosyalardan {len(stale_ids)} chunk temizlendi.")

    _rebuild_bm25()

    stats = vs.get_stats()
    print(f"\n  Tamamlandi!")
    print(f"    Yeni chunk eklendi   : {new_chunks_total}")
    print(f"    Atlanan kaynak       : {skipped_docs} (degismemis)")
    print(f"    Toplam chunk (DB)    : {stats['bilgi_tabanı']}")
    print(f"    Kullanici hafizasi   : {stats['kullanıcı_hafızası']}")
    print("=" * 55 + "\n")


def _rebuild_bm25() -> None:
    try:
        from core.retriever import build_bm25_index
        from core.vector_store import _col, COLLECTION_KNOWLEDGE
        col = _col(COLLECTION_KNOWLEDGE)
        if col.count() == 0:
            return
        result = col.get(include=["documents", "metadatas"])
        chunks = [
            {
                "text":   doc,
                "source": meta.get("source", ""),
                "title":  meta.get("title", ""),
            }
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]
        build_bm25_index(chunks)
        print(f"  BM25 indeksi guncellendi ({len(chunks)} chunk)")
    except Exception as e:
        print(f"  [!] BM25 indeksi olusturulamadi: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TARS v4 - Veri Ingestion")
    parser.add_argument("--full",  action="store_true", help="Sifirdan olustur")
    parser.add_argument("--clean", action="store_true", help="Silinenleri temizle")
    args = parser.parse_args()
    run_ingestion(full=args.full, clean_deleted=args.clean)
