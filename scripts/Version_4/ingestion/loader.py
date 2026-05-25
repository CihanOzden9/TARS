# TARS v4 - Loader
# PDF, DOCX ve JSONL dosyalarindan metin cikarimi.

from __future__ import annotations
from pathlib import Path


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        print(f"  [!] PDF okunamadi [{path.name}]: {e}")
        return ""


def read_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"  [!] DOCX okunamadi [{path.name}]: {e}")
        return ""


def read_jsonl(path: Path) -> list:
    import json
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"  [!] JSONL okunamadi [{path.name}]: {e}")
    return records


def jsonl_to_document(record: dict, source: str = "jsonl") -> dict:
    instruction = record.get("instruction", "")
    inp         = record.get("input", "").strip()
    output      = record.get("output", "")

    if inp:
        text = f"Soru: {instruction}\nBagalam: {inp}\nCevap: {output}"
    else:
        text = f"Soru: {instruction}\nCevap: {output}"

    return {
        "title":  instruction[:60],
        "text":   text,
        "source": source,
    }


def load_all_documents(docs_dir: Path, jsonl_dirs: list) -> list:
    """
    Tum kaynaklardan belgeleri yukle.
    Returns: [{"title": ..., "text": ..., "source": ..., "file_path": Path}, ...]
    file_path: Artimli guncelleme icin dosya hash'i hesaplamada kullanilir.
    """
    documents = []

    # PDF
    for path in sorted(docs_dir.glob("**/*.pdf")):
        text = read_pdf(path)
        if text.strip():
            documents.append({
                "title":     path.stem[:60],
                "text":      text,
                "source":    path.name,
                "file_path": path,
            })
            print(f"  PDF  : {path.name} ({len(text):,} kar.)")

    # DOCX
    for path in sorted(docs_dir.glob("**/*.docx")):
        text = read_docx(path)
        if text.strip():
            documents.append({
                "title":     path.stem[:60],
                "text":      text,
                "source":    path.name,
                "file_path": path,
            })
            print(f"  DOCX : {path.name} ({len(text):,} kar.)")

    # JSONL - her dosya tum kayitlariyla ayni file_path hash'ini paylasir
    for jsonl_dir in jsonl_dirs:
        if not jsonl_dir.exists():
            continue
        for path in sorted(jsonl_dir.glob("*.jsonl")):
            records = read_jsonl(path)
            for rec in records:
                doc = jsonl_to_document(rec, source=path.name)
                if doc["text"].strip():
                    doc["file_path"] = path
                    documents.append(doc)
            if records:
                print(f"  JSONL: {path.name} ({len(records)} kayit)")

    return documents
