"""Ekstraksi teks per halaman dari PDF (PyMuPDF), Markdown, atau TXT."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

SUPPORTED = {".pdf", ".md", ".txt"}


@dataclass
class Document:
    doc_id: str       # "<doc_type>/<nama_file_tanpa_ekstensi>"
    file: str         # path relatif terhadap DATA_DIR
    doc_type: str     # nama folder level pertama: cv, transkrip, sertifikat, project, ...
    title: str
    pages: list[str]  # teks per halaman; pages[0] = halaman 1


def _pdf_pages(path: Path) -> list[str]:
    # Urutan default mengikuti content stream PDF. Untuk CV dua kolom hasilnya bisa
    # tercampur, jadi cek dengan `python cli.py chunks --doc <doc_id>`.
    with fitz.open(path) as pdf:
        return [page.get_text("text") for page in pdf]


def _title(pages: list[str], fallback: str) -> str:
    for page in pages:
        for line in page.splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                return line[:100]
    return fallback


def load_documents(data_dir: Path) -> tuple[list[Document], list[str]]:
    """Baca semua dokumen di data_dir. Tipe dokumen = nama subfolder pertama."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Folder dokumen tidak ditemukan: {data_dir}")

    docs: list[Document] = []
    warnings: list[str] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        rel = path.relative_to(data_dir)
        doc_type = rel.parts[0] if len(rel.parts) > 1 else "lainnya"

        if path.suffix.lower() == ".pdf":
            pages = _pdf_pages(path)
        else:
            pages = [path.read_text(encoding="utf-8")]

        empty = [i + 1 for i, p in enumerate(pages) if not p.strip()]
        if len(empty) == len(pages):
            warnings.append(f"{rel.as_posix()}: tidak ada teks (kemungkinan hasil scan). "
                            "Tambahkan versi .md/.txt atau jalankan OCR. Dilewati.")
            continue
        if empty:
            warnings.append(f"{rel.as_posix()}: halaman {empty} tanpa teks.")

        docs.append(Document(
            doc_id=f"{doc_type}/{path.stem}",
            file=rel.as_posix(),
            doc_type=doc_type,
            title=_title(pages, path.stem),
            pages=pages,
        ))
    return docs, warnings
