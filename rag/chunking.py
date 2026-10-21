"""Memecah dokumen jadi chunk. Strateginya dibedakan per tipe dokumen:

- cv, project, organisasi, transkrip : per section (heading)
- sertifikat                         : satu dokumen (per halaman) = satu chunk
- tipe lain (esai, dll.)             : per paragraf, dikemas sampai MAX_CHUNK_CHARS

Section yang terlalu panjang dipecah lagi per paragraf dengan nama section yang sama.
Chunk tidak pernah menyeberang halaman, jadi nomor halaman di sitasi selalu tepat.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .loaders import Document

SECTION_TYPES = {"cv", "project", "organisasi", "transkrip"}
WHOLE_TYPES = {"sertifikat"}

HEADING_WORDS = (
    "ringkasan", "profil", "summary", "about", "tentang saya", "kontak", "contact",
    "pendidikan", "education", "pengalaman", "experience", "work experience",
    "project", "projects", "proyek", "portofolio", "portfolio",
    "skill", "skills", "keahlian", "kemampuan",
    "organisasi", "organization", "kepanitiaan", "volunteer",
    "sertifikat", "sertifikasi", "certification", "certifications", "pelatihan",
    "penghargaan", "awards", "prestasi", "bahasa", "languages",
    "latar belakang", "tujuan", "metode", "hasil", "teknologi", "peran", "materi",
)
SEMESTER_RE = re.compile(r"^semester\s+\d+", re.I)
UPPER_HEADING_RE = re.compile(r"[A-Z][A-Z &/-]{2,}")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    file: str
    doc_type: str
    title: str
    page: int      # halaman tempat chunk dimulai
    section: str
    text: str

    def search_text(self) -> str:
        # Judul dokumen & nama section ikut di-index supaya chunk pendek tetap punya konteks.
        return f"{self.title} | {self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _heading(line: str) -> str | None:
    s = line.strip()
    if s.startswith("#"):
        return s.lstrip("#").strip() or None
    if not s or len(s) > 40:
        return None
    plain = s.rstrip(":").strip()
    low = plain.lower()
    if SEMESTER_RE.match(low) or any(low == w or low.startswith(w + " ") for w in HEADING_WORDS):
        return plain.title() if plain.isupper() else plain
    # Baris kapital semua tanpa angka, mis. "WORK EXPERIENCE". Baris seperti "IPK 3,62" bukan heading.
    if UPPER_HEADING_RE.fullmatch(plain) and len(plain.split()) <= 4:
        return plain.title()
    return None


Line = tuple[int, str]  # (nomor_halaman, teks_baris)


def _lines(doc: Document) -> list[Line]:
    return [(page_no, line) for page_no, text in enumerate(doc.pages, start=1) for line in text.splitlines()]


def _sections(doc: Document) -> list[tuple[str, list[Line]]]:
    """Kelompokkan baris per heading. Baris sebelum heading pertama masuk section "Header"."""
    out: list[tuple[str, list[Line]]] = []
    name, lines = "Header", []
    for page, line in _lines(doc):
        heading = _heading(line)
        if heading is None:
            lines.append((page, line))
            continue
        if any(text.strip() for _, text in lines):
            out.append((name, lines))
        name, lines = heading, []
    if any(text.strip() for _, text in lines):
        out.append((name, lines))
    return out


def _paragraphs(lines: list[Line]) -> list[Line]:
    """Gabungkan baris jadi paragraf. Baris kosong dan pergantian halaman memutus paragraf."""
    paras: list[Line] = []
    buf: list[str] = []
    page = None
    for line_page, text in lines:
        if buf and (not text.strip() or line_page != page):
            paras.append((page, "\n".join(buf).strip()))
            buf = []
        if text.strip():
            if not buf:
                page = line_page
            buf.append(text)
    if buf:
        paras.append((page, "\n".join(buf).strip()))
    return paras


def _pack(paras: list[Line], max_chars: int) -> list[Line]:
    """Kemas paragraf berurutan sampai max_chars tanpa menyeberang halaman, supaya nomor halaman
    di sitasi tepat. Paragraf yang terlalu panjang dipotong di spasi."""
    pieces: list[Line] = []
    for page, p in paras:
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append((page, p[:cut].strip()))
            p = p[cut:].strip()
        pieces.append((page, p))

    packed: list[Line] = []
    for page, p in pieces:
        if packed and packed[-1][0] == page and len(packed[-1][1]) + len(p) + 2 <= max_chars:
            packed[-1] = (page, packed[-1][1] + "\n\n" + p)
        else:
            packed.append((page, p))
    return packed


def chunk_document(doc: Document, max_chars: int) -> list[Chunk]:
    units: list[tuple[str, int, str]] = []
    if doc.doc_type in SECTION_TYPES:
        for name, lines in _sections(doc):
            units += [(name, page, text) for page, text in _pack(_paragraphs(lines), max_chars)]
    else:
        name = "Dokumen lengkap" if doc.doc_type in WHOLE_TYPES else "Paragraf"
        units += [(name, page, text) for page, text in _pack(_paragraphs(_lines(doc)), max_chars)]

    return [
        Chunk(f"{doc.doc_id}#c{i}", doc.doc_id, doc.file, doc.doc_type, doc.title, page, section, text)
        for i, (section, page, text) in enumerate(units)
    ]


def chunk_documents(docs: list[Document], max_chars: int) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, max_chars)]
