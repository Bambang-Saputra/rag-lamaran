"""Isian tetap (nama, email, HP, LinkedIn, GitHub) diambil dari CV dengan regex, tanpa LLM."""
from __future__ import annotations

import re

from .loaders import Document

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?:\+62|62|0)8\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{3,5}")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?", re.I)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w-]+/?", re.I)

PATTERNS = {"email": EMAIL_RE, "telepon": PHONE_RE, "linkedin": LINKEDIN_RE, "github": GITHUB_RE}

# Judul dokumen yang sering muncul di baris pertama CV, sebelum nama.
CV_TITLE_RE = re.compile(r"^(curriculum vitae|cv|resume|résumé|daftar riwayat hidup|riwayat hidup|biodata"
                         r"|data pribadi|profil|profile|personal (information|details|data))$", re.I)
NAME_LABEL_RE = re.compile(r"^nama(\s+lengkap)?\s*[:\-]\s*(?P<name>.+)$", re.I)
NAME_LINE_RE = re.compile(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ.'\- ]{1,48}$")
NAME_SCAN_LINES = 10


def _clean(line: str) -> str:
    return line.strip().lstrip("#").strip().strip("*_").strip()


def _format_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    return name.title() if name.isupper() else name


def extract_name(text: str) -> tuple[str, str] | None:
    """Cari nama di bagian atas CV. Kembalikan (nama, cara) atau None.

    Urutan: label "Nama:" di mana pun pada baris-baris awal, lalu baris pertama yang terlihat seperti nama
    (2-5 kata huruf, tanpa angka, email, atau URL), setelah melewati judul seperti "Curriculum Vitae".
    """
    lines = [_clean(line) for line in text.splitlines()]
    lines = [line for line in lines if line][:NAME_SCAN_LINES]
    for line in lines:
        m = NAME_LABEL_RE.match(line)
        if m:
            return _format_name(m.group("name")), "label Nama di CV"
    for line in lines:
        if CV_TITLE_RE.match(line.rstrip(":").strip()):
            continue
        if NAME_LINE_RE.match(line) and 2 <= len(line.split()) <= 5:
            return _format_name(line), "baris nama di awal CV"
    return None


def extract_profile(docs: list[Document]) -> dict:
    cvs = [d for d in docs if d.doc_type == "cv"]
    if not cvs:
        return {}
    cv = cvs[0]
    profile = {}
    found = extract_name(cv.pages[0]) if cv.pages else None
    if found:
        profile["nama"] = {"value": found[0], "source": {"file": cv.file, "page": 1, "method": found[1]}}
    for page_no, text in enumerate(cv.pages, start=1):
        for key, pattern in PATTERNS.items():
            if key in profile:
                continue
            m = pattern.search(text)
            if m:
                profile[key] = {"value": m.group(0).strip(),
                                "source": {"file": cv.file, "page": page_no, "method": "regex"}}
    return profile
