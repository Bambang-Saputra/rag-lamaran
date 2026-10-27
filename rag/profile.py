"""Isian tetap (nama, nama depan/belakang, email, HP, LinkedIn, GitHub, website, alamat) tanpa LLM: dari CV dengan
regex, ditambah file `profil.json` opsional untuk isian yang tidak ada (atau tidak tepat) di CV."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .loaders import Document

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Termasuk format "(+62) 812-..." dan "+62 812-...", yang sering dipakai di CV.
PHONE_RE = re.compile(r"(?:\(\+?62\)[\s.-]?|\+62[\s.-]?|62|0)8\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{3,5}")
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


def split_name(name: str) -> tuple[str, str]:
    """Pisahkan nama untuk form yang meminta First name / Last name: kata terakhir jadi nama belakang, sisanya
    nama depan ("Raka Adi Pratama" -> "Raka Adi", "Pratama"). Nama satu kata tidak punya nama
    belakang."""
    words = name.split()
    if len(words) < 2:
        return name.strip(), ""
    return " ".join(words[:-1]), words[-1]


def add_name_parts(profile: dict) -> None:
    """Tambahkan nama_depan dan nama_belakang dari nama lengkap. Kalau salah satunya sudah diisi di profil.json,
    tidak ada yang diturunkan, supaya pembagian manual tidak tercampur dengan hasil tebakan."""
    if "nama" not in profile or "nama_depan" in profile or "nama_belakang" in profile:
        return
    first, last = split_name(profile["nama"]["value"])
    source = {**profile["nama"]["source"], "method": "dari nama lengkap (kata terakhir = nama belakang)"}
    for key, value in (("nama_depan", first), ("nama_belakang", last)):
        if value:
            profile[key] = {"value": value, "source": dict(source)}


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


# File di akar DATA_DIR (bukan di subfolder tipe dokumen, jadi tidak ikut di-index). Isinya objek JSON, mis.
# {"alamat": "Jl. ...", "telepon": "0812..."}. Nilainya menang atas hasil regex dari CV, karena diisi sendiri.
# Kunci yang diawali "_" dianggap catatan.
MANUAL_FILE = "profil.json"


def load_manual_profile(data_dir: Path) -> tuple[dict, list[str]]:
    path = data_dir / MANUAL_FILE
    if not path.exists():
        return {}, []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {}, [f"{MANUAL_FILE}: bukan JSON yang valid ({e}). Dilewati."]
    if not isinstance(raw, dict):
        return {}, [f"{MANUAL_FILE}: isinya harus objek JSON, mis. {{\"alamat\": \"...\"}}. Dilewati."]
    profile, warnings = {}, []
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, str):
            warnings.append(f"{MANUAL_FILE}: nilai '{key}' harus teks. Dilewati.")
        elif value.strip():
            profile[key] = {"value": value.strip(), "source": {"file": MANUAL_FILE, "page": None, "method": "diisi manual"}}
    return profile, warnings
