"""Isian tetap (nama, email, HP, LinkedIn, GitHub) diambil dari CV dengan regex, tanpa LLM."""
from __future__ import annotations

import re

from .loaders import Document

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?:\+62|62|0)8\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{3,5}")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?", re.I)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w-]+/?", re.I)

PATTERNS = {"email": EMAIL_RE, "telepon": PHONE_RE, "linkedin": LINKEDIN_RE, "github": GITHUB_RE}


def extract_profile(docs: list[Document]) -> dict:
    cvs = [d for d in docs if d.doc_type == "cv"]
    if not cvs:
        return {}
    cv = cvs[0]
    profile = {
        # Heuristik: baris pertama CV biasanya nama.
        "nama": {"value": cv.title, "source": {"file": cv.file, "page": 1, "method": "baris pertama CV"}},
    }
    for page_no, text in enumerate(cv.pages, start=1):
        for key, pattern in PATTERNS.items():
            if key in profile:
                continue
            m = pattern.search(text)
            if m:
                profile[key] = {"value": m.group(0).strip(),
                                "source": {"file": cv.file, "page": page_no, "method": "regex"}}
    return profile
