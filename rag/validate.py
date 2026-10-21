"""Validasi jawaban: sumber harus berasal dari chunk yang di-retrieve, isi harus didukung teks sumber,
jawaban harus membahas hal yang ditanyakan (cek relevansi oleh LLM), dan format isian tertentu
(email, IPK, telepon) harus valid.

Grounding di sini heuristik (substring / overlap token), bukan bukti kebenaran. Ambangnya perlu dievaluasi.
"""
from __future__ import annotations

import re

from . import llm
from .bm25 import tokenize
from .config import Settings
from .profile import EMAIL_RE, PHONE_RE
from .prompts import RELEVANCE_SCHEMA, RELEVANCE_SYSTEM, build_relevance_prompt

SUPPORT_THRESHOLD = {"short": 0.8, "long": 0.6}


def _normalize(text: str) -> str:
    text = re.sub(r"(\d),(\d)", r"\1.\2", text.lower())  # 3,62 -> 3.62
    return re.sub(r"\s+", " ", text).strip()


def support_ratio(answer: str, chunks: list[dict]) -> float:
    """1.0 jika jawaban muncul utuh di sumber; selain itu proporsi token jawaban yang ada di sumber."""
    a = _normalize(answer)
    source = " ".join(_normalize(c["text"]) for c in chunks)
    if not a or not source:
        return 0.0
    if a in source:
        return 1.0
    tokens = tokenize(a)
    if not tokens:
        return 0.0
    source_tokens = set(tokenize(source))
    return sum(t in source_tokens for t in tokens) / len(tokens)


def check_relevance(question: str, answer_text: str, settings: Settings) -> dict:
    """Tanya LLM apakah jawaban menjawab hal yang ditanyakan (bukan hal lain yang mirip).

    Melengkapi support_ratio: jawaban bisa didukung sumber tapi menjawab pertanyaan lain,
    mis. ditanya skor tes A dijawab skor tes B. Kalau output pemeriksa tidak terbaca,
    `relevant` bernilai None dan status jawaban tidak diubah.
    """
    raw = llm.chat_json(RELEVANCE_SYSTEM, build_relevance_prompt(question, answer_text), RELEVANCE_SCHEMA,
                        model=settings.relevance_model or None)
    verdict = raw.get("relevant")
    return {"relevant": verdict if isinstance(verdict, bool) else None,
            "reason": str(raw.get("reason") or "")[:300]}


def _ipk_ok(value: str) -> bool:
    m = re.search(r"\d+(?:[.,]\d+)?", value)
    return bool(m) and 0.0 <= float(m.group(0).replace(",", ".")) <= 4.0


FORMAT_CHECKS = {
    "email": lambda v: bool(EMAIL_RE.fullmatch(v.strip())),
    "telepon": lambda v: bool(PHONE_RE.search(v)),
    "ipk": _ipk_ok,
}


def check_format(fmt: str, value: str) -> bool:
    check = FORMAT_CHECKS.get(fmt)
    return check(value) if check else True
