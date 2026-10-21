"""Isi form per field. Tipe field (dari config form):

- fixed   : ambil dari profile.json (regex), tanpa LLM
- factual : RAG, jawaban singkat
- open    : RAG, jawaban paragraf; posisi yang dilamar ikut dipakai
- manual  : tidak dijawab sistem (kesediaan, gaji, jadwal); pelamar mengisi sendiri

Cek relevansi dijalankan setelah semua field selesai di-generate. Kalau diselipkan di antara field,
jawaban field berikutnya bisa berubah (terukur di evaluasi, lihat README bagian Cek relevansi).
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from .config import ROOT, get_settings
from .index import Index
from .pipeline import answer, run_relevance_check
from .validate import check_format

DEFAULT_FORM = ROOT / "forms" / "magang_form.json"


def load_form(path: Path = DEFAULT_FORM) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed(field: dict, profile: dict) -> dict:
    entry = profile.get(field.get("profile_key", field["name"]))
    if not entry:
        return {"value": "", "status": "not_found", "sources": [], "flags": ["tidak_ada_di_profil"]}
    return {"value": entry["value"], "status": "ok", "sources": [entry["source"]], "flags": []}


MANUAL_RESULT = {"value": "", "status": "not_found", "sources": [], "flags": ["isi_sendiri"]}


def _rag_answer(field: dict, index: Index, position: str, job_description: str, settings) -> dict:
    question = field["question"]
    retrieval_query = question
    task_context = ""
    if field["kind"] == "open" and position:
        retrieval_query = f"{question} {position}"
        task_context = f"Posisi: {position}\n{job_description[:1500]}".strip()
    return answer(question, index, doc_types=field.get("doc_types"),
                  style="short" if field["kind"] == "factual" else "long",
                  task_context=task_context, retrieval_query=retrieval_query, settings=settings)


def _rag_result(out: dict) -> dict:
    return {
        "value": out["answer"],
        "status": out["status"],
        "sources": out["sources"],
        "flags": out["flags"],
        "support": out["support"],
        "relevance": out["relevance"],
        "retrieved": [{k: r[k] for k in ("rank", "chunk_id", "section", "scores")} for r in out["retrieved"]],
        "timing_ms": out["timing_ms"],
    }


def autofill(form: dict, index: Index, profile: dict, *, position: str = "",
             job_description: str = "") -> dict:
    s = get_settings()
    # Tahap 1: generate semua field RAG tanpa cek relevansi.
    rag_outputs = {field["name"]: _rag_answer(field, index, position, job_description,
                                              replace(s, relevance_check=False))
                   for field in form["fields"] if field["kind"] not in ("fixed", "manual")}
    # Tahap 2: cek relevansi untuk semua jawaban sekaligus.
    if s.relevance_check:
        for out in rag_outputs.values():
            run_relevance_check(out, s)

    results = {}
    for field in form["fields"]:
        name = field["name"]
        if field["kind"] == "fixed":
            res = _fixed(field, profile)
        elif field["kind"] == "manual":
            res = {**MANUAL_RESULT, "flags": list(MANUAL_RESULT["flags"])}
        else:
            res = _rag_result(rag_outputs[name])
        fmt = field.get("format")
        if res["value"] and fmt and not check_format(fmt, res["value"]):
            res["status"] = "needs_review"
            res["flags"].append(f"format_{fmt}_tidak_valid")
        results[name] = res
    return results


# ---------- mode tempel pertanyaan dari form asli ----------

MAX_QUESTIONS = 20

PHONE_Q = r"\b(no\.?|nomor)\s*(hp|handphone|telepon|telp|wa|whatsapp)\b|\b(hp|handphone|telepon|whatsapp|phone)\b"
FIXED_KEYWORDS = [
    ("email", re.compile(r"\be-?mail\b")),
    ("telepon", re.compile(PHONE_Q)),
    ("linkedin", re.compile(r"\blinkedin\b")),
    ("github", re.compile(r"\bgithub\b")),
    ("nama", re.compile(r"^nama(\s+lengkap)?\W*$|\bnama lengkap\b|^full name\b")),
]
FORMAT_KEYWORDS = [("ipk", re.compile(r"\b(ipk|gpa)\b")), ("email", re.compile(r"\be-?mail\b")),
                   ("telepon", re.compile(PHONE_Q))]
LONG_HINTS = re.compile(r"ceritakan|jelaskan|kenapa|mengapa|bagaimana|deskripsikan|uraikan|motivasi|alasan"
                        r"|tell us|describe|why|explain")
# Kesediaan, ekspektasi, dan jadwal adalah keputusan pelamar, bukan fakta di dokumen. Pernah terjadi:
# "Apakah kamu bersedia magang onsite di Jakarta?" dijawab "Ya" tanpa bukti. Pertanyaan seperti ini tidak dikirim ke LLM.
MANUAL_HINTS = re.compile(r"bersedia|kesediaan|gaji|salary|ekspektasi|expected|kapan (anda|kamu|bisa|dapat) mulai"
                          r"|tanggal mulai|start date|ketersediaan|availability|willing")
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def parse_questions(text: str) -> list[str]:
    """Satu pertanyaan per baris. Nomor/bullet di depan dan tanda wajib (*) di belakang dibuang."""
    questions = []
    for line in text.splitlines():
        q = BULLET_RE.sub("", line).strip().rstrip("*").strip()
        if q and q not in questions:
            questions.append(q)
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"Maksimal {MAX_QUESTIONS} pertanyaan sekali jalan (ada {len(questions)}).")
    return questions


def classify_question(question: str) -> dict:
    """Tebak jenis isian dari teks pertanyaan: fixed (profil), manual (isi sendiri), factual (singkat),
    atau open (paragraf)."""
    q = question.lower().strip()
    field: dict = {"question": question}
    for key, pattern in FIXED_KEYWORDS:
        if pattern.search(q):
            return {**field, "kind": "fixed", "profile_key": key}
    if MANUAL_HINTS.search(q):
        return {**field, "kind": "manual"}
    field["kind"] = "open" if LONG_HINTS.search(q) else "factual"
    fmt = next((name for name, pattern in FORMAT_KEYWORDS if pattern.search(q)), None)
    if fmt:
        field["format"] = fmt
    return field


def draft_answers(questions: list[str], index: Index, profile: dict, *, position: str = "",
                  job_description: str = "") -> list[dict]:
    """Draf jawaban untuk pertanyaan bebas (mis. disalin dari form lamaran asli). Dokumen dicari tanpa filter tipe."""
    fields = [{"name": f"q{i}", "label": q, **classify_question(q)} for i, q in enumerate(questions, start=1)]
    results = autofill({"fields": fields}, index, profile, position=position, job_description=job_description)
    return [{"id": f["name"], "question": f["label"], "kind": f["kind"], **results[f["name"]]} for f in fields]
