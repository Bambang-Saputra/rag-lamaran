"""Isi form per field. Tipe field (dari config form):

- fixed   : ambil dari profile.json (regex), tanpa LLM
- factual : RAG, jawaban singkat
- open    : RAG, jawaban paragraf; posisi yang dilamar ikut dipakai

Cek relevansi dijalankan setelah semua field selesai di-generate. Kalau diselipkan di antara field,
jawaban field berikutnya bisa berubah (terukur di evaluasi, lihat README bagian Cek relevansi).
"""
from __future__ import annotations

import json
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
                   for field in form["fields"] if field["kind"] != "fixed"}
    # Tahap 2: cek relevansi untuk semua jawaban sekaligus.
    if s.relevance_check:
        for out in rag_outputs.values():
            run_relevance_check(out, s)

    results = {}
    for field in form["fields"]:
        name = field["name"]
        res = _fixed(field, profile) if field["kind"] == "fixed" else _rag_result(rag_outputs[name])
        fmt = field.get("format")
        if res["value"] and fmt and not check_format(fmt, res["value"]):
            res["status"] = "needs_review"
            res["flags"].append(f"format_{fmt}_tidak_valid")
        results[name] = res
    return results
