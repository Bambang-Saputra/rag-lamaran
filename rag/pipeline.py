"""Fungsi inti `answer()`: retrieve -> generate -> validasi. Dipakai bersama oleh CLI dan API."""
from __future__ import annotations

import time

from . import llm
from .config import Settings, get_settings
from .index import Index
from .prompts import ANSWER_SCHEMA, SYSTEM_PROMPT, build_user_prompt, format_context
from .retrieve import retrieve
from .validate import SUPPORT_THRESHOLD, check_relevance, support_ratio

# Status akhir: ok | needs_review | not_found


def answer(question: str, index: Index, *, doc_types: list[str] | None = None, style: str = "long",
           k: int | None = None, task_context: str = "", retrieval_query: str | None = None,
           settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    t0 = time.perf_counter()
    retrieved = retrieve(index, retrieval_query or question, k or s.top_k,
                         doc_types=doc_types, mode=s.retrieval_mode)
    t1 = time.perf_counter()

    result = {"question": question, "doc_types": doc_types, "style": style, "retrieved": retrieved}
    if not retrieved:
        return {**result, "answer": "", "status": "not_found", "sources": [], "support": 0.0,
                "relevance": None, "flags": ["retrieval_kosong"],
                "timing_ms": {"retrieval": round((t1 - t0) * 1000)}}

    context, used_ids = format_context(retrieved, s.max_context_chars)
    raw = llm.chat_json(SYSTEM_PROMPT, build_user_prompt(question, context, style, task_context),
                        ANSWER_SCHEMA)
    t2 = time.perf_counter()

    text = str(raw.get("answer") or "").strip()
    cited = [str(x) for x in (raw.get("sources") or [])]
    valid = [cid for cid in dict.fromkeys(cited) if cid in used_ids]
    by_id = {c["chunk_id"]: c for c in retrieved}

    flags = []
    if "parse_error" in raw:
        flags.append("output_llm_bukan_json")
    if len(valid) < len(set(cited)):
        flags.append("sumber_tidak_valid")

    support = 0.0
    if raw.get("status") == "not_found" or not text:
        status, text, valid = "not_found", "", []
    else:
        support = round(support_ratio(text, [by_id[cid] for cid in valid]), 3)
        if not valid:
            flags.append("tanpa_sumber")
        if support < SUPPORT_THRESHOLD[style]:
            flags.append("dukungan_sumber_rendah")
        status = "needs_review" if (not valid or support < SUPPORT_THRESHOLD[style]
                                    or "sumber_tidak_valid" in flags) else "ok"

    sources = [{k: by_id[cid][k] for k in ("chunk_id", "file", "page", "section")} for cid in valid]
    out = {**result, "answer": text, "status": status, "sources": sources, "support": support,
           "relevance": None, "flags": flags,
           "timing_ms": {"retrieval": round((t1 - t0) * 1000), "generation": round((t2 - t1) * 1000)}}
    if s.relevance_check:
        run_relevance_check(out, s)
    return out


def run_relevance_check(out: dict, settings: Settings) -> dict:
    """Cek relevansi untuk hasil answer() yang sudah ada (mengubah `out`). Dipisah dari answer() supaya
    evaluasi bisa menjalankan semua generation dulu, baru cek relevansi di tahap kedua."""
    if out["status"] == "not_found":
        return out
    t0 = time.perf_counter()
    verdict = check_relevance(out["question"], out["answer"], settings)
    out["relevance"] = verdict
    out["timing_ms"]["relevance"] = round((time.perf_counter() - t0) * 1000)
    if verdict["relevant"] is False:
        out["flags"].append("jawaban_tidak_relevan")
        out["status"] = "needs_review"
    elif verdict["relevant"] is None:
        out["flags"].append("cek_relevansi_gagal")
    return out
