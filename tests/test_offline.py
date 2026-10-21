"""Tes offline: embedding dan LLM diganti stub, jadi bisa jalan tanpa Ollama.

Yang dicek adalah alur (ingest -> retrieve -> answer -> validasi -> autofill -> API), bukan kualitas jawaban.
Jalankan: python -m tests.test_offline   (atau pytest, kalau ter-install)
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import zlib

import numpy as np

os.environ["DATA_DIR"] = "data/dummy"
os.environ["STORAGE_DIR"] = tempfile.mkdtemp(prefix="rag-lamaran-test-")
os.environ["RETRIEVAL_MODE"] = "hybrid"

from rag import embeddings, llm  # noqa: E402
from rag.autofill import autofill, load_form  # noqa: E402
from rag.bm25 import tokenize  # noqa: E402
from rag.config import get_settings  # noqa: E402
from rag.index import build_index, load_index, load_profile  # noqa: E402
from rag.pipeline import answer  # noqa: E402
from rag.prompts import RELEVANCE_SYSTEM  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402


def fake_embed(texts, batch_size=32):
    """Bag-of-words di-hash ke 256 dimensi. Cukup untuk menguji alur, bukan kualitas semantik."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in tokenize(t):
            out[i, zlib.crc32(tok.encode()) % 256] += 1.0
    return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)


BLOCK_RE = re.compile(r"\[chunk_id: (\S+) \|[^\]]*\]\n(.*?)(?=\n\n---|\n\nINFO POSISI|\n\nPERTANYAAN)", re.S)


def first_chunk_answer(user):
    """Jawab dengan baris pertama chunk peringkat 1, dan sebut chunk itu sebagai sumber."""
    m = BLOCK_RE.search(user)
    if m is None:  # mis. panggilan pemanasan tanpa konteks
        return {"answer": "", "status": "not_found", "sources": []}
    first_line = next(line for line in m.group(2).splitlines() if line.strip()).strip()
    return {"answer": first_line, "status": "answered", "sources": [m.group(1)]}


def stub_chat(answer_fn, relevance=None):
    """Stub chat_json: panggilan pemeriksa relevansi dijawab `relevance`, sisanya oleh answer_fn(user)."""
    relevance = relevance or {"relevant": True, "reason": "stub"}

    def chat(system, user, schema, model=None):
        return dict(relevance) if system == RELEVANCE_SYSTEM else answer_fn(user)
    return chat


fake_chat = stub_chat(first_chunk_answer)
embeddings.embed_texts = fake_embed
llm.chat_json = fake_chat

SETTINGS = get_settings()
META = build_index(SETTINGS)
INDEX = load_index(SETTINGS)
PROFILE = load_profile(SETTINGS)


def test_ingest_counts_and_metadata():
    assert META["n_docs"] == 10, META
    assert set(META["by_type"]) == {"cv", "transkrip", "sertifikat", "project", "organisasi", "esai"}
    assert META["by_type"]["sertifikat"] == {"dokumen": 4, "chunk": 4}  # 1 sertifikat = 1 chunk
    for c in INDEX.chunks:
        assert c.page >= 1 and c.section and c.text.strip()
    cv_sections = {c.section for c in INDEX.chunks if c.doc_type == "cv"}
    assert {"Pendidikan", "Pengalaman", "Skills"} <= cv_sections, cv_sections
    # Baris "IPK 3,62 / 4,00" harus tetap jadi isi section Pendidikan, bukan heading baru.
    edu = next(c for c in INDEX.chunks if c.doc_type == "cv" and c.section == "Pendidikan")
    assert "IPK 3,62" in edu.text, edu.text


def test_heading_detection():
    from rag.chunking import _heading
    assert _heading("## Pendidikan") == "Pendidikan"
    assert _heading("WORK EXPERIENCE") == "Work Experience"
    assert _heading("Semester 3") == "Semester 3"
    assert _heading("Pengalaman:") == "Pengalaman"
    assert _heading("IPK 3,62 / 4,00") is None
    assert _heading("Python, SQL, Pandas") is None


def test_profile_regex():
    assert PROFILE["nama"]["value"] == "Raka Pratama"
    assert PROFILE["email"]["value"] == "raka.pratama@example.com"
    assert PROFILE["telepon"]["value"].startswith("0812")
    assert "linkedin.com/in/raka-pratama-dummy" in PROFILE["linkedin"]["value"]


def test_retrieve_filter_and_ranking():
    res = retrieve(INDEX, "sertifikat machine learning scikit-learn", 3, doc_types=["sertifikat"])
    assert res and all(r["doc_type"] == "sertifikat" for r in res)
    assert res[0]["doc_id"] == "sertifikat/sert_dasar_machine_learning", [r["doc_id"] for r in res]
    assert {"rrf", "bm25", "vector"} <= set(res[0]["scores"])


def test_answer_sources_come_from_retrieved():
    out = answer("Berapa IPK kumulatif saya?", INDEX, doc_types=["transkrip"], style="short")
    retrieved_ids = {r["chunk_id"] for r in out["retrieved"]}
    assert out["status"] == "ok", out
    assert out["sources"] and all(s["chunk_id"] in retrieved_ids for s in out["sources"])


def test_hallucinated_source_is_flagged():
    llm.chat_json = stub_chat(lambda u: {"answer": "IPK 3,62", "status": "answered", "sources": ["ngarang#c9"]})
    try:
        out = answer("Berapa IPK saya?", INDEX, doc_types=["transkrip"], style="short")
    finally:
        llm.chat_json = fake_chat
    assert out["status"] == "needs_review"
    assert "sumber_tidak_valid" in out["flags"] and "tanpa_sumber" in out["flags"]


def test_unsupported_answer_needs_review():
    llm.chat_json = stub_chat(lambda u: {"answer": "Lulusan Harvard dengan IPK 4,0", "status": "answered",
                                         "sources": [BLOCK_RE.search(u).group(1)]})
    try:
        out = answer("Di universitas mana saya kuliah?", INDEX, doc_types=["cv"], style="short")
    finally:
        llm.chat_json = fake_chat
    assert out["status"] == "needs_review" and "dukungan_sumber_rendah" in out["flags"], out


def test_not_found():
    llm.chat_json = stub_chat(lambda u: {"answer": "", "status": "not_found", "sources": []})
    try:
        out = answer("Project computer vision apa yang pernah saya buat?", INDEX, doc_types=["project"])
    finally:
        llm.chat_json = fake_chat
    assert out["status"] == "not_found" and out["sources"] == [] and out["relevance"] is None

    out = answer("apa saja", INDEX, doc_types=["tipe_yang_tidak_ada"])
    assert out["status"] == "not_found" and "retrieval_kosong" in out["flags"]


def test_autofill_field_without_documents():
    # Mis. data asli tanpa folder esai/: retrieval kosong tidak boleh membuat autofill crash.
    form = {"fields": [{"name": "motivasi", "label": "Motivasi", "kind": "open",
                        "question": "Kenapa saya tertarik?", "doc_types": ["tipe_yang_tidak_ada"]}]}
    res = autofill(form, INDEX, PROFILE, position="Data Analyst Intern")
    assert res["motivasi"]["status"] == "not_found" and "retrieval_kosong" in res["motivasi"]["flags"]


def test_parse_json_non_object():
    from rag.llm import _parse_json
    assert _parse_json('{"answer": "x", "status": "answered", "sources": []}')["answer"] == "x"
    assert _parse_json('Berikut jawabannya: {"answer": "y", "status": "answered", "sources": []}')["answer"] == "y"
    assert "parse_error" in _parse_json('["bukan", "object"]')
    assert "parse_error" in _parse_json("bukan json")


def test_relevance_check():
    # Jawaban didukung sumber (baris pertama chunk), tapi pemeriksa bilang tidak menjawab pertanyaannya.
    llm.chat_json = stub_chat(first_chunk_answer, {"relevant": False, "reason": "ditanya IELTS, dijawab TOEFL"})
    try:
        out = answer("Berapa skor IELTS saya?", INDEX, doc_types=["cv"], style="short")
    finally:
        llm.chat_json = fake_chat
    assert out["status"] == "needs_review" and "jawaban_tidak_relevan" in out["flags"], out
    assert out["relevance"]["relevant"] is False and "relevance" in out["timing_ms"]

    # Output pemeriksa tidak terbaca: ditandai, tapi status tidak diubah.
    llm.chat_json = stub_chat(first_chunk_answer, {"oops": 1})
    try:
        out = answer("Berapa IPK kumulatif saya?", INDEX, doc_types=["transkrip"], style="short")
    finally:
        llm.chat_json = fake_chat
    assert out["status"] == "ok" and "cek_relevansi_gagal" in out["flags"], out

    # Bisa dimatikan lewat RELEVANCE_CHECK=false.
    os.environ["RELEVANCE_CHECK"] = "false"
    try:
        out = answer("Berapa IPK kumulatif saya?", INDEX, doc_types=["transkrip"], style="short")
    finally:
        os.environ.pop("RELEVANCE_CHECK")
    assert out["relevance"] is None and "relevance" not in out["timing_ms"]


def test_autofill_routes_fields():
    form = load_form()
    calls = []

    def chat(system, user, schema, model=None):
        calls.append("judge" if system == RELEVANCE_SYSTEM else "gen")
        return fake_chat(system, user, schema, model)

    llm.chat_json = chat
    try:
        res = autofill(form, INDEX, PROFILE, position="Data Analyst Intern")
    finally:
        llm.chat_json = fake_chat
    # Semua generation selesai sebelum cek relevansi pertama (dua tahap).
    assert "judge" in calls and calls.index("judge") > max(i for i, c in enumerate(calls) if c == "gen")
    assert list(res) == [f["name"] for f in form["fields"]]
    assert res["email"]["status"] == "ok" and res["email"]["sources"][0]["method"] == "regex"
    assert "retrieved" not in res["nama"] and "retrieved" in res["sertifikasi_relevan"]
    for r in res.values():
        assert r["status"] in {"ok", "needs_review", "not_found"}


def test_api_endpoints():
    from fastapi.testclient import TestClient
    import api

    api._cache.clear()
    client = TestClient(api.app)
    assert client.get("/").status_code == 200
    assert client.get("/api/form").json()["fields"]
    filled = client.post("/api/autofill", json={"position": "Data Analyst Intern"}).json()
    cid = filled["sertifikasi_relevan"]["sources"][0]["chunk_id"]
    assert client.get("/api/chunk", params={"id": cid}).json()["chunk_id"] == cid
    assert client.get("/api/chunk", params={"id": "tidak#ada"}).status_code == 404
    draft = {k: v["value"] for k, v in filled.items()}
    final = {**draft, "motivasi": "Saya edit sendiri."}
    log = client.post("/api/qc-log", json={"position": "x", "draft": draft, "final": final}).json()
    assert log["edited_fields"] == 1


def test_eval_labels_match_dummy_chunks():
    from evaluation.run_eval import check_labels, load_questions
    assert check_labels(load_questions(), INDEX) == []


def test_eval_grading():
    from evaluation.run_eval import grade, term_in
    assert term_in("A-", "Nilai Data Mining: A-") and not term_in("A", "Nilai: A-")
    assert term_in("3.62", "IPK saya 3,62") and term_in("Feb|Februari", "Februari 2024")
    assert not term_in("feb", "februari")

    ipk_chunk = next(c for c in INDEX.chunks if "IPK Kumulatif" in c.text)
    q = {"id": "x", "question": "IPK?", "must_include": ["3.62"],
         "gold": [{"doc": ipk_chunk.doc_id, "contains": "IPK Kumulatif: 3,62"}]}
    out = {"status": "ok", "answer": "3,62", "flags": [], "sources": [{"chunk_id": ipk_chunk.chunk_id}]}
    assert grade(q, out, INDEX) == {"correct": True, "answered": True, "missing": [], "forbidden": [],
                                    "citation_ok": True, "json_ok": True}
    unans = {"id": "u", "question": "IELTS?", "expected_status": "not_found", "must_not_include": ["540"]}
    g = grade(unans, {"status": "needs_review", "answer": "540", "flags": [], "sources": []}, INDEX)
    assert g["correct"] is False and g["forbidden"] == ["540"] and g["citation_ok"] is None


def test_eval_model_smoke():
    import io
    from evaluation.run_eval import eval_model, eval_retrieval, load_questions, render_markdown
    questions = [q for q in load_questions() if q["id"] in {"f01", "c01", "u03"}]
    retrieval, _ = eval_retrieval(questions, INDEX, ["bm25", "hybrid"], 5)
    assert set(retrieval) == {"bm25", "hybrid"} and retrieval["bm25"]["n"] == 2

    old = {k: os.environ.get(k) for k in ("LLM_MODEL", "LLM_THINK")}
    try:
        summary = eval_model({"name": "stub", "think": ""}, questions, INDEX, io.StringIO())
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert summary["n"] == 3 and summary["errors"] == 0
    assert set(summary["per_question"]) == {"f01", "c01", "u03"}
    meta = {"run_at": "x", "settings": SETTINGS.public(), "index": META}
    assert "| `stub` |" in render_markdown(meta, retrieval, [summary], questions)


def test_eval_relevance_separate_stage():
    """Mode separate: semua panggilan generation selesai sebelum panggilan pemeriksa pertama."""
    import io
    from evaluation.run_eval import eval_model, load_questions
    questions = [q for q in load_questions() if q["id"] in {"f01", "u03", "o05"}]
    calls = []

    def chat(system, user, schema, model=None):
        is_judge = system == RELEVANCE_SYSTEM
        calls.append("judge" if is_judge else "gen")
        if is_judge:
            return {"relevant": "IELTS" not in user, "reason": "stub"}
        return first_chunk_answer(user)

    old = {k: os.environ.get(k) for k in ("LLM_MODEL", "LLM_THINK", "RELEVANCE_CHECK")}
    llm.chat_json = chat
    os.environ["RELEVANCE_CHECK"] = "true"
    try:
        out = io.StringIO()
        summary = eval_model({"name": "stub", "think": ""}, questions, INDEX, out, relevance_stage="separate")
        assert os.environ["RELEVANCE_CHECK"] == "true"
    finally:
        llm.chat_json = fake_chat
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    gen_calls = [i for i, c in enumerate(calls) if c == "gen"]
    judge_calls = [i for i, c in enumerate(calls) if c == "judge"]
    assert judge_calls and max(gen_calls) < min(judge_calls), calls
    rows = {r["id"]: r for r in map(json.loads, out.getvalue().splitlines())}
    assert "jawaban_tidak_relevan" in rows["u03"]["flags"] and rows["u03"]["status"] == "needs_review"
    assert rows["f01"]["relevance"]["relevant"] is True
    assert summary["relevance_stage"] == "separate"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} lulus")
    sys.exit(1 if failed else 0)
