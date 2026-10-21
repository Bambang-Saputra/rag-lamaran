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
ORIG_EMBED_TEXTS, ORIG_CHAT_JSON = embeddings.embed_texts, llm.chat_json
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


def test_extract_name_variants():
    from rag.profile import extract_name
    cases = {
        "# Raka Pratama\nEmail: raka@example.com": "Raka Pratama",
        "CURRICULUM VITAE\nBAMBANG SAPUTRA\nJakarta, Indonesia | bambang@example.com": "Bambang Saputra",
        "Resume\n\nSiti Aminah\n0812-3456-7890": "Siti Aminah",
        "DAFTAR RIWAYAT HIDUP\nData Pribadi\nNama Lengkap : Dewi Lestari\nTempat lahir: Bandung": "Dewi Lestari",
        "Nama: Budi Santoso\nAlamat: Jl. Merdeka 1": "Budi Santoso",
    }
    for text, expected in cases.items():
        assert extract_name(text)[0] == expected, (text, extract_name(text))
    # Tidak ada baris yang mirip nama: lebih baik kosong daripada menebak salah.
    assert extract_name("CURRICULUM VITAE\nbambang@example.com\n0812-3456-7890") is None


def test_save_location_depends_on_data_dir():
    import cli
    from rag.config import ROOT
    assert cli.output_dir(get_settings()) == ROOT / "outputs" / "examples"  # data dummy
    os.environ["DATA_DIR"] = "data/private"
    try:
        s = get_settings()
        assert cli.output_dir(s) == s.storage_dir / "examples"
    finally:
        os.environ["DATA_DIR"] = "data/dummy"


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


def test_keep_alive_sent_to_ollama():
    sent = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embeddings": [[1.0, 0.0]]}

    def fake_post(url, json=None, **kwargs):
        sent[url.rsplit("/", 1)[-1]] = json
        return FakeResponse()

    orig_requests_post, orig_llm_post = embeddings.requests.post, llm._post
    embeddings.requests.post = fake_post
    chat_payloads = []
    llm._post = lambda url, payload, headers, timeout: (
        chat_payloads.append(payload) or {"message": {"content": '{"answer": "", "status": "not_found", "sources": []}'}})
    os.environ["KEEP_ALIVE"] = "45m"
    try:
        ORIG_EMBED_TEXTS(["tes"])
        ORIG_CHAT_JSON("sys", "user", {})
        os.environ["KEEP_ALIVE"] = ""
        ORIG_CHAT_JSON("sys", "user", {})
    finally:
        embeddings.requests.post, llm._post = orig_requests_post, orig_llm_post
        os.environ.pop("KEEP_ALIVE", None)
    assert sent["embed"]["keep_alive"] == "45m"
    assert chat_payloads[0]["keep_alive"] == "45m"
    assert "keep_alive" not in chat_payloads[1]  # KEEP_ALIVE kosong = pakai default Ollama


def test_parse_and_classify_questions():
    from rag.autofill import MAX_QUESTIONS, classify_question, parse_questions
    qs = parse_questions("1. Nama lengkap *\n- Email\n\n• Berapa IPK terakhir Anda?\nEmail\n"
                         "Nama universitas\nCeritakan pengalaman organisasi Anda")
    assert qs == ["Nama lengkap", "Email", "Berapa IPK terakhir Anda?", "Nama universitas",
                  "Ceritakan pengalaman organisasi Anda"]
    kinds = {q: classify_question(q) for q in qs}
    assert kinds["Nama lengkap"]["kind"] == "fixed" and kinds["Nama lengkap"]["profile_key"] == "nama"
    assert kinds["Email"]["profile_key"] == "email"
    assert kinds["Berapa IPK terakhir Anda?"] == {"question": "Berapa IPK terakhir Anda?", "kind": "factual",
                                                  "format": "ipk"}
    assert kinds["Nama universitas"]["kind"] == "factual"  # bukan nama pelamar
    assert kinds["Ceritakan pengalaman organisasi Anda"]["kind"] == "open"
    assert classify_question("Nomor HP / WhatsApp aktif")["profile_key"] == "telepon"
    assert classify_question("Apakah kamu bersedia magang onsite di Jakarta?")["kind"] == "manual"
    assert classify_question("Berapa ekspektasi gaji Anda?")["kind"] == "manual"
    try:
        parse_questions("\n".join(f"pertanyaan {i}" for i in range(MAX_QUESTIONS + 1)))
        raise AssertionError("seharusnya ditolak")
    except ValueError:
        pass


def test_draft_answers_for_pasted_questions():
    from rag.autofill import draft_answers
    calls = []

    def chat(system, user, schema, model=None):
        calls.append(user)
        return fake_chat(system, user, schema, model)

    llm.chat_json = chat
    try:
        res = draft_answers(["Email", "Berapa IPK kumulatif saya?", "Ceritakan project yang relevan",
                             "Apakah kamu bersedia magang onsite?"], INDEX, PROFILE, position="Data Analyst Intern")
    finally:
        llm.chat_json = fake_chat
    assert [r["id"] for r in res] == ["q1", "q2", "q3", "q4"]
    assert [r["kind"] for r in res] == ["fixed", "factual", "open", "manual"]
    assert res[0]["value"] == "raka.pratama@example.com" and res[0]["status"] == "ok"
    assert res[2]["sources"] and res[2]["question"] == "Ceritakan project yang relevan"
    # Pertanyaan kesediaan tidak dijawab dan tidak dikirim ke LLM.
    assert res[3]["status"] == "not_found" and res[3]["flags"] == ["isi_sendiri"]
    assert not any("bersedia" in u for u in calls)


def test_warmup_failure_is_reported():
    import api
    orig = api.warmup
    api.warmup = lambda s: (_ for _ in ()).throw(ValueError("LLM_PROVIDER tidak dikenal: x"))
    try:
        api._warmup_in_background()
    finally:
        api.warmup = orig
    assert api._warm["state"] == "gagal" and "ValueError" in api._warm["error"]
    api._warm.update(state="belum", timings={}, error="")


def test_autofill_unknown_kind_falls_back_to_rag():
    form = {"fields": [{"name": "x", "label": "X", "kind": "salah_ketik", "question": "Berapa IPK kumulatif saya?"}]}
    res = autofill(form, INDEX, PROFILE)
    assert res["x"]["status"] in {"ok", "needs_review", "not_found"} and "retrieved" in res["x"]


def test_work_and_organization_experience_are_separated():
    from rag.autofill import classify_question
    org = classify_question("Pengalaman organisasi atau kepanitiaan")
    work = classify_question("Pengalaman kerja / magang")
    assert "organisasi" in org["doc_types"] and "bukan pengalaman kerja" in org["question"]
    assert work["doc_types"][0] == "cv" and "bukan organisasi" in work["question"]
    # Kata "magang" saja (bukan "pengalaman magang") tidak membatasi pencarian, supaya esai motivasi tetap terbaca.
    assert "doc_types" not in classify_question("Kenapa kamu tertarik dengan program magang ini?")
    # Kalau dua-duanya disebut, jangan dibatasi.
    assert "doc_types" not in classify_question("Ceritakan pengalaman kerja dan organisasi kamu")
    names = [f["name"] for f in load_form()["fields"]]
    assert "pengalaman_kerja" in names and "pengalaman_organisasi" in names


def test_long_questions_become_paragraph_answers():
    from rag.autofill import clean_question, draft_answers
    q = "Pengalaman organisasi atau kepanitiaan"
    assert draft_answers([q], INDEX, PROFILE)[0]["kind"] == "factual"  # tebakan dari teks saja
    assert draft_answers([q], INDEX, PROFILE, long_questions={q})[0]["kind"] == "open"  # kolomnya textarea
    assert clean_question("3. Nama lengkap *") == "Nama lengkap"


def test_api_demo_page_and_long_questions():
    from fastapi.testclient import TestClient
    import api

    api._cache.clear()
    client = TestClient(api.app)
    page = client.get("/demo")
    assert page.status_code == 200 and "PT Contoh Karya Digital" in page.text and "fiktif" in page.text
    assert "Pengalaman kerja / magang" in page.text and "Pengalaman organisasi atau kepanitiaan" in page.text
    res = client.post("/api/draft", json={"questions_text": "Pengalaman organisasi atau kepanitiaan *",
                                          "long_questions": ["Pengalaman organisasi atau kepanitiaan *"]})
    assert res.status_code == 200 and res.json()[0]["kind"] == "open"


def test_api_draft_and_status():
    from fastapi.testclient import TestClient
    import api

    api._cache.clear()
    client = TestClient(api.app)
    status = client.get("/api/status").json()
    assert status["state"] in {"belum", "memuat", "siap", "gagal"} and status["keep_alive"]
    res = client.post("/api/draft", json={"questions_text": "Email\nBerapa IPK saya?"})
    assert res.status_code == 200 and [r["kind"] for r in res.json()] == ["fixed", "factual"]
    assert client.post("/api/draft", json={"questions_text": "  \n "}).status_code == 400
    many = "\n".join(f"pertanyaan {i}" for i in range(25))
    assert client.post("/api/draft", json={"questions_text": many}).status_code == 400


def test_chat_loop():
    import cli
    inputs = iter(["", "/singkat", "/chunks", "Berapa IPK kumulatif saya?", "/keluar", "tidak dibaca"])
    calls = []

    def chat(system, user, schema, model=None):
        calls.append(system == RELEVANCE_SYSTEM)
        return fake_chat(system, user, schema, model)

    llm.chat_json = chat
    try:
        cli.run_chat(INDEX, read=lambda prompt: next(inputs))
    finally:
        llm.chat_json = fake_chat
    assert calls.count(False) == 1  # tepat satu pertanyaan yang dijawab, lalu keluar
    assert next(inputs) == "tidak dibaca"


def test_warmup_loads_models():
    from rag.warmup import warmup
    timings = warmup(get_settings())
    assert set(timings) == {SETTINGS.embed_model, SETTINGS.llm_model}


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
