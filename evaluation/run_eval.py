"""Evaluasi retrieval dan perbandingan LLM pada set pertanyaan berlabel (selalu memakai data/dummy).

    python -m evaluation.run_eval --check                 # cek label cocok dengan chunk (tanpa Ollama)
    python -m evaluation.run_eval --retrieval-only        # bm25 vs vector vs hybrid
    python -m evaluation.run_eval                         # retrieval + semua model di models.json
    python -m evaluation.run_eval --models qwen3:4b-instruct --limit 5

Hasil disimpan ke outputs/eval/<waktu>/ (results.jsonl, summary.json, summary.md).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from rag import llm
from rag.config import ROOT, get_settings
from rag.index import Index, build_index, load_index
from rag.pipeline import answer, run_relevance_check
from rag.prompts import ANSWER_SCHEMA, SYSTEM_PROMPT
from rag.retrieve import retrieve

HERE = Path(__file__).resolve().parent
ALL_MODES = ["bm25", "vector", "hybrid"]


# ---------- label & grading ----------

def normalize(text: str) -> str:
    text = re.sub(r"(\d),(\d)", r"\1.\2", text.lower())  # 3,62 -> 3.62
    return re.sub(r"\s+", " ", text).strip()


def term_in(term: str, text: str) -> bool:
    """True jika salah satu alternatif ('a|b') muncul sebagai kata utuh di text."""
    t = normalize(text)
    for alt in term.split("|"):
        a = normalize(alt)
        if a and re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", t):
            return True
    return False


def chunk_matches(chunk: dict, gold: dict) -> bool:
    """Gold tanpa 'contains' berarti chunk mana pun dari dokumen itu dianggap relevan."""
    return chunk["doc_id"] == gold["doc"] and normalize(gold.get("contains", "")) in normalize(chunk["text"])


def is_answerable(q: dict) -> bool:
    return q.get("expected_status") != "not_found"


def load_questions(path: Path = HERE / "questions.json") -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def check_labels(questions: list[dict], index: Index) -> list[str]:
    problems = []
    ids = [q["id"] for q in questions]
    problems += [f"id dobel: {i}" for i in {i for i in ids if ids.count(i) > 1}]
    for q in questions:
        if is_answerable(q):
            if not q.get("gold") or not q.get("must_include"):
                problems.append(f"{q['id']}: pertanyaan bisa dijawab tapi gold/must_include kosong")
            for g in q.get("gold", []):
                if not any(chunk_matches(c.to_dict(), g) for c in index.chunks):
                    problems.append(f"{q['id']}: gold tidak ditemukan di chunk mana pun: {g}")
        elif q.get("gold"):
            problems.append(f"{q['id']}: pertanyaan not_found tidak boleh punya gold")
    return problems


def grade(q: dict, out: dict, index: Index) -> dict:
    answered = out["status"] != "not_found"
    text = out["answer"]
    missing = [t for t in q.get("must_include", []) if not term_in(t, text)]
    forbidden = [t for t in q.get("must_not_include", []) if term_in(t, text)]
    if is_answerable(q):
        correct = answered and not missing and not forbidden
        cited = [index.get(s["chunk_id"]) for s in out["sources"]]
        citation_ok = any(c is not None and any(chunk_matches(c.to_dict(), g) for g in q["gold"]) for c in cited)
    else:
        correct, citation_ok = not answered, None
    return {"correct": correct, "answered": answered, "missing": missing, "forbidden": forbidden,
            "citation_ok": citation_ok, "json_ok": "output_llm_bukan_json" not in out["flags"]}


# ---------- retrieval ----------

def eval_retrieval(questions: list[dict], index: Index, modes: list[str], k: int) -> tuple[dict, list[dict]]:
    summary, details = {}, []
    answerable = [q for q in questions if is_answerable(q)]
    for mode in modes:
        hits, coverage, rr = [], [], []
        for q in answerable:
            res = retrieve(index, q["question"], k, doc_types=q.get("doc_types"), mode=mode)
            matched = [any(chunk_matches(r, g) for r in res) for g in q["gold"]]
            first = next((r["rank"] for r in res if any(chunk_matches(r, g) for g in q["gold"])), None)
            hits.append(any(matched))
            coverage.append(sum(matched) / len(matched))
            rr.append(1 / first if first else 0.0)
            details.append({"mode": mode, "id": q["id"], "hit": any(matched), "first_relevant_rank": first,
                            "gold_covered": f"{sum(matched)}/{len(matched)}",
                            "retrieved": [r["chunk_id"] for r in res]})
        summary[mode] = {"hit_at_k": round(statistics.mean(hits), 3),
                         "gold_coverage": round(statistics.mean(coverage), 3),
                         "mrr": round(statistics.mean(rr), 3), "n": len(answerable), "k": k}
    return summary, details


# ---------- generation ----------

def _ollama_ps(model: str) -> dict | None:
    """Berapa bagian model yang ada di VRAM (dari /api/ps). None jika tidak tersedia."""
    s = get_settings()
    try:
        data = requests.get(f"{s.llm_base_url}/api/ps", timeout=5).json()
    except (requests.RequestException, ValueError):
        return None
    for m in data.get("models", []):
        if m.get("name") == model or m.get("model") == model:
            size, vram = m.get("size") or 0, m.get("size_vram") or 0
            return {"size_gb": round(size / 1e9, 2), "vram_gb": round(vram / 1e9, 2),
                    "gpu_pct": round(100 * vram / size) if size else None}
    return None


def _set_model(model: dict) -> None:
    os.environ["LLM_MODEL"] = model["name"]
    os.environ["LLM_THINK"] = model.get("think", "")


def eval_model(model: dict, questions: list[dict], index: Index, results_file,
               relevance_stage: str = "separate") -> dict:
    """relevance_stage: "inline" = cek relevansi langsung setelah tiap jawaban (seperti di aplikasi);
    "separate" = semua generation dulu, baru cek relevansi di tahap kedua, supaya panggilan pemeriksa
    tidak bisa memengaruhi jawaban (mis. lewat cache prompt Ollama)."""
    _set_model(model)
    name = model["name"]
    relevance_on = get_settings().relevance_check
    separate = relevance_on and relevance_stage == "separate"
    stage_label = relevance_stage if relevance_on else "off"
    print(f"\n=== {name} (think={model.get('think') or '-'}, cek relevansi={stage_label}) ===")
    t0 = time.perf_counter()
    try:
        # Panggilan pertama memuat model ke memori; tidak dihitung di latensi.
        llm.chat_json(SYSTEM_PROMPT, "KONTEKS:\n(kosong)\n\nPERTANYAAN: tes", ANSWER_SCHEMA)
    except RuntimeError as e:
        print(f"  dilewati: {e}\n  (sudah `ollama pull {name}`?)")
        return {"model": name, "think": model.get("think", ""), "error": str(e)}
    load_s = round(time.perf_counter() - t0, 1)
    memory = _ollama_ps(name)

    # Tahap 1: generation. Pada mode separate, cek relevansi dimatikan dulu.
    outputs: list[tuple[dict, dict | None, str | None]] = []
    if separate:
        os.environ["RELEVANCE_CHECK"] = "false"
    try:
        for q in questions:
            try:
                outputs.append((q, answer(q["question"], index, doc_types=q.get("doc_types"), style=q["style"]), None))
            except RuntimeError as e:
                outputs.append((q, None, str(e)))
    finally:
        if separate:
            os.environ["RELEVANCE_CHECK"] = "true"

    # Tahap 2: cek relevansi untuk semua jawaban sekaligus.
    if separate:
        for q, out, err in outputs:
            if out is not None:
                try:
                    run_relevance_check(out, get_settings())
                except RuntimeError as e:
                    out["flags"].append("cek_relevansi_gagal")
                    print(f"  {q['id']}: cek relevansi error {e}")

    rows = []
    for q, out, err in outputs:
        if out is None:
            rows.append({"model": name, "id": q["id"], "error": err})
            print(f"  {q['id']}: error {err}")
            continue
        g = grade(q, out, index)
        row = {"model": name, "id": q["id"], "category": q["category"], "question": q["question"],
               "expected_status": q.get("expected_status", "answered"), "status": out["status"],
               "answer": out["answer"], "sources": [s["chunk_id"] for s in out["sources"]],
               "flags": out["flags"], "relevance": out.get("relevance"),
               "generation_ms": out["timing_ms"].get("generation"),
               "total_ms": out["timing_ms"].get("generation", 0) + out["timing_ms"].get("relevance", 0), **g}
        rows.append(row)
        results_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {q['id']:<4} {'BENAR' if g['correct'] else 'SALAH':<6} [{out['status']}] "
              f"{' '.join(out['answer'].split())[:90]}")

    return summarize_rows(rows, {"model": name, "think": model.get("think", ""), "load_s": load_s,
                                 "memory": memory, "relevance_check": relevance_on,
                                 "relevance_stage": stage_label})


def summarize_rows(rows: list[dict], base: dict) -> dict:
    graded = [r for r in rows if "error" not in r]
    ans = [r for r in graded if r["expected_status"] != "not_found"]
    unans = [r for r in graded if r["expected_status"] == "not_found"]
    halluc = [r for r in unans if r["answered"]]
    answered_ans = [r for r in ans if r["answered"]]
    answered_wrong = [r for r in graded if r["answered"] and not r["correct"]]
    answered_right = [r for r in graded if r["answered"] and r["correct"]]
    lat = [r["generation_ms"] for r in graded if r["generation_ms"] is not None]
    total = [r.get("total_ms") or r["generation_ms"] for r in graded if r["generation_ms"] is not None]

    def rate(xs: list, pred) -> float | None:
        return round(sum(1 for x in xs if pred(x)) / len(xs), 3) if xs else None

    def irrelevant(r: dict) -> bool:
        return "jawaban_tidak_relevan" in r["flags"]

    return {
        **base, "n": len(graded), "errors": len(rows) - len(graded),
        "answer_accuracy": rate(ans, lambda r: r["correct"]),
        "abstain_accuracy": rate(unans, lambda r: r["correct"]),
        "false_abstain": rate(ans, lambda r: not r["answered"]),
        "hallucination_caught": rate(halluc, lambda r: r["status"] == "needs_review"),
        # Cek relevansi dinilai terhadap grader: menandai jawaban salah = tangkapan,
        # menandai jawaban benar = salah tandai (biaya QC tambahan).
        "relevance_catch": rate(answered_wrong, irrelevant),
        "relevance_false_flag": rate(answered_right, irrelevant),
        "json_valid": rate(graded, lambda r: r["json_ok"]),
        "citation_accuracy": rate(answered_ans, lambda r: r["citation_ok"]),
        "median_s": round(statistics.median(lat) / 1000, 1) if lat else None,
        "mean_s": round(statistics.mean(lat) / 1000, 1) if lat else None,
        "median_total_s": round(statistics.median(total) / 1000, 1) if total else None,
        "per_question": {r["id"]: (None if "error" in r else r["correct"]) for r in rows},
    }


# ---------- laporan ----------

def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def render_markdown(meta: dict, retrieval: dict, models: list[dict], questions: list[dict]) -> str:
    s = meta["settings"]
    lines = [f"# Hasil evaluasi ({meta['run_at']})", "",
             f"Data: `data/dummy` ({meta['index']['n_docs']} dokumen, {meta['index']['n_chunks']} chunk). "
             f"Embedding: `{meta['index']['embed_model'] or 'tidak dipakai (BM25 saja)'}`. "
             f"Retrieval untuk generation: `{s['retrieval_mode']}`, "
             f"TOP_K={s['top_k']}. {len(questions)} pertanyaan "
             f"({sum(is_answerable(q) for q in questions)} bisa dijawab, "
             f"{sum(not is_answerable(q) for q in questions)} tidak).", ""]
    if retrieval:
        lines += ["## Retrieval", "", "| Mode | Hit@k | Cakupan gold | MRR |", "|---|---|---|---|"]
        for mode, r in retrieval.items():
            lines.append(f"| {mode} | {_pct(r['hit_at_k'])} | {_pct(r['gold_coverage'])} | {r['mrr']} |")
        lines.append("")
    ok = [m for m in models if "error" not in m]
    if ok:
        lines += [f"## Generation (cek relevansi: {ok[0].get('relevance_stage') or 'off'})", "",
                  "| Model | Akurasi (bisa dijawab) | Abstain benar | Salah abstain | Halusinasi tertangkap validasi "
                  "| Relevansi: jawaban salah tertangkap | Relevansi: jawaban benar ikut ditandai | JSON valid "
                  "| Sitasi benar | Median detik (generate) | Median detik (total) | Di GPU |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in ok:
            gpu = f"{m['memory']['gpu_pct']}%" if m.get("memory") and m["memory"].get("gpu_pct") is not None else "-"
            lines.append(f"| `{m['model']}` | {_pct(m['answer_accuracy'])} | {_pct(m['abstain_accuracy'])} | "
                         f"{_pct(m['false_abstain'])} | {_pct(m['hallucination_caught'])} | "
                         f"{_pct(m.get('relevance_catch'))} | {_pct(m.get('relevance_false_flag'))} | "
                         f"{_pct(m['json_valid'])} | {_pct(m['citation_accuracy'])} | {m['median_s']} | "
                         f"{m.get('median_total_s')} | {gpu} |")
        lines += ["", "### Per pertanyaan", "",
                  "| ID | Kategori | " + " | ".join(f"`{m['model']}`" for m in ok) + " |",
                  "|---|---|" + "---|" * len(ok)]
        mark = {True: "benar", False: "salah", None: "error"}
        for q in questions:
            cells = [mark[m["per_question"].get(q["id"])] for m in ok]
            lines.append(f"| {q['id']} | {q['category']} | " + " | ".join(cells) + " |")
        lines.append("")
    for m in models:
        if "error" in m:
            lines.append(f"- `{m['model']}` dilewati: {m['error']}")
    return "\n".join(lines) + "\n"


def rescore(out_dir: Path) -> str:
    """Hitung ulang summary.json/summary.md dari results.jsonl, mis. setelah definisi metrik berubah.
    Jawaban model tidak dijalankan ulang."""
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    keep = ("model", "think", "load_s", "memory", "relevance_check", "relevance_stage")
    models = []
    for m in summary["models"]:
        if "error" in m:
            models.append(m)
            continue
        base = {k: m.get(k) for k in keep}
        # Run sebelum opsi --relevance-stage ada selalu memakai cek inline.
        base["relevance_stage"] = base["relevance_stage"] or ("inline" if base["relevance_check"] else "off")
        models.append(summarize_rows([r for r in rows if r["model"] == m["model"]], base))
    ids = {r["id"] for r in rows}
    questions = [q for q in load_questions() if q["id"] in ids]
    summary["models"] = models
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {k: summary[k] for k in ("run_at", "settings", "index")}
    report = render_markdown(meta, summary["retrieval"], models, questions)
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="Cek label saja (tanpa Ollama)")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    p.add_argument("--gen-mode", default="hybrid", choices=ALL_MODES, help="Retrieval mode untuk generation")
    p.add_argument("--models", nargs="+", help="Filter nama model dari models.json")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, help="Hanya N pertanyaan pertama (uji cepat)")
    p.add_argument("--ids", nargs="+", help="Hanya pertanyaan dengan id ini, mis. u01 u03 f01")
    p.add_argument("--relevance", choices=["on", "off"], default="on", help="Cek relevansi jawaban (LLM)")
    p.add_argument("--relevance-stage", choices=["separate", "inline"], default="separate",
                   help="separate = generation semua soal dulu, baru cek relevansi; inline = seperti di aplikasi")
    p.add_argument("--rescore", metavar="DIR", help="Hitung ulang ringkasan run yang sudah ada (tanpa Ollama)")
    args = p.parse_args()
    if args.rescore:
        print(rescore(Path(args.rescore)))
        return
    os.environ["RELEVANCE_CHECK"] = "true" if args.relevance == "on" else "false"

    # Label mengacu ke data dummy; index evaluasi dipisah dari index utama.
    os.environ["DATA_DIR"] = str(ROOT / "data" / "dummy")
    os.environ["STORAGE_DIR"] = str(ROOT / "storage" / "eval")
    os.environ["TOP_K"] = str(args.k)
    # Index dibangun dengan embedding kalau ada mode yang butuh vektor; --check cukup BM25 (tanpa Ollama).
    needs_vectors = not args.check and (args.gen_mode != "bm25" or any(m != "bm25" for m in args.modes))
    os.environ["RETRIEVAL_MODE"] = "hybrid" if needs_vectors else "bm25"

    questions = load_questions()
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]
    if args.limit:
        questions = questions[: args.limit]
    try:
        index_meta = build_index(get_settings())
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    index = load_index(get_settings())

    problems = check_labels(questions, index)
    print(f"{len(questions)} pertanyaan, {index_meta['n_chunks']} chunk. Masalah label: {len(problems)}")
    for pr in problems:
        print(f"  - {pr}")
    if args.check or problems:
        sys.exit(1 if problems else 0)

    os.environ["RETRIEVAL_MODE"] = args.gen_mode
    retrieval, retrieval_details = eval_retrieval(questions, index, args.modes, args.k)
    print("\nRetrieval:")
    for mode, r in retrieval.items():
        print(f"  {mode:<7} hit@{args.k}={_pct(r['hit_at_k'])}  cakupan={_pct(r['gold_coverage'])}  mrr={r['mrr']}")

    out_dir = ROOT / "outputs" / "eval" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_summaries = []
    if not args.retrieval_only:
        models = json.loads((HERE / "models.json").read_text(encoding="utf-8"))
        if args.models:
            models = [m for m in models if m["name"] in args.models]
        with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
            for m in models:
                model_summaries.append(eval_model(m, questions, index, fh, relevance_stage=args.relevance_stage))

    meta = {"run_at": datetime.now().isoformat(timespec="seconds"), "settings": get_settings().public(),
            "index": index_meta}
    (out_dir / "summary.json").write_text(json.dumps(
        {**meta, "retrieval": retrieval, "retrieval_details": retrieval_details, "models": model_summaries},
        ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_markdown(meta, retrieval, model_summaries, questions)
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    print(f"\n{report}\nDisimpan di {out_dir}")


if __name__ == "__main__":
    main()
