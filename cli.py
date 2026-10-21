"""CLI: ingest, lihat chunk, retrieve (tanpa LLM), tanya-jawab, dan autofill.

Contoh:
    python cli.py ingest
    python cli.py chunks --type sertifikat
    python cli.py retrieve "sertifikat machine learning"
    python cli.py ask "Berapa IPK saya?" --type transkrip --short --save
    python cli.py autofill --position "Data Analyst Intern" --save
    python cli.py serve
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from rag.autofill import load_form, autofill
from rag.config import ROOT, get_settings
from rag.index import build_index, load_index, load_profile
from rag.pipeline import answer
from rag.retrieve import retrieve

# Konsol Windows default-nya cp1252; paksa UTF-8 supaya karakter non-ASCII tidak bikin crash.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = ROOT / "outputs" / "examples"


def _save(kind: str, label: str, payload: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:50] or kind
    path = OUTPUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{kind}_{slug}.json"
    payload = {"saved_at": datetime.now().isoformat(timespec="seconds"),
               "settings": get_settings().public(), **payload}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _print_retrieved(results: list[dict]) -> None:
    print(f"\n--- {len(results)} chunk ter-retrieve ---")
    for r in results:
        scores = "  ".join(f"{k}={v}" for k, v in r["scores"].items())
        print(f"#{r['rank']} [{r['chunk_id']}] {r['file']} hlm {r['page']} | {r['section']}  ({scores})")
        snippet = " ".join(r["text"].split())
        print(f"    {snippet[:220]}{'...' if len(snippet) > 220 else ''}")


def cmd_ingest(args) -> None:
    meta = build_index(get_settings())
    print(f"Index dibuat dari {meta['data_dir']}")
    print(f"{meta['n_docs']} dokumen, {meta['n_chunks']} chunk, embedding: {meta['embed_model']}")
    for doc_type, n in meta["by_type"].items():
        print(f"  {doc_type:<12} {n['dokumen']:>3} dokumen  {n['chunk']:>4} chunk")
    for w in meta["warnings"]:
        print(f"  PERINGATAN: {w}")


def cmd_chunks(args) -> None:
    index = load_index(get_settings())
    for c in index.chunks:
        if (args.type and c.doc_type != args.type) or (args.doc and c.doc_id != args.doc):
            continue
        print(f"\n[{c.chunk_id}] {c.file} hlm {c.page} | {c.section} ({len(c.text)} karakter)")
        print(c.text if args.full else " ".join(c.text.split())[:300])


def cmd_retrieve(args) -> None:
    s = get_settings()
    results = retrieve(load_index(s), args.query, args.k or s.top_k, doc_types=args.type, mode=s.retrieval_mode)
    _print_retrieved(results)
    if args.save:
        print(f"\nDisimpan: {_save('retrieve', args.query, {'query': args.query, 'retrieved': results})}")


def cmd_ask(args) -> None:
    s = get_settings()
    out = answer(args.question, load_index(s), doc_types=args.type, k=args.k,
                 style="short" if args.short else "long")
    _print_retrieved(out["retrieved"])
    print(f"\n--- Jawaban ({out['status']}, support={out['support']}) ---")
    print(out["answer"] or "(tidak ditemukan di dokumen)")
    print("\nSumber:")
    for src in out["sources"]:
        print(f"  - [{src['chunk_id']}] {src['file']} hlm {src['page']} | {src['section']}")
    if out["flags"]:
        print(f"Flags: {', '.join(out['flags'])}")
    if out.get("relevance"):
        print(f"Cek relevansi: {out['relevance']['relevant']} ({out['relevance']['reason']})")
    print(f"Waktu: {out['timing_ms']}")
    if args.save:
        print(f"\nDisimpan: {_save('ask', args.question, out)}")


def cmd_autofill(args) -> None:
    s = get_settings()
    jd = Path(args.jd).read_text(encoding="utf-8") if args.jd else ""
    form = load_form(Path(args.form)) if args.form else load_form()
    results = autofill(form, load_index(s), load_profile(s), position=args.position, job_description=jd)
    for field in form["fields"]:
        r = results[field["name"]]
        src = ", ".join(x.get("chunk_id") or f"{x['file']} ({x.get('method')})" for x in r["sources"])
        print(f"\n[{r['status']:<12}] {field['label']}")
        print(f"    {r['value'] or '-'}")
        if src:
            print(f"    sumber: {src}")
        if r["flags"]:
            print(f"    flags: {', '.join(r['flags'])}")
    if args.save:
        payload = {"position": args.position, "job_description": jd, "results": results}
        print(f"\nDisimpan: {_save('autofill', args.position or 'form', payload)}")


def cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run("api:app", host=args.host, port=args.port)


def main() -> None:
    p = argparse.ArgumentParser(description="RAG pengisi form lamaran dari dokumen pribadi")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="Baca dokumen, chunking, embedding, simpan index").set_defaults(fn=cmd_ingest)

    c = sub.add_parser("chunks", help="Lihat hasil chunking")
    c.add_argument("--type", help="Filter tipe dokumen, mis. sertifikat")
    c.add_argument("--doc", help="Filter doc_id, mis. cv/cv_raka_pratama")
    c.add_argument("--full", action="store_true", help="Tampilkan teks lengkap")
    c.set_defaults(fn=cmd_chunks)

    for name, fn, arg, helptext in [("retrieve", cmd_retrieve, "query", "Retrieval saja, tanpa LLM"),
                                    ("ask", cmd_ask, "question", "Tanya-jawab dengan sumber")]:
        q = sub.add_parser(name, help=helptext)
        q.add_argument(arg)
        q.add_argument("--type", action="append", help="Filter tipe dokumen (boleh diulang)")
        q.add_argument("--k", type=int, help="Jumlah chunk (default TOP_K)")
        q.add_argument("--save", action="store_true", help="Simpan output ke outputs/examples/")
        if name == "ask":
            q.add_argument("--short", action="store_true", help="Jawaban singkat (nilai saja)")
        q.set_defaults(fn=fn)

    a = sub.add_parser("autofill", help="Isi form simulasi dari dokumen")
    a.add_argument("--position", default="", help="Posisi yang dilamar")
    a.add_argument("--jd", help="File teks deskripsi lowongan")
    a.add_argument("--form", help="File config form (default forms/magang_form.json)")
    a.add_argument("--save", action="store_true")
    a.set_defaults(fn=cmd_autofill)

    sv = sub.add_parser("serve", help="Jalankan API + form simulasi")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)

    args = p.parse_args()
    try:
        args.fn(args)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
