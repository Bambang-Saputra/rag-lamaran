"""API + halaman form simulasi. Jalankan: python cli.py serve  (lalu buka http://127.0.0.1:8000)"""
from __future__ import annotations

import difflib
import json
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag.autofill import autofill, load_form
from rag.config import ROOT, get_settings
from rag.index import load_index, load_profile
from rag.pipeline import answer

app = FastAPI(title="RAG Lamaran (simulasi)")
_cache: dict = {}


def _state():
    if "index" not in _cache:
        s = get_settings()
        try:
            _cache["index"] = load_index(s)
        except (FileNotFoundError, RuntimeError) as e:
            raise HTTPException(409, str(e)) from e
        _cache["profile"] = load_profile(s)
    return _cache["index"], _cache["profile"]


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:  # LLM / embedding tidak bisa diakses
        raise HTTPException(503, str(e)) from e


class AutofillRequest(BaseModel):
    position: str = ""
    job_description: str = ""


class AskRequest(BaseModel):
    question: str
    doc_types: list[str] | None = None
    style: str = "long"


class QCLogRequest(BaseModel):
    position: str = ""
    draft: dict[str, str]
    final: dict[str, str]


@app.get("/")
def page():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/form")
def form():
    return load_form()


@app.post("/api/autofill")
def api_autofill(req: AutofillRequest):
    index, profile = _state()
    return _run(autofill, load_form(), index, profile, position=req.position,
                job_description=req.job_description)


@app.post("/api/ask")
def api_ask(req: AskRequest):
    index, _ = _state()
    return _run(answer, req.question, index, doc_types=req.doc_types, style=req.style)


@app.get("/api/chunk")
def api_chunk(id: str):
    index, _ = _state()
    chunk = index.get(id)
    if chunk is None:
        raise HTTPException(404, f"Chunk tidak ditemukan: {id}")
    return chunk.to_dict()


@app.post("/api/qc-log")
def api_qc_log(req: QCLogRequest):
    """Simpan draf vs hasil QC. Disimpan di storage/ (di-ignore git) karena bisa berisi data asli."""
    fields = {}
    for name, draft in req.draft.items():
        final = req.final.get(name, "")
        fields[name] = {"draft": draft, "final": final, "edited": draft.strip() != final.strip(),
                        "similarity": round(difflib.SequenceMatcher(None, draft, final).ratio(), 3)}
    record = {"at": datetime.now().isoformat(timespec="seconds"), "position": req.position, "fields": fields}
    path = get_settings().storage_dir / "qc_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    edited = sum(f["edited"] for f in fields.values())
    return {"saved": True, "edited_fields": edited, "total_fields": len(fields)}
