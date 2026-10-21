"""Bangun dan muat index: chunks.jsonl + vectors.npy + profile.json di STORAGE_DIR.

Vector store sengaja berupa matriks numpy (exact search). Untuk corpus puluhan dokumen /
ratusan chunk ini cukup cepat, tanpa approximation error, dan mudah diinspeksi.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import embeddings
from .bm25 import BM25
from .chunking import Chunk, chunk_documents
from .config import Settings, display_path
from .loaders import load_documents
from .profile import extract_profile


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray | None
    bm25: BM25
    meta: dict

    def get(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.chunk_id == chunk_id), None)


def _files(storage: Path) -> dict[str, Path]:
    return {name: storage / f for name, f in
            [("chunks", "chunks.jsonl"), ("vectors", "vectors.npy"),
             ("meta", "meta.json"), ("profile", "profile.json")]}


def build_index(settings: Settings) -> dict:
    docs, warnings = load_documents(settings.data_dir)
    chunks = chunk_documents(docs, settings.max_chunk_chars)
    if not chunks:
        raise RuntimeError(f"Tidak ada chunk yang dihasilkan dari {settings.data_dir}")

    vectors = None
    if settings.retrieval_mode in ("hybrid", "vector"):
        vectors = embeddings.embed_texts([c.search_text() for c in chunks])

    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    f = _files(settings.storage_dir)
    with f["chunks"].open("w", encoding="utf-8") as out:
        for c in chunks:
            out.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    if vectors is not None:
        np.save(f["vectors"], vectors)
    elif f["vectors"].exists():
        f["vectors"].unlink()

    profile = extract_profile(docs)
    f["profile"].write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    by_type: dict[str, dict[str, int]] = {}
    for d in docs:
        by_type.setdefault(d.doc_type, {"dokumen": 0, "chunk": 0})["dokumen"] += 1
    for c in chunks:
        by_type[c.doc_type]["chunk"] += 1

    meta = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": display_path(settings.data_dir),
        "embed_model": settings.embed_model if vectors is not None else None,
        "max_chunk_chars": settings.max_chunk_chars,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "by_type": by_type,
        "warnings": warnings,
    }
    f["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_index(settings: Settings) -> Index:
    f = _files(settings.storage_dir)
    if not f["chunks"].exists():
        raise FileNotFoundError("Index belum ada. Jalankan dulu: python cli.py ingest")
    with f["chunks"].open(encoding="utf-8") as fh:
        chunks = [Chunk(**json.loads(line)) for line in fh if line.strip()]
    vectors = np.load(f["vectors"]) if f["vectors"].exists() else None
    meta = json.loads(f["meta"].read_text(encoding="utf-8"))

    if settings.retrieval_mode in ("hybrid", "vector"):
        if vectors is None:
            raise RuntimeError("Index dibuat tanpa embedding. Jalankan ulang `python cli.py ingest` "
                               f"dengan RETRIEVAL_MODE={settings.retrieval_mode}.")
        if meta.get("embed_model") != settings.embed_model:
            raise RuntimeError(f"Index dibuat dengan embedding '{meta.get('embed_model')}', setting sekarang "
                               f"'{settings.embed_model}'. Jalankan ulang `python cli.py ingest`.")
    return Index(chunks, vectors, BM25([c.search_text() for c in chunks]), meta)


def load_profile(settings: Settings) -> dict:
    path = _files(settings.storage_dir)["profile"]
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
