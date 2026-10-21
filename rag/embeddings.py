"""Embedding lewat Ollama (/api/embed). Vektor dinormalisasi supaya dot product = cosine similarity."""
from __future__ import annotations

import numpy as np
import requests

from .config import get_settings


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    s = get_settings()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        try:
            payload = {"model": s.embed_model, "input": texts[i:i + batch_size]}
            if s.keep_alive:
                payload["keep_alive"] = s.keep_alive
            resp = requests.post(f"{s.embed_base_url}/api/embed", json=payload, timeout=s.request_timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Gagal memanggil embedding '{s.embed_model}' di {s.embed_base_url}. "
                f"Pastikan Ollama jalan dan model sudah di-pull (ollama pull {s.embed_model}). Detail: {e}"
            ) from e
        vectors.extend(resp.json()["embeddings"])

    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def embed_query(query: str) -> np.ndarray:
    """Query diberi instruksi tugas (format Qwen3-Embedding); dokumen di-embed tanpa instruksi.
    Kosongkan EMBED_QUERY_INSTRUCTION untuk model embedding yang tidak memakai instruksi."""
    instruction = get_settings().embed_query_instruction
    text = f"Instruct: {instruction}\nQuery:{query}" if instruction else query
    return embed_texts([text])[0]
