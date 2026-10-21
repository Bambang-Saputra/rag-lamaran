"""Memuat model embedding dan LLM ke memori sebelum dipakai, supaya pertanyaan pertama tidak menunggu
model dimuat (terukur 26-101 detik di GTX 1650). Model tetap di memori selama KEEP_ALIVE."""
from __future__ import annotations

import time

from . import embeddings, llm
from .config import Settings
from .prompts import ANSWER_SCHEMA, SYSTEM_PROMPT


def warmup(settings: Settings) -> dict:
    timings = {}
    if settings.retrieval_mode in ("hybrid", "vector"):
        t0 = time.perf_counter()
        embeddings.embed_query("tes")
        timings[settings.embed_model] = round(time.perf_counter() - t0, 1)
    models = [settings.llm_model]
    if settings.relevance_check and settings.relevance_model and settings.relevance_model != settings.llm_model:
        models.append(settings.relevance_model)
    for model in models:
        t0 = time.perf_counter()
        llm.chat_json(SYSTEM_PROMPT, "KONTEKS:\n(kosong)\n\nPERTANYAAN: tes", ANSWER_SCHEMA, model=model)
        timings[model] = round(time.perf_counter() - t0, 1)
    return timings
