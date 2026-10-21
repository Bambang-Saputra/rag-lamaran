"""Pemanggilan LLM yang mengembalikan JSON. Provider: Ollama lokal atau API kompatibel OpenAI."""
from __future__ import annotations

import json
import re

import requests

from .config import get_settings


def _post(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Gagal memanggil LLM di {url}. Pastikan server jalan dan model tersedia. "
                           f"Detail: {e}") from e
    return resp.json()


def _parse_json(content: str) -> dict:
    """JSON object dari output LLM. Kalau tidak terbaca (atau bukan object), kembalikan hasil
    not_found dengan `parse_error` supaya pipeline menandainya, bukan crash."""
    candidates = [content]
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        candidates.append(m.group(0))
    for text in candidates:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {"answer": "", "status": "not_found", "sources": [], "parse_error": content[:500]}


def chat_json(system: str, user: str, schema: dict, model: str | None = None) -> dict:
    s = get_settings()
    model = model or s.llm_model
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if s.llm_provider == "ollama":
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": schema,  # Ollama memaksa output mengikuti JSON schema
            "options": {"temperature": s.temperature},
        }
        # Hanya dikirim jika di-set: model tanpa mode thinking menolak parameter ini.
        if s.llm_think in ("true", "false"):
            payload["think"] = s.llm_think == "true"
        data = _post(f"{s.llm_base_url}/api/chat", payload, headers={}, timeout=s.request_timeout)
        content = data["message"]["content"]
    elif s.llm_provider == "openai_compat":
        data = _post(f"{s.llm_base_url}/chat/completions", {
            "model": model,
            "messages": messages,
            "temperature": s.temperature,
            "response_format": {"type": "json_object"},
        }, headers={"Authorization": f"Bearer {s.llm_api_key}"}, timeout=s.request_timeout)
        content = data["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"LLM_PROVIDER tidak dikenal: {s.llm_provider}")

    return _parse_json(content)
