"""Semua setting model, provider, dan retrieval dibaca dari environment / file .env."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Parser .env sederhana (KEY=VALUE per baris). Env yang sudah di-set tidak ditimpa."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def display_path(p: Path) -> str:
    """Path relatif terhadap root project untuk disimpan di output (tanpa struktur folder lokal)."""
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _path(key: str, default: str) -> Path:
    p = Path(os.environ.get(key, default))
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    storage_dir: Path
    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    temperature: float
    llm_think: str  # "" = tidak dikirim, "true"/"false" = parameter think di Ollama
    relevance_check: bool
    relevance_model: str  # kosong = pakai LLM_MODEL
    embed_model: str
    embed_base_url: str
    embed_query_instruction: str
    retrieval_mode: str
    top_k: int
    max_chunk_chars: int
    max_context_chars: int
    request_timeout: int
    keep_alive: str  # berapa lama Ollama menyimpan model di memori setelah dipakai, mis. "30m"; kosong = default Ollama

    def public(self) -> dict:
        """Snapshot setting untuk disimpan bersama output (tanpa API key)."""
        d = {k: display_path(v) if isinstance(v, Path) else str(v) for k, v in asdict(self).items()}
        d.pop("llm_api_key")
        return d


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        data_dir=_path("DATA_DIR", "data/dummy"),
        storage_dir=_path("STORAGE_DIR", "storage"),
        llm_provider=env("LLM_PROVIDER", "ollama"),
        llm_model=env("LLM_MODEL", "qwen3:4b-instruct"),
        llm_base_url=env("LLM_BASE_URL", "http://localhost:11434").rstrip("/"),
        llm_api_key=env("LLM_API_KEY", ""),
        temperature=float(env("TEMPERATURE", "0")),
        llm_think=env("LLM_THINK", "").strip().lower(),
        relevance_check=env("RELEVANCE_CHECK", "true").strip().lower() in ("1", "true", "yes"),
        relevance_model=env("RELEVANCE_MODEL", ""),
        embed_model=env("EMBED_MODEL", "qwen3-embedding:0.6b"),
        embed_base_url=env("EMBED_BASE_URL", "http://localhost:11434").rstrip("/"),
        embed_query_instruction=env(
            "EMBED_QUERY_INSTRUCTION",
            "Given a question about a job applicant, retrieve passages from the applicant's documents that answer it",
        ),
        retrieval_mode=env("RETRIEVAL_MODE", "hybrid"),
        top_k=int(env("TOP_K", "5")),
        max_chunk_chars=int(env("MAX_CHUNK_CHARS", "1200")),
        max_context_chars=int(env("MAX_CONTEXT_CHARS", "6000")),
        request_timeout=int(env("REQUEST_TIMEOUT", "120")),
        keep_alive=env("KEEP_ALIVE", "30m").strip(),
    )
