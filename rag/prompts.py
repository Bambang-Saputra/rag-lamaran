"""Prompt dan format konteks untuk generation."""
from __future__ import annotations

SYSTEM_PROMPT = """Kamu membantu pelamar magang menyiapkan draf isian form lamaran dari dokumen milik pelamar sendiri.

Aturan:
1. Jawab HANYA berdasarkan potongan dokumen di bagian KONTEKS. Jangan memakai pengetahuan umum dan jangan mengarang.
2. Isi "sources" dengan chunk_id dari setiap potongan yang benar-benar kamu pakai.
3. Jika KONTEKS tidak memuat jawabannya, isi "status": "not_found", "answer": "", dan "sources": [].
4. INFO POSISI (jika ada) hanya menjelaskan lowongan, BUKAN bukti tentang pelamar.
5. Balas dalam JSON: {"answer": string, "status": "answered" | "not_found", "sources": [chunk_id, ...]}"""

STYLE_HINTS = {
    "short": "Jawab singkat: tulis nilainya saja (misalnya nama institusi, angka, atau daftar dipisah koma).",
    "long": "Jawab dalam 2-4 kalimat dari sudut pandang orang pertama (saya), siap ditempel di form, "
            "hanya memakai fakta dari KONTEKS.",
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "status": {"type": "string", "enum": ["answered", "not_found"]},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "status", "sources"],
}


# Pemeriksa relevansi hanya melihat pertanyaan + jawaban, bukan dokumen. Contoh di prompt sengaja
# dipilih yang tidak ada di set evaluasi (evaluation/questions.json) supaya hasil evaluasi tidak bocor.
RELEVANCE_SYSTEM = """Kamu pemeriksa draf isian form. Nilai HANYA apakah JAWABAN benar-benar membahas hal spesifik yang ditanyakan PERTANYAAN.
Jangan menilai apakah faktanya benar.
Jawaban yang membahas hal lain yang mirip tetapi bukan yang ditanyakan dianggap TIDAK relevan.
Contoh: ditanya nomor paspor, dijawab nomor KTP -> tidak relevan. Ditanya hobi, dijawab "Saya suka membaca" -> relevan.
Balas dalam JSON: {"relevant": true | false, "reason": string singkat}"""

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["relevant", "reason"],
}


def build_relevance_prompt(question: str, answer_text: str) -> str:
    return f"PERTANYAAN: {question}\n\nJAWABAN: {answer_text}"


def format_context(chunks: list[dict], max_chars: int) -> tuple[str, list[str]]:
    """Gabungkan chunk sesuai urutan rank sampai batas max_chars. Kembalikan teks dan chunk_id yang masuk."""
    blocks, used, total = [], [], 0
    for c in chunks:
        block = (f"[chunk_id: {c['chunk_id']} | file: {c['file']} | hlm {c['page']} | "
                 f"section: {c['section']}]\n{c['text']}")
        if blocks and total + len(block) > max_chars:
            break
        blocks.append(block)
        used.append(c["chunk_id"])
        total += len(block)
    return "\n\n---\n\n".join(blocks), used


def build_user_prompt(question: str, context: str, style: str, task_context: str = "") -> str:
    parts = [f"KONTEKS:\n{context}"]
    if task_context:
        parts.append(f"INFO POSISI:\n{task_context}")
    parts.append(f"PERTANYAAN: {question}")
    parts.append(STYLE_HINTS[style])
    return "\n\n".join(parts)
