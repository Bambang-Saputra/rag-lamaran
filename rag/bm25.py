"""BM25 (Okapi) sederhana. Corpus kecil, jadi skor dihitung ulang tiap query tanpa library tambahan."""
from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Kata tanya/penghubung umum ID+EN yang hanya menambah noise pada skor keyword.
STOPWORDS = {
    "apa", "apakah", "saja", "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "atau", "ini", "itu",
    "saya", "aku", "kamu", "pelamar", "berapa", "siapa", "mana", "bagaimana", "kenapa", "mengapa",
    "sebutkan", "ceritakan", "jelaskan", "pernah", "punya", "memiliki", "dimiliki", "adalah", "dalam",
    "pada", "sebagai", "posisi", "the", "a", "an", "of", "and", "or", "to", "in", "for", "with",
    "what", "which", "who", "how", "is", "are", "my", "i",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(t)) for t in texts]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(texts)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out
