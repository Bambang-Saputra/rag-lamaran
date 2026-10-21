# Hasil evaluasi (2026-10-18T23:24:34)

Data: `data/dummy` (10 dokumen, 34 chunk). Embedding: `qwen3-embedding:0.6b`. Retrieval untuk generation: `hybrid`, TOP_K=5. 26 pertanyaan (20 bisa dijawab, 6 tidak).

## Retrieval

| Mode | Hit@k | Cakupan gold | MRR |
|---|---|---|---|
| bm25 | 100% | 95% | 0.883 |
| vector | 100% | 93% | 0.787 |
| hybrid | 100% | 98% | 0.917 |

