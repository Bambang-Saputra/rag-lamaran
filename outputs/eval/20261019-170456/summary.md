# Hasil evaluasi (2026-10-19T17:10:49)

Data: `data/dummy` (10 dokumen, 34 chunk). Embedding: `qwen3-embedding:0.6b`. Retrieval untuk generation: `hybrid`, TOP_K=5. 26 pertanyaan (20 bisa dijawab, 6 tidak).

## Retrieval

| Mode | Hit@k | Cakupan gold | MRR |
|---|---|---|---|
| bm25 | 100% | 95% | 0.883 |
| vector | 100% | 93% | 0.787 |
| hybrid | 100% | 98% | 0.917 |

## Generation

| Model | Akurasi (bisa dijawab) | Abstain benar | Salah abstain | Halusinasi tertangkap validasi | JSON valid | Sitasi benar | Median detik | Di GPU |
|---|---|---|---|---|---|---|---|---|
| `qwen3.5:4b` | 90% | 83% | 0% | 0% | 100% | 85% | 8.8 | 48% |

### Per pertanyaan

| ID | Kategori | `qwen3.5:4b` |
|---|---|---|
| f01 | factual | benar |
| f02 | factual | benar |
| f03 | factual | benar |
| f04 | factual | benar |
| f05 | factual | benar |
| f06 | factual | benar |
| f07 | factual | benar |
| f08 | factual | benar |
| f09 | factual | benar |
| f10 | factual | benar |
| f11 | factual | benar |
| f12 | factual | benar |
| c01 | cross_doc | salah |
| c02 | cross_doc | salah |
| c03 | cross_doc | benar |
| o01 | open | benar |
| o02 | open | benar |
| o03 | open | benar |
| o04 | open | benar |
| o05 | open | benar |
| u01 | unanswerable | benar |
| u02 | unanswerable | salah |
| u03 | unanswerable | benar |
| u04 | unanswerable | benar |
| u05 | unanswerable | benar |
| u06 | unanswerable | benar |

