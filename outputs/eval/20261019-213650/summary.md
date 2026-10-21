# Hasil evaluasi (2026-10-19T21:58:43)

Data: `data/dummy` (10 dokumen, 34 chunk). Embedding: `qwen3-embedding:0.6b`. Retrieval untuk generation: `hybrid`, TOP_K=5. 26 pertanyaan (20 bisa dijawab, 6 tidak).

## Retrieval

| Mode | Hit@k | Cakupan gold | MRR |
|---|---|---|---|
| bm25 | 100% | 95% | 0.883 |
| vector | 100% | 93% | 0.787 |
| hybrid | 100% | 98% | 0.917 |

## Generation (cek relevansi: inline)

| Model | Akurasi (bisa dijawab) | Abstain benar | Salah abstain | Halusinasi tertangkap validasi | Relevansi: jawaban salah tertangkap | Relevansi: jawaban benar ikut ditandai | JSON valid | Sitasi benar | Median detik (generate) | Median detik (total) | Di GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3:4b-instruct` | 90% | 83% | 0% | 100% | 0% | 17% | 100% | 95% | 8.9 | 14.4 | 66% |
| `qwen3.5:4b` | 95% | 83% | 0% | 100% | 0% | 5% | 100% | 85% | 9.5 | 15.5 | 48% |
| `gemma4:e2b` | 95% | 50% | 0% | 67% | 50% | 10% | 100% | 90% | 5.8 | 9.3 | 100% |

### Per pertanyaan

| ID | Kategori | `qwen3:4b-instruct` | `qwen3.5:4b` | `gemma4:e2b` |
|---|---|---|---|---|
| f01 | factual | benar | benar | benar |
| f02 | factual | benar | benar | benar |
| f03 | factual | benar | benar | benar |
| f04 | factual | benar | benar | benar |
| f05 | factual | benar | benar | benar |
| f06 | factual | benar | benar | benar |
| f07 | factual | benar | benar | benar |
| f08 | factual | benar | benar | benar |
| f09 | factual | benar | benar | benar |
| f10 | factual | benar | benar | benar |
| f11 | factual | benar | benar | benar |
| f12 | factual | benar | benar | benar |
| c01 | cross_doc | benar | benar | benar |
| c02 | cross_doc | salah | salah | salah |
| c03 | cross_doc | benar | benar | benar |
| o01 | open | benar | benar | benar |
| o02 | open | benar | benar | benar |
| o03 | open | salah | benar | benar |
| o04 | open | benar | benar | benar |
| o05 | open | benar | benar | benar |
| u01 | unanswerable | benar | benar | salah |
| u02 | unanswerable | salah | salah | salah |
| u03 | unanswerable | benar | benar | salah |
| u04 | unanswerable | benar | benar | benar |
| u05 | unanswerable | benar | benar | benar |
| u06 | unanswerable | benar | benar | benar |

