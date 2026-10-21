# RAG Lamaran: draf isian form lamaran magang dari dokumen pribadi

> Status: pipeline sudah dijalankan dengan Ollama lokal dan dievaluasi pada 26 pertanyaan berlabel
> (lihat bagian Evaluasi). Bagian bertanda TODO masih perlu diisi.

## Masalah

Saat mencari magang, saya mengisi form lamaran berulang kali dengan informasi yang sama. Sebagian
pertanyaan juga membuat saya membuka CV, transkrip, sertifikat, dan laporan project untuk mencari jawaban
yang relevan dengan posisi. Sistem ini mencari jawaban itu dari dokumen saya sendiri, menyusun draf isian,
dan menyertakan sumbernya, jadi saya tinggal melakukan QC.

## Scope

- Sistem ini asisten di sisi pelamar. Form yang dipakai (`web/index.html`) adalah simulasi dan tidak
  terhubung ke website perusahaan mana pun.
- Sistem hanya membuat draf dan tidak pernah submit otomatis.
- Di luar scope: integrasi ke website lamaran asli (misalnya lewat browser extension) dan OCR untuk dokumen hasil scan.

## Dokumen

| Folder | Isi | Status git |
|---|---|---|
| `data/dummy/` | Persona sintetis "Raka Pratama" (10 dokumen .md, ditulis manual untuk demo & tes) | di-commit |
| `data/private/` | Dokumen asli saya (PDF) | di-ignore, tidak pernah di-commit |

Tipe dokumen diambil dari nama subfolder: `cv`, `transkrip`, `sertifikat`, `project`, `organisasi`, `esai`.
Semua nama, institusi, email, dan nomor di data dummy fiktif.

TODO: isi jumlah dokumen, tipe file, dan ukuran koleksi asli yang dipakai saat evaluasi.

## Arsitektur

```
Fase 1: indexing (python cli.py ingest)
  data/<tipe>/*.pdf|md|txt
    -> ekstraksi teks per halaman (PyMuPDF)
    -> chunking sesuai tipe (per section / per dokumen / per paragraf) + metadata
    -> embedding (Ollama) + BM25
    -> storage/: chunks.jsonl, vectors.npy, profile.json, meta.json

Fase 2: isi form (python cli.py autofill / web)
  tiap field form
    -> fixed   : profile.json (regex dari CV, tanpa LLM)
    -> factual : RAG, jawaban singkat
    -> open    : RAG, jawaban paragraf (posisi yang dilamar ikut di query)
    -> validasi (sumber valid? didukung teks sumber? format benar?)
    -> draf + status (ok / needs_review / not_found) + sumber -> QC manual
```

CLI tanya-jawab dan form memakai fungsi inti yang sama, `rag.pipeline.answer()`.

| Modul | Tugas |
|---|---|
| `rag/loaders.py` | Ekstraksi teks per halaman dari PDF/MD/TXT |
| `rag/chunking.py` | Strategi chunking per tipe dokumen; chunk tidak menyeberang halaman |
| `rag/profile.py` | Regex untuk nama, email, HP, LinkedIn, GitHub |
| `rag/bm25.py`, `rag/embeddings.py`, `rag/index.py` | Index keyword + vektor |
| `rag/retrieve.py` | Hybrid retrieval dengan Reciprocal Rank Fusion |
| `rag/prompts.py`, `rag/llm.py` | Prompt, format konteks, panggilan LLM (output JSON) |
| `rag/validate.py` | Cek sumber, grounding heuristik, cek relevansi, format isian |
| `rag/autofill.py` | Routing per field berdasarkan `forms/magang_form.json` |
| `cli.py`, `api.py`, `web/index.html` | Antarmuka |

## Model dan tools

| Komponen | Default | Alasan |
|---|---|---|
| LLM | `qwen3:4b-instruct` via Ollama (2,5 GB) | Gratis dan lokal, jadi dokumen pribadi tidak keluar dari laptop. Muat di VRAM 4 GB. Varian instruct ini tidak punya mode thinking, sehingga lebih cepat dan output JSON-nya lebih stabil |
| Embedding | `qwen3-embedding:0.6b` via Ollama (639 MB) | Skor MTEB Multilingual tertinggi di kelas ~0.6B (mean 64.33, retrieval 64.64; bge-m3: 59.56 / 54.60). Query diberi instruksi tugas sesuai model card |
| Vector store | Matriks numpy (exact search) | Corpus-nya kecil (puluhan dokumen), jadi pencarian exact cukup cepat, tanpa approximation error, dan mudah diinspeksi |
| Keyword | BM25 implementasi sendiri | Nama skill/tools (SQL, FastAPI) adalah keyword exact |
| PDF | PyMuPDF | Ekstraksi teks per halaman untuk sitasi |

Semua bisa diganti lewat `.env`. TODO: catat versi persis (`ollama --version`, `ollama show <model>`).

### Kandidat LLM yang dipertimbangkan (riset Oktober 2026)

Batasannya: VRAM 4 GB, paham bahasa Indonesia, output JSON stabil, dan idealnya tanpa mode thinking.

| Model | Ukuran Ollama | Bukti yang ada | Catatan |
|---|---|---|---|
| `qwen3:4b-instruct` (2507) | 2,5 GB | IFEval 83.4, MMLU-ProX 61.6, INCLUDE 60.1 (model card) | Muat di VRAM, tanpa thinking. Default (dikonfirmasi di Evaluasi) |
| `qwen3.5:4b` | 3,3-4,0 GB | IFEval 89.8, MMLU-ProX 71.5, INCLUDE 71.0 (model card, mode tidak dilabeli) | Thinking aktif secara default. Ada laporan `format` JSON diabaikan saat thinking dimatikan (Ollama issue #14645, varian 35B) |
| `gemma4:e2b` | 4,6-7,5 GB | SEA-HELM Indonesia 64.13, instruction following 90.23 | Satu-satunya model kecil dengan skor bahasa Indonesia terverifikasi, tapi tidak muat utuh di VRAM 4 GB |
| `llama3.2:3b`, `phi4-mini` | ~2-2,5 GB | - | Bahasa Indonesia tidak ada di daftar bahasa resminya, jadi tidak dipilih |

Angka di atas diambil dari model card dan leaderboard yang berbeda, jadi tidak bisa dibandingkan langsung.
Perbandingan di kondisi yang sama ada di bagian Evaluasi.

## Setup

Kebutuhan: Python 3.11+, [Ollama](https://ollama.com), dan sekitar 3,2 GB disk untuk kedua model.
Dikembangkan di Ryzen 5 3500X, RAM 16 GB, GTX 1650 (VRAM 4 GB). Kalau masih terlalu berat, pakai
`LLM_MODEL=qwen3:1.7b` dan/atau `RETRIEVAL_MODE=bm25` (tanpa model embedding sama sekali).

```bash
ollama pull qwen3:4b-instruct
ollama pull qwen3-embedding:0.6b
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Cara menjalankan

```bash
python cli.py ingest                                   # bangun index
python cli.py chunks --type cv                         # cek hasil chunking
python cli.py retrieve "sertifikat machine learning"   # retrieval saja, tanpa LLM
python cli.py ask "Berapa IPK saya?" --type transkrip --short --save
python cli.py autofill --position "Data Analyst Intern" --save
python cli.py serve                                    # buka http://127.0.0.1:8000
python -m tests.test_offline                           # tes tanpa Ollama
python -m evaluation.run_eval                          # evaluasi (lihat bagian Evaluasi)
```

`--save` menyimpan output lengkap (chunk ter-retrieve, skor, jawaban, sumber, setting model) ke `outputs/examples/`.
Tombol "Kirim (simulasi)" di web menyimpan draf dan hasil QC ke `storage/qc_log.jsonl`.

## Konfigurasi

| Variabel | Default | Keterangan |
|---|---|---|
| `DATA_DIR` | `data/dummy` | Folder dokumen |
| `STORAGE_DIR` | `storage` | Lokasi index & log |
| `LLM_PROVIDER` | `ollama` | `ollama` atau `openai_compat` |
| `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` | lihat `.env.example` | API key hanya untuk `openai_compat` |
| `LLM_THINK` | kosong | `false` untuk mematikan thinking di model hybrid (qwen3.5, gemma4); biarkan kosong untuk model tanpa thinking |
| `RELEVANCE_CHECK`, `RELEVANCE_MODEL` | `true`, kosong | Cek apakah jawaban membahas hal yang ditanyakan (panggilan LLM tambahan); model kosong = `LLM_MODEL` |
| `EMBED_MODEL`, `EMBED_BASE_URL` | `qwen3-embedding:0.6b`, Ollama lokal | |
| `EMBED_QUERY_INSTRUCTION` | instruksi tugas (lihat `.env.example`) | Dipakai di query saja; kosongkan untuk embedding tanpa instruksi |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid`, `bm25`, atau `vector` |
| `TOP_K` | `5` | Jumlah chunk ke LLM |
| `MAX_CHUNK_CHARS` | `1200` | Batas panjang chunk |
| `MAX_CONTEXT_CHARS` | `6000` | Batas total konteks ke LLM |

## Contoh output

TODO: jalankan beberapa pertanyaan dengan `--save`, lalu rangkum di sini: pertanyaan, chunk yang ter-retrieve,
jawaban, dan sumber. Sertakan contoh yang berhasil, yang gagal, dan satu pertanyaan tanpa bukti
(misalnya "project computer vision", yang memang tidak ada di data dummy).

## Keputusan desain

- Chunking mengikuti tipe dokumen. CV dan laporan project punya heading yang jelas, jadi dipotong per section.
  Sertifikat pendek, jadi satu dokumen jadi satu chunk. Judul dokumen dan nama section ikut di-index.
- Chunk tidak pernah menyeberang halaman, supaya nomor halaman di sitasi tepat.
- Isian tetap (nama, email, HP) diambil dengan regex, bukan LLM. Cara ini lebih cepat, hasilnya selalu sama,
  dan tidak bisa berhalusinasi.
- LLM wajib menyebut `chunk_id` sumbernya. Sumber yang tidak ada di hasil retrieve ditandai `sumber_tidak_valid`.
- Grounding dicek secara heuristik: jawaban dibandingkan dengan teks sumber (substring / overlap token), dan
  jawaban di bawah ambang ditandai `needs_review`.
- INFO POSISI dipisah dari KONTEKS supaya deskripsi lowongan tidak dianggap bukti tentang pelamar.

## Keterbatasan yang sudah diketahui

- PDF hasil scan tidak punya teks. File seperti ini dilewati dengan peringatan (perlu OCR atau versi .md).
- CV dua kolom bisa tercampur urutannya saat diekstrak; cek dengan `python cli.py chunks`.
- Deteksi heading berbasis kata kunci dan huruf kapital, jadi heading dengan nama tak lazim tidak terdeteksi.
- Ambang grounding (0.8 / 0.6) belum dievaluasi.
- Tes offline memakai stub. Tes itu hanya membuktikan alurnya berjalan, bukan kualitas retrieval atau jawaban.

## Evaluasi

Set uji ada di [`evaluation/questions.json`](evaluation/questions.json): 26 pertanyaan tentang persona dummy.

| Kategori | Jumlah | Contoh |
|---|---|---|
| `factual` | 12 | "Berapa IPK kumulatif saya?" |
| `cross_doc` (jawaban tersebar di beberapa dokumen) | 3 | "Sertifikat apa saja dari Akademi Data Nusantara?" |
| `open` (jawaban paragraf) | 5 | "Ceritakan project yang paling relevan untuk posisi Data Analyst Intern." |
| `unanswerable` (jawabannya tidak ada, beberapa berupa jebakan) | 6 | "Berapa skor IELTS saya?" (yang ada hanya TOEFL) |

Tiap pertanyaan punya `gold` (dokumen + potongan teks yang membuktikan jawaban), `must_include`
(istilah yang wajib ada di jawaban), dan `must_not_include` (opsional). Retrieval dievaluasi tanpa filter tipe dokumen.

```bash
python -m evaluation.run_eval --check            # cek label cocok dengan chunk (tanpa Ollama)
python -m evaluation.run_eval --retrieval-only   # bm25 vs vector vs hybrid
python -m evaluation.run_eval                    # retrieval + semua model di evaluation/models.json
python -m evaluation.run_eval --models qwen3:4b-instruct --limit 5   # uji cepat
```

Untuk retrieval, metriknya hit@k, cakupan gold, dan MRR. Untuk generation: akurasi pada pertanyaan yang bisa
dijawab, ketepatan abstain (`not_found`), salah abstain, halusinasi yang tertangkap validasi, persentase JSON
valid, ketepatan sitasi, median detik per jawaban, dan porsi model yang ada di GPU (dari `ollama ps`).
Hasil tersimpan di `outputs/eval/<waktu>/summary.md`. Model yang dibandingkan diatur di
[`evaluation/models.json`](evaluation/models.json). Tiap model perlu di-`ollama pull`, dan `gemma4:e2b`
sendiri berukuran 4,6 GB ke atas.

Hasil retrieval (diamati 18 Okt 2026, `qwen3-embedding:0.6b`, k=5, 20 pertanyaan yang bisa dijawab):

| Mode | Hit@5 | Cakupan gold | MRR |
|---|---|---|---|
| bm25 | 100% | 95% | 0.883 |
| vector | 100% | 93% | 0.787 |
| hybrid | 100% | 98% | 0.917 |

- f03 "program studi": CV menulis "S1 Teknik Informatika" tanpa kata "program studi". BM25 hanya menemukan
  transkrip, sedangkan vector dan hybrid menemukan CV juga.
- c01 "sertifikat dari Akademi Data Nusantara": vector hanya menemukan 1 dari 2 (rank 5). Nama lembaga
  adalah keyword exact, jadi BM25 dan hybrid lebih baik di sini.
- c02 "pekerjaan dengan FastAPI": vector menemukan keduanya, tetapi di hybrid chunk freelance di CV
  terdorong keluar dari top-5. RRF tidak selalu lebih baik dari komponen terbaiknya.

Angka ini kemungkinan terlalu optimis. Dokumen dummy dan pertanyaannya ditulis oleh orang yang sama,
dan corpus-nya kecil (34 chunk), sehingga top-5 sudah mencakup sekitar 15% corpus.

Perbandingan tiga LLM (diamati 19 Okt 2026, satu run, Ollama 0.40.6, retrieval hybrid, k=5, temperature 0,
GTX 1650 4 GB). Laporan lengkapnya di `outputs/eval/20261019-195450/summary.md`.

| Model | Akurasi (20 bisa dijawab) | Abstain benar (6) | Halusinasi tertangkap validasi | JSON valid | Sitasi benar | Median per jawaban | Di GPU |
|---|---|---|---|---|---|---|---|
| `qwen3:4b-instruct` | 90% (18/20) | 83% (5/6) | 1/1 | 100% | 95% | 8,5 detik | 66% |
| `qwen3.5:4b` (thinking off) | 90% (18/20) | 83% (5/6) | 0/1 | 100% | 85% | 9,2 detik | 48% |
| `gemma4:e2b` (thinking off) | 95% (19/20) | 50% (3/6) | 0/3 | 100% | 90% | 6,0 detik | tidak terukur* |

\* `ollama ps` melaporkan hanya 0,32 GB untuk `gemma4:e2b`, jauh di bawah ukuran unduhannya (4,6 GB), jadi angka GPU-nya tidak dipakai.

Temuan:
- Akurasi pada pertanyaan yang bisa dijawab hampir sama. Selisihnya 1 soal (5 poin persen), dan dengan 20 soal
  itu belum bisa disebut perbedaan nyata.
- Perbedaan terbesar ada di pertanyaan yang jawabannya tidak ada. `gemma4:e2b` mengarang di 3 dari 6, termasuk
  menjawab "TOEFL ITP 540" untuk pertanyaan skor IELTS. Tidak satu pun tertangkap validasi karena isinya memang
  ada di dokumen. Untuk form lamaran, isian yang salah tapi terlihat yakin lebih berbahaya daripada isian kosong.
- Validasi hanya mengecek apakah jawaban didukung sumber, bukan apakah jawaban relevan dengan pertanyaan.
  Semua halusinasi yang lolos (`qwen3.5` u02, `gemma4` u01/u02/u03) adalah fakta benar yang menjawab pertanyaan lain.
- `gemma4:e2b` paling sering menyebut `chunk_id` yang tidak ada (f04, o02, o05), sehingga jawabannya ditandai
  `needs_review` walaupun isinya benar.
- Jawaban stabil antar run selama urutan panggilan ke model sama. Teks jawaban sama persis (0/26 berubah)
  untuk `qwen3:4b-instruct` di run 18, 19, dan 20 Okt, dan untuk `qwen3.5:4b` serta `gemma4:e2b` di run 19 dan
  20 Okt. Satu-satunya kondisi yang membuat jawaban berubah adalah cek relevansi yang diselipkan di antara soal
  (lihat bagian Cek relevansi).

Keputusan: `qwen3:4b-instruct` tetap jadi default, walaupun selisihnya dengan `qwen3.5:4b` kecil. Akurasi
keduanya sama (90%), dan selisih 1 soal pun sebaiknya tidak dianggap berarti dengan 20 soal. `qwen3:4b-instruct`
dipilih karena sitasinya paling tepat, porsinya di GPU lebih besar (66% vs 48%), dan lebih cepat.
`gemma4:e2b` paling cepat, tetapi terlalu sering mengarang pada pertanyaan yang jawabannya tidak ada.

Analisis kegagalan `qwen3:4b-instruct`:

| ID | Yang terjadi | Sumber masalah |
|---|---|---|
| c02 | Hanya menyebut sistem inventaris; website katalog UMKM (freelance) tidak disebut | Retrieval: chunk Pengalaman di CV tidak masuk top-5 hybrid (gagal juga di dua model lain) |
| o03 | Menyebut "2 jam jadi 20 menit" tetapi melewatkan "150 alat", padahal chunk Hasil ikut ter-retrieve | Generation: jawaban tidak lengkap |
| u02 | Menjawab "Saya tidak pernah magang" (menyimpulkan dari ketiadaan bukti), bukan `not_found` | Generation, dan labelnya bisa diperdebatkan. Validasi menandai `needs_review` karena dukungan sumber rendah |
| c03 | Dinilai benar oleh grader, tetapi menyebut "mata kuliah Machine Learning dengan nilai 88/100". Nilai 88 milik sertifikat, nilai mata kuliahnya A | Generation (mencampur fakta). Grader berbasis kata kunci tidak menangkap ini; dua model lain tidak melakukannya |
| f08 | Jawaban benar ("A-") tetapi `chunk_id` yang disebut tidak ada di hasil retrieve (run 18 Okt) | Generation (format sitasi). Validasi bekerja sesuai desain |

Catatan memori: model embedding ikut termuat di VRAM (1,23 GB), sehingga hanya 66% `qwen3:4b-instruct` yang muat
di GPU. Memuat model pertama kali butuh 26-101 detik, tergantung modelnya.

### Cek relevansi (diamati 19 Okt 2026)

Semua halusinasi yang lolos validasi di atas adalah fakta yang ada di dokumen tetapi menjawab pertanyaan lain
(ditanya IELTS dijawab TOEFL; ditanya project computer vision dijawab project lain). `support_ratio` tidak bisa
menangkap ini karena jawabannya memang didukung sumber.

Yang dicoba:
1. Cek kata kunci, yaitu menandai pertanyaan yang memuat istilah yang tidak ada di dokumen mana pun. Cara ini
   ditolak karena akan menandai 14 dari 20 pertanyaan yang bisa dijawab: kata umum seperti "kapan", "sudah",
   dan "rencana" juga tidak ada di dokumen.
2. LLM sebagai pemeriksa (`rag/validate.py: check_relevance`). Model yang sama dipanggil sekali lagi dan hanya
   melihat pertanyaan + jawaban. Kalau jawabannya membahas hal lain, jawaban ditandai `jawaban_tidak_relevan`
   (status `needs_review`). Contoh di prompt pemeriksa sengaja dipilih yang tidak ada di set evaluasi.

Run pertama dengan pemeriksa (`20261019-213650`) menjalankan cek relevansi langsung setelah tiap jawaban.
Ternyata 6/26 (`qwen3`), 7/26 (`qwen3.5`), dan 2/26 (`gemma4`) jawaban berubah teksnya dibanding run tanpa cek,
walaupun cek itu sendiri tidak menyentuh jawaban. Akurasi `qwen3.5` bahkan naik dari 90% ke 95% karena c01
kebetulan berubah menjadi benar.

Untuk menguji dugaan bahwa panggilan pemeriksa di antara soal memengaruhi jawaban berikutnya, evaluasi
dijalankan dua tahap (`--relevance-stage separate`, default): semua soal di-generate dulu, baru semua jawaban
diperiksa. Di run `20261020-004417`, teks jawaban ketiga model sama persis dengan run tanpa cek (0/26 berubah).
Jadi penyebabnya adalah panggilan yang diselipkan, walaupun mekanisme pastinya (dugaan: cache prompt di Ollama)
belum dibuktikan. Autofill di aplikasi memakai urutan dua tahap yang sama.

Hasil cek relevansi dari run dua tahap `20261020-004417` (jawabannya identik dengan run tanpa cek):

| Model | Jawaban salah yang tertangkap cek relevansi | Jawaban benar yang ikut ditandai | Median detik per jawaban (tanpa -> dengan cek) |
|---|---|---|---|
| `qwen3:4b-instruct` | 0/3 | 3/18 (f07, f08, f12) | 8,5 -> 14,4 |
| `qwen3.5:4b` | 2/3 (c01, u02) | 1/18 (f12) | 9,2 -> 15,1 |
| `gemma4:e2b` | 2/4 (u01, u03) | 2/19 (f05, f12) | 6,0 -> 9,3 |

"Jawaban salah" berarti jawaban yang dinilai salah oleh grader, baik untuk soal yang bisa dijawab maupun yang tidak.

- Pemeriksa menangkap kasus yang paling berbahaya: "TOEFL ITP 540" untuk skor IELTS (`gemma4`), dan pengalaman
  asisten lab yang disajikan sebagai magang (`qwen3.5`).
- Ketiga jawaban salah `qwen3` (c02 dan o03 tidak lengkap; u02 "tidak pernah magang") memang menjawab
  pertanyaannya, jadi wajar tidak ditandai. u02 tetap tertangkap oleh cek dukungan sumber.
- Pemeriksa yang sama meloloskan u02 milik `gemma4`.
- Semua jawaban benar yang ikut ditandai adalah jawaban singkat tanpa konteks, misalnya "40" tanpa kata "jam"
  atau "Kelas Koding Indonesia" yang tidak dikenali sebagai nama lembaga.
- Biayanya sekitar 3,5-6 detik tambahan per jawaban.

Keputusan: `RELEVANCE_CHECK=true` jadi default. Untuk form lamaran, isian yang salah tetapi terlihat yakin
lebih berbahaya daripada satu isian tambahan yang perlu dicek. Di PC yang lebih lambat, cek ini bisa dimatikan.
Perbaikan prompt pemeriksa untuk jawaban singkat sebaiknya diuji pada set pertanyaan baru, bukan set ini,
supaya prompt-nya tidak disesuaikan dengan soal ujian.

## Eksperimen

Sudah dijalankan (hasilnya di bagian Evaluasi):
1. Retrieval `bm25` vs `vector` vs `hybrid`.
2. LLM kecil di VRAM 4 GB: `qwen3:4b-instruct` vs `qwen3.5:4b` vs `gemma4:e2b`.
3. Cek relevansi jawaban terhadap pertanyaan (cek kata kunci ditolak, pemeriksa LLM dipakai).
4. Cek relevansi inline vs dua tahap (dua tahap membuat jawaban identik dengan run tanpa cek).

TODO:
5. Autofill full-context (semua dokumen masuk prompt) vs retrieval per field.
6. Tingkat edit setelah QC dari `storage/qc_log.jsonl`.
7. Instruksi query embedding: dengan vs tanpa `EMBED_QUERY_INSTRUCTION` (recall@k).
8. Embedding dijalankan di CPU supaya LLM mendapat seluruh VRAM (kecepatan vs akurasi retrieval).
9. Set pertanyaan kedua (held-out) untuk menguji perbaikan prompt pemeriksa tanpa menyesuaikan dengan set ini.
10. Mekanisme perubahan jawaban saat cek diselipkan (cache prompt Ollama?). Penyebabnya sudah terisolasi:
    dengan evaluasi dua tahap, jawaban kembali identik.

## Rencana selanjutnya

TODO.

## Penggunaan AI

Kerangka kode awal dibuat dengan bantuan Claude Code (Anthropic).
TODO: jelaskan bagian mana yang saya cek, ubah, dan kembangkan sendiri.
