# Sistem Pengisian Form Lamaran Otomatis

Asisten RAG lokal yang menyusun draf isian form lamaran magang dari dokumen saya sendiri (CV, transkrip,
sertifikat, laporan project, organisasi, esai), lengkap dengan sumbernya, supaya saya tinggal melakukan QC.
Semua berjalan di laptop dengan Ollama, tanpa API key. Nama repo: `rag-lamaran`.

Ringkasan hasil pada 26 pertanyaan berlabel (data dummy, satu run per model):

- Retrieval hybrid (BM25 + vektor, digabung dengan RRF): Hit@5 100%, cakupan bukti 98%, MRR 0,917.
- `qwen3:4b-instruct` (default): akurasi 90% pada 20 soal yang bisa dijawab, abstain benar pada 5 dari 6 soal
  yang jawabannya tidak ada, sitasi benar 95%, median 8,5 detik per jawaban di GTX 1650 4 GB.
- Cek relevansi menangkap jawaban yang didukung sumber tetapi menjawab hal lain, misalnya "TOEFL ITP 540" untuk
  pertanyaan skor IELTS.
- Angka ini kemungkinan terlalu optimis: dokumen dan pertanyaannya ditulis oleh orang yang sama, dan corpus-nya
  hanya 34 chunk.

Isi:
[Masalah](#masalah) ·
[Scope](#scope) ·
[Dokumen](#dokumen) ·
[Arsitektur](#arsitektur) ·
[Model dan tools](#model-dan-tools) ·
[Setup](#setup) ·
[Cara menjalankan](#cara-menjalankan) ·
[Browser extension](#browser-extension) ·
[Konfigurasi](#konfigurasi) ·
[Contoh output](#contoh-output) ·
[Evaluasi](#evaluasi) ·
[Keputusan desain](#keputusan-desain) ·
[Keterbatasan](#keterbatasan) ·
[Rencana selanjutnya](#rencana-selanjutnya) ·
[Penggunaan AI](#penggunaan-ai)

## Masalah

Saat mencari magang, saya mengisi form lamaran berulang kali dengan informasi yang sama. Sebagian pertanyaan
juga membuat saya membuka CV, transkrip, sertifikat, dan laporan project untuk mencari jawaban yang relevan
dengan posisi. Sistem ini mencari jawaban itu dari dokumen saya sendiri, menyusun draf isian, dan menyertakan
sumbernya.

## Scope

- Asisten di sisi pelamar, bukan sistem rekrutmen perusahaan. Form di repo ini (`web/index.html` dan halaman
  `/demo`) adalah simulasi dan tidak terhubung ke website perusahaan mana pun.
- Sistem hanya membuat draf. Tidak ada yang dikirim otomatis: browser extension mengisi draf di halaman, tetapi
  tidak pernah menekan Next atau Submit.
- Semua berjalan lokal, jadi dokumen pribadi tidak keluar dari laptop.
- Di luar scope: OCR untuk dokumen hasil scan.

## Dokumen

| Folder | Isi | Status git |
|---|---|---|
| `data/dummy/` | Persona sintetis "Raka Pratama": 10 file Markdown (sekitar 25 KB), ditulis manual untuk demo dan tes | di-commit |
| `data/private/` | Dokumen asli saya (PDF) | di-ignore, tidak pernah di-commit |

Tipe dokumen diambil dari nama subfolder: `cv`, `transkrip`, `sertifikat`, `project`, `organisasi`, `esai`.
Data dummy berisi 1 CV, 1 transkrip, 4 sertifikat, 2 laporan project, 1 dokumen organisasi, dan 1 esai, yang
menjadi 34 chunk. Semua nama, institusi, email, dan nomor di dalamnya fiktif.

Evaluasi, contoh output, dan demo memakai data dummy. Pipeline juga menerima PDF dan TXT; jalur PDF dites dengan
PDF assessment 13 halaman, dan nomor halaman di sitasinya sesuai. Dokumen asli saya tidak dipakai di angka mana
pun dan tidak ada di repo.

## Arsitektur

```
Fase 1: indexing (python cli.py ingest)
  data/<tipe>/*.pdf|md|txt
    -> ekstraksi teks per halaman (PyMuPDF)
    -> chunking sesuai tipe (per section / per dokumen / per paragraf) + metadata
    -> embedding (Ollama) + BM25
    -> storage/: chunks.jsonl, vectors.npy, profile.json, meta.json

Fase 2: tiap pertanyaan atau kolom form
  kolom form
    -> fixed   : profile.json (regex dari CV, tanpa LLM)
    -> manual  : kesediaan, gaji, jadwal (tidak dikirim ke LLM, diisi sendiri)
    -> factual : RAG, jawaban singkat
    -> open    : RAG, jawaban paragraf (posisi yang dilamar ikut di query)
    -> validasi (sumber valid? didukung teks sumber? relevan? format benar?)
    -> draf + status (ok / needs_review / not_found) + sumber -> QC manual
```

Semua antarmuka (CLI, web, tempel pertanyaan, extension) memakai fungsi inti yang sama, `rag.pipeline.answer()`.
Tiap file di `rag/` mengerjakan satu tahap, jadi tiap tahap bisa dites dan diganti sendiri. Contohnya, evaluasi
retrieval hanya memanggil `retrieve.py` tanpa LLM, dan tes offline mengganti embedding dan LLM dengan stub.

| Modul | Tugas |
|---|---|
| `rag/loaders.py` | Ekstraksi teks per halaman dari PDF/MD/TXT |
| `rag/chunking.py` | Strategi chunking per tipe dokumen; chunk tidak menyeberang halaman |
| `rag/profile.py` | Nama, email, HP, LinkedIn, GitHub dengan regex. Judul seperti "Curriculum Vitae" dilewati saat mencari nama |
| `rag/bm25.py`, `rag/embeddings.py`, `rag/index.py` | Index keyword dan vektor |
| `rag/retrieve.py` | Hybrid retrieval dengan Reciprocal Rank Fusion |
| `rag/prompts.py`, `rag/llm.py` | Prompt, format konteks, panggilan LLM dengan output JSON |
| `rag/validate.py` | Cek sumber, dukungan sumber, relevansi, dan format isian |
| `rag/pipeline.py` | `answer()`: retrieve, generate, validasi |
| `rag/autofill.py` | Routing per kolom dari `forms/magang_form.json`, dan klasifikasi pertanyaan yang ditempel |
| `rag/warmup.py` | Memuat model ke memori sebelum dipakai |
| `cli.py`, `api.py`, `web/index.html` | Antarmuka CLI, server, dan halaman web |
| `extension/` | Browser extension (Chrome/Edge, MV3) yang mengisi draf lewat `/api/draft` |
| `web/demo_lamaran.html` | Halaman lamaran fiktif 2 langkah (`/demo`) untuk mencoba extension |
| `evaluation/`, `tests/` | Set uji berlabel dan harness evaluasi; tes offline tanpa Ollama |

## Model dan tools

| Komponen | Default | Alasan |
|---|---|---|
| LLM | `qwen3:4b-instruct` via Ollama (2,5 GB) | Gratis dan lokal. Muat di VRAM 4 GB. Varian instruct ini tidak punya mode thinking, jadi lebih cepat dan output JSON-nya stabil |
| Embedding | `qwen3-embedding:0.6b` via Ollama (639 MB) | Skor MTEB Multilingual tertinggi di kelas ~0.6B (mean 64,33, retrieval 64,64; bge-m3: 59,56 / 54,60). Query diberi instruksi tugas sesuai model card |
| Vector store | Matriks numpy (exact search) | Corpus-nya kecil, jadi pencarian exact cukup cepat, tanpa approximation error, dan mudah diinspeksi |
| Keyword | BM25 implementasi sendiri | Nama skill dan tools (SQL, FastAPI) adalah keyword exact |
| PDF | PyMuPDF | Ekstraksi teks per halaman untuk sitasi |

Semua bisa diganti lewat `.env` (lihat [Konfigurasi](#konfigurasi)).

Versi yang dipakai (dicek 21 Okt 2026 dengan `ollama --version`, `ollama list`, `ollama show`):

| Komponen | Versi |
|---|---|
| Ollama | 0.40.6 (run evaluasi pertama `qwen3:4b-instruct` tanggal 18 Okt memakai 0.40.5) |
| `qwen3:4b-instruct` | ID `0edcdef34593`, arsitektur qwen3, 4.0B parameter, kuantisasi Q4_K_M |
| `qwen3-embedding:0.6b` | ID `ac6da0dfba84`, 595.78M parameter, vektor 1024 dimensi, kuantisasi Q8_0 |
| `qwen3.5:4b`, `gemma4:e2b` (hanya untuk perbandingan) | ID `d8b0f5e9760c`, `b37049369adf` |

Model card Qwen3-4B-Instruct-2507 menyebut model ini hanya punya mode non-thinking, walaupun `ollama show`
tetap mencantumkan capability "thinking". Karena itu parameter `think` tidak dikirim untuk model ini.

Kandidat LLM yang dipertimbangkan (riset Oktober 2026). Batasannya: VRAM 4 GB, paham bahasa Indonesia, output
JSON stabil, dan idealnya tanpa mode thinking.

| Model | Ukuran Ollama | Bukti yang ada | Catatan |
|---|---|---|---|
| `qwen3:4b-instruct` (2507) | 2,5 GB | IFEval 83.4, MMLU-ProX 61.6, INCLUDE 60.1 (model card) | Dipilih (dikonfirmasi di Evaluasi) |
| `qwen3.5:4b` | 3,3-4,0 GB | IFEval 89.8, MMLU-ProX 71.5, INCLUDE 71.0 (model card, mode tidak dilabeli) | Thinking aktif secara default. Ada laporan `format` JSON diabaikan saat thinking dimatikan (Ollama issue #14645, varian 35B) |
| `gemma4:e2b` | 4,6-7,5 GB | SEA-HELM Indonesia 64.13, instruction following 90.23 | Satu-satunya model kecil dengan skor bahasa Indonesia terverifikasi, tetapi tidak muat utuh di VRAM 4 GB |
| `llama3.2:3b`, `phi4-mini` | ~2-2,5 GB | - | Bahasa Indonesia tidak ada di daftar bahasa resminya |

Angka di tabel ini diambil dari model card dan leaderboard yang berbeda, jadi tidak bisa dibandingkan langsung.
Perbandingan di kondisi yang sama ada di bagian [Evaluasi](#evaluasi).

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
python cli.py chat                                     # tanya-jawab berulang di terminal
python cli.py autofill --position "Data Analyst Intern" --save
python cli.py draft --questions forms/contoh_pertanyaan.txt --position "Data Analyst Intern"
python cli.py warmup                                   # muat model ke memori sekarang
python cli.py serve                                    # buka http://127.0.0.1:8000
python -m tests.test_offline                           # tes tanpa Ollama
python -m evaluation.run_eval                          # evaluasi (lihat bagian Evaluasi)
```

`--save` menyimpan output lengkap: chunk ter-retrieve, skor, jawaban, sumber, dan setting model. Kalau
`DATA_DIR=data/dummy`, file disimpan ke `outputs/examples/` dan ikut di-commit sebagai contoh. Untuk folder data
lain (mis. `data/private`), file disimpan ke `storage/examples/` yang di-ignore git, supaya isi dokumen pribadi
tidak ikut ter-commit.

`serve` dan `chat` memuat model di awal, dan model disimpan di memori selama `KEEP_ALIVE` (default 30 menit).

Halaman web (`http://127.0.0.1:8000`) punya tiga tab:

- Form simulasi: isi otomatis form contoh di `forms/magang_form.json`. Tombol "Kirim (simulasi)" hanya
  menyimpan draf dan hasil QC ke `storage/qc_log.jsonl`.
- Tempel pertanyaan: salin pertanyaan dari form lamaran asli (satu per baris, maksimal 20), lalu salin draf
  jawabannya kembali ke form itu. Jenis tiap pertanyaan ditebak dengan aturan dari teksnya: isian profil (nama,
  email, HP), jawaban singkat, jawaban paragraf, atau "isi sendiri" untuk kesediaan, gaji, dan jadwal.
- Chat: tanya bebas tentang dokumen. Tiap pertanyaan dijawab sendiri; sistem belum mengingat percakapan.

### Waktu respons

Diukur 21 Okt 2026 di GTX 1650 4 GB dengan `qwen3:4b-instruct` dan cek relevansi aktif.

| Kegiatan | Waktu |
|---|---|
| `ingest` 10 dokumen dummy (termasuk memuat model embedding) | 9,3 detik |
| Memuat `qwen3:4b-instruct` dari kondisi tidak dimuat | 8,9 detik* |
| Satu pertanyaan `ask` (retrieval 3,5 + generation 13,3 + cek relevansi 8,0) | 25,4 detik |
| `chat`, 3 pertanyaan berturut-turut | 44 detik |
| Form simulasi, 13 isian saat diukur (8 lewat RAG) | 155-160 detik (88 detik tanpa cek relevansi) |
| Tempel 9 pertanyaan dari form (6 lewat RAG) | 121 detik |

\* File model kemungkinan masih ada di cache sistem operasi. Saat evaluasi, memuat model butuh 26-101 detik.

Retrieval sendiri butuh 2,2-3,5 detik per pertanyaan. Dugaan penyebabnya: model embedding dan LLM berebut VRAM
(`ollama ps`: 34% LLM berjalan di CPU). Ini belum diselidiki.

## Browser extension

Extension di folder `extension/` membaca label kolom yang masih kosong di halaman lamaran, mengirimnya ke
`/api/draft` di server lokal (jalur yang sama dengan mode tempel pertanyaan), lalu menulis draf ke kolom-kolom itu.
Extension tidak pernah menekan tombol halaman (Next, Submit), tidak menyentuh upload file atau CAPTCHA, dan tidak
menimpa kolom yang sudah diisi pengguna.

Warna garis tepi kolom:

| Warna | Arti |
|---|---|
| Hijau | Terisi |
| Kuning | Perlu dicek |
| Abu-abu | Tidak ditemukan, atau isi sendiri |
| Biru | Kamu memilih mengisi sendiri |
| Abu-abu putus-putus | Kamu memilih mengosongkan |

Arahkan kursor ke kolom untuk melihat sumber dan flags-nya.

Cara kerjanya:

- Kolom wajib (tanda `*`, atau atribut `required` / `aria-required`) langsung diisi.
- Kolom tidak wajib tidak dikirim ke model dulu. Setelah kolom wajib selesai, panel di kanan bawah menanyakan
  tiap kolom: "Isi dari dokumen", "Isi sendiri", atau "Kosongkan". Default-nya "Kosongkan", supaya sistem tidak
  mengarang isian yang tidak diminta. Kalau halaman sama sekali tidak menandai kolom wajib, semua kolom
  ditanyakan lewat panel ini.
- Isi otomatis setelah Next: setelah extension dijalankan sekali di sebuah tab, extension memantau halaman. Begitu
  kamu menekan Next dan muncul kolom kosong baru, extension langsung mengisinya dengan aturan yang sama. Ini hanya
  untuk form yang berganti langkah tanpa reload. Kalau halaman pindah alamat, klik extension sekali lagi. Fitur
  ini bisa dimatikan di popup.
- Pengalaman kerja dan pengalaman organisasi dipisah. Pertanyaan yang jelas menanyakan salah satunya dibatasi ke
  tipe dokumen yang sesuai dan diberi instruksi supaya model tidak mencampur keduanya. Dokumen pengalaman kerja
  selain CV (mis. surat keterangan magang) bisa ditaruh di folder `pengalaman/` atau `kerja/`.

Pasang (Chrome atau Edge):

1. Buka `chrome://extensions` (Edge: `edge://extensions`) dan aktifkan Developer mode.
2. Klik "Load unpacked" lalu pilih folder `extension/`.
3. Jalankan `python cli.py serve`. Popup extension menunjukkan apakah server dan model sudah siap.

Demo dengan halaman lamaran fiktif:

1. Buka `http://127.0.0.1:8000/demo`, isi posisi di popup extension, lalu klik "Isi draf di halaman ini"
   (atau Alt+Shift+D). Popup boleh ditutup; progres terlihat di banner kanan atas.
2. Kolom wajib terisi. Pilih apa yang dilakukan untuk "Program studi / jurusan" (tidak wajib) di panel kanan bawah.
3. Cek isian, lalu tekan "Selanjutnya" sendiri. Langkah 2 terisi otomatis, dan panel menanyakan "Pengalaman kerja /
   magang" dan "Pengalaman organisasi atau kepanitiaan".

Label di halaman demo sengaja ditulis dengan 6 cara berbeda (`<label for>`, label yang membungkus kolom,
`aria-labelledby`, teks di atas kolom, placeholder, `aria-label`). Panel "State form" di bawahnya meniru state
framework seperti React: panel ini hanya berubah lewat event `input`/`change`, jadi terlihat apakah isian dari
extension benar-benar tercatat oleh halaman.

Hasil uji di halaman demo (22 Okt 2026, model asli):

- Langkah 1: keenam gaya label terbaca benar. 5 kolom wajib terisi dalam 27 detik dan tercatat di panel state.
  "Program studi / jurusan" masuk panel dan terisi setelah dipilih "Isi dari dokumen".
- Langkah 2: setelah "Selanjutnya" ditekan, kolom baru terdeteksi sekitar 1 detik kemudian dan 2 kolom wajib
  terisi dalam 50 detik. Dropdown kesediaan dibiarkan untuk diisi sendiri, upload CV tidak disentuh, dan tidak
  ada tombol yang ditekan.
- "Pengalaman kerja / magang" berisi asisten laboratorium dan freelance UMKM, tanpa tercampur organisasi.
- Dua bug ditemukan dan diperbaiki saat uji: tanda `*` hilang dari label berupa teks di atas kolom, dan pemantau
  Next tidak bereaksi pada perubahan atribut `hidden`.
- Versi pertama (tanpa pembedaan wajib/tidak wajib) sudah dicoba sebagai extension terpasang di Chrome pada kedua
  langkah. Di versi sekarang, panel pilihan sudah terlihat berjalan saat terpasang; isi otomatis setelah Next baru
  dites dengan harness yang meniru `background.js`.

Keamanan: server tidak mengirim header CORS, jadi website lain yang terbuka di browser tidak bisa membaca jawaban
dari server lokal. Extension boleh memanggil server karena `host_permissions` hanya untuk `127.0.0.1` dan
`localhost`. Extension memakai izin `activeTab`, bukan izin untuk semua website, karena extension ini memegang
data pribadi.

## Konfigurasi

| Variabel | Default | Keterangan |
|---|---|---|
| `DATA_DIR` | `data/dummy` | Folder dokumen |
| `STORAGE_DIR` | `storage` | Lokasi index dan log |
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
| `KEEP_ALIVE` | `30m` | Lama model disimpan di memori Ollama setelah dipakai; kosong = default Ollama (5m) |
| `WARMUP_ON_START` | `true` | `serve` memuat model di background saat server nyala |

## Contoh output

Delapan contoh dijalankan pada 21 Okt 2026 dengan data dummy, `qwen3:4b-instruct`, retrieval hybrid, k=5, dan
cek relevansi aktif. Output lengkapnya (semua chunk ter-retrieve beserta teks dan skornya, jawaban, sumber,
flags, waktu, dan setting model) ada di [`outputs/examples/`](outputs/examples/).

Satu pertanyaan dari awal sampai akhir (E1, `python cli.py ask "Berapa IPK kumulatif saya?" --short --save`):

1. Retrieval (2,3 detik) mengembalikan 5 chunk: `transkrip#c0` (header transkrip, ada "IPK Kumulatif: 3,62"),
   `cv#c2` (section Pendidikan, ada "IPK 3,62 / 4,00"), lalu `transkrip#c5`, `cv#c3`, `cv#c5`.
2. Generation (8,5 detik): LLM menjawab "3,62" dan menyebut `transkrip#c0` dan `cv#c2` sebagai sumber.
3. Validasi: kedua sumber ada di hasil retrieve, jawaban muncul utuh di teks sumber (support 1,0), dan cek
   relevansi (5,3 detik) menilai jawaban relevan. Status akhirnya `ok`.

| ID | Perintah dan pertanyaan | Chunk relevan yang ter-retrieve | Jawaban | Status | Catatan |
|---|---|---|---|---|---|
| E1 | `ask --short` "Berapa IPK kumulatif saya?" | `transkrip#c0` (rank 1), `cv#c2` (rank 2) | 3,62 | ok | Berhasil |
| E2 | `ask --short` "Sertifikat apa saja yang saya dapat dari Akademi Data Nusantara?" | kedua sertifikat (rank 1 dan 4) | Python untuk Analisis Data, Dasar Machine Learning | ok | Berhasil, jawaban dari dua dokumen |
| E3 | `ask` "Ceritakan project saya yang paling relevan untuk posisi Data Analyst Intern." | hanya 1 chunk project (`prediksi_harga_rumah#c4`, rank 3); rank 1 malah esai | Paragraf tentang project prediksi harga rumah | needs_review | Isi project benar, tetapi ada kalimat tambahan yang tidak ada di sumber (support 0,32), jadi ditandai |
| E4 | `ask` "Project atau pekerjaan apa saja yang pernah saya kerjakan dengan FastAPI?" | chunk sistem inventaris dan sertifikat FastAPI; section Pengalaman di CV (freelance UMKM) tidak masuk top-5 | Hanya sistem inventaris lab | ok | Gagal di retrieval: jawaban tidak lengkap tetapi tidak ditandai |
| E5 | `ask` "Project computer vision apa yang pernah saya buat?" | project lain (tidak relevan) | tidak ditemukan | not_found | Berhasil abstain |
| E6 | `ask --short` "Berapa skor IELTS saya?" | `cv#c7` (section Bahasa, isinya TOEFL) di rank 1 | tidak ditemukan | not_found | Berhasil, tidak tertipu skor TOEFL |
| E7 | `draft` dengan [`forms/contoh_pertanyaan.txt`](forms/contoh_pertanyaan.txt) (6 pertanyaan) | lihat file | Nama, email, IPK benar; "Jurusan" dijawab nama universitas | campuran | "Jurusan" tertangkap cek relevansi; "Ceritakan pengalaman organisasi" ditandai karena dokumen organisasi tidak masuk top-3 tanpa filter tipe dan jawabannya menambah kalimat; pertanyaan kesediaan ditandai "isi sendiri" |
| E8 | `autofill` form simulasi (13 isian saat itu) | lihat file | 11 ok, 2 needs_review | campuran | Isian organisasi lengkap karena form membatasi pencarian ke dokumen organisasi dan CV. Isian project ditandai karena menulis "akurasi cukup tinggi", yang tidak ada di dokumen. Jurusan diisi beserta universitas dan IPK tetapi tidak ditandai |

Waktu: 9-26 detik per pertanyaan `ask` (not_found lebih cepat karena tidak ada cek relevansi), 55 detik untuk
E7, dan 158 detik untuk E8.

Yang terlihat dari contoh ini:

- Pertanyaan faktual dan pertanyaan tanpa bukti ditangani dengan baik (E1, E2, E5, E6).
- Masalah retrieval tidak terdeteksi validasi kalau jawabannya benar tetapi tidak lengkap (E4).
- Validasi menangkap kalimat tambahan yang tidak ada di sumber pada jawaban paragraf (E3, E7, E8), tetapi tidak
  menangkap jawaban singkat yang menambahkan informasi lain (jurusan di E8).
- Filter tipe dokumen di config form membantu retrieval: isian organisasi di E8 lebih baik daripada pertanyaan
  yang sama di E7, yang mencari ke semua dokumen.

## Evaluasi

### Set uji

Set uji ada di [`evaluation/questions.json`](evaluation/questions.json): 26 pertanyaan tentang persona dummy.

| Kategori | Jumlah | Contoh |
|---|---|---|
| `factual` | 12 | "Berapa IPK kumulatif saya?" |
| `cross_doc` (jawaban tersebar di beberapa dokumen) | 3 | "Sertifikat apa saja dari Akademi Data Nusantara?" |
| `open` (jawaban paragraf) | 5 | "Ceritakan project yang paling relevan untuk posisi Data Analyst Intern." |
| `unanswerable` (jawabannya tidak ada, beberapa berupa jebakan) | 6 | "Berapa skor IELTS saya?" (yang ada hanya TOEFL) |

Tiap pertanyaan punya `gold` (dokumen dan potongan teks yang membuktikan jawaban), `must_include` (istilah yang
wajib ada di jawaban), dan `must_not_include` (opsional). Retrieval dievaluasi tanpa filter tipe dokumen.

```bash
python -m evaluation.run_eval --check            # cek label cocok dengan chunk (tanpa Ollama)
python -m evaluation.run_eval --retrieval-only   # bm25 vs vector vs hybrid
python -m evaluation.run_eval                    # retrieval + semua model di evaluation/models.json
python -m evaluation.run_eval --models qwen3:4b-instruct --limit 5   # uji cepat
```

Metrik retrieval: hit@k, cakupan gold, dan MRR. Metrik generation: akurasi pada pertanyaan yang bisa dijawab,
ketepatan abstain (`not_found`), salah abstain, halusinasi yang tertangkap validasi, persentase JSON valid,
ketepatan sitasi, median detik per jawaban, dan porsi model di GPU (dari `ollama ps`). Hasil tersimpan di
`outputs/eval/<waktu>/summary.md`. Model yang dibandingkan diatur di [`evaluation/models.json`](evaluation/models.json);
tiap model perlu di-`ollama pull` dulu, dan `gemma4:e2b` sendiri berukuran 4,6 GB ke atas.

### Retrieval

Diamati 18 Okt 2026, `qwen3-embedding:0.6b`, k=5, 20 pertanyaan yang bisa dijawab.

| Mode | Hit@5 | Cakupan gold | MRR |
|---|---|---|---|
| bm25 | 100% | 95% | 0,883 |
| vector | 100% | 93% | 0,787 |
| hybrid | 100% | 98% | 0,917 |

- f03 "program studi": CV menulis "S1 Teknik Informatika" tanpa kata "program studi". BM25 hanya menemukan
  transkrip, sedangkan vector dan hybrid menemukan CV juga.
- c01 "sertifikat dari Akademi Data Nusantara": vector hanya menemukan 1 dari 2 (rank 5). Nama lembaga adalah
  keyword exact, jadi BM25 dan hybrid lebih baik di sini.
- c02 "pekerjaan dengan FastAPI": vector menemukan keduanya, tetapi di hybrid chunk freelance di CV terdorong
  keluar dari top-5. RRF tidak selalu lebih baik dari komponen terbaiknya.

Angka ini kemungkinan terlalu optimis. Dokumen dummy dan pertanyaannya ditulis oleh orang yang sama, dan
corpus-nya kecil (34 chunk), sehingga top-5 sudah mencakup sekitar 15% corpus.

### Perbandingan LLM

Diamati 19 Okt 2026, satu run, Ollama 0.40.6, retrieval hybrid, k=5, temperature 0, GTX 1650 4 GB. Laporan
lengkapnya di `outputs/eval/20261019-195450/summary.md`.

| Model | Akurasi (20 bisa dijawab) | Abstain benar (6) | Halusinasi tertangkap validasi | JSON valid | Sitasi benar | Median per jawaban | Di GPU |
|---|---|---|---|---|---|---|---|
| `qwen3:4b-instruct` | 90% (18/20) | 83% (5/6) | 1/1 | 100% | 95% | 8,5 detik | 66% |
| `qwen3.5:4b` (thinking off) | 90% (18/20) | 83% (5/6) | 0/1 | 100% | 85% | 9,2 detik | 48% |
| `gemma4:e2b` (thinking off) | 95% (19/20) | 50% (3/6) | 0/3 | 100% | 90% | 6,0 detik | tidak terukur* |

\* `ollama ps` melaporkan hanya 0,32 GB untuk `gemma4:e2b`, jauh di bawah ukuran unduhannya (4,6 GB), jadi angka
GPU-nya tidak dipakai.

Temuan:

- Akurasi pada pertanyaan yang bisa dijawab hampir sama. Selisihnya 1 soal (5 poin persen), dan dengan 20 soal
  itu belum bisa disebut perbedaan nyata.
- Perbedaan terbesar ada di pertanyaan yang jawabannya tidak ada. `gemma4:e2b` mengarang di 3 dari 6, termasuk
  menjawab "TOEFL ITP 540" untuk pertanyaan skor IELTS. Tidak satu pun tertangkap validasi karena isinya memang
  ada di dokumen. Untuk form lamaran, isian yang salah tapi terlihat yakin lebih berbahaya daripada isian kosong.
- `gemma4:e2b` paling sering menyebut `chunk_id` yang tidak ada (f04, o02, o05), sehingga jawabannya ditandai
  `needs_review` walaupun isinya benar.
- Jawaban stabil antar run selama urutan panggilan ke model sama: teks jawaban sama persis (0/26 berubah) untuk
  `qwen3:4b-instruct` di run 18, 19, dan 20 Okt, dan untuk `qwen3.5:4b` serta `gemma4:e2b` di run 19 dan 20 Okt.

Keputusan: `qwen3:4b-instruct` tetap jadi default. Akurasinya sama dengan `qwen3.5:4b` (90%), tetapi sitasinya
paling tepat, porsinya di GPU lebih besar (66% vs 48%), dan lebih cepat. `gemma4:e2b` paling cepat, tetapi
terlalu sering mengarang pada pertanyaan yang jawabannya tidak ada.

Catatan memori: model embedding ikut termuat di VRAM (1,23 GB), sehingga hanya 66% `qwen3:4b-instruct` yang muat
di GPU.

### Analisis kegagalan `qwen3:4b-instruct`

| ID | Yang terjadi | Sumber masalah |
|---|---|---|
| c02 | Hanya menyebut sistem inventaris; website katalog UMKM (freelance) tidak disebut | Retrieval: chunk Pengalaman di CV tidak masuk top-5 hybrid (gagal juga di dua model lain) |
| o03 | Menyebut "2 jam jadi 20 menit" tetapi melewatkan "150 alat", padahal chunk Hasil ikut ter-retrieve | Generation: jawaban tidak lengkap |
| u02 | Menjawab "Saya tidak pernah magang" (menyimpulkan dari ketiadaan bukti), bukan `not_found` | Generation, dan labelnya bisa diperdebatkan. Validasi menandai `needs_review` karena dukungan sumber rendah |
| c03 | Dinilai benar oleh grader, tetapi menyebut "mata kuliah Machine Learning dengan nilai 88/100". Nilai 88 milik sertifikat, nilai mata kuliahnya A | Generation (mencampur fakta). Grader berbasis kata kunci tidak menangkap ini; dua model lain tidak melakukannya |
| f08 | Jawaban benar ("A-") tetapi `chunk_id` yang disebut tidak ada di hasil retrieve (run 18 Okt) | Generation (format sitasi). Validasi bekerja sesuai desain |

Cara membedakannya: lihat dulu chunk yang ter-retrieve. Kalau bukti tidak ada di sana, masalahnya di retrieval;
kalau bukti ada tetapi jawabannya salah, masalahnya di generation.

### Cek relevansi

Semua halusinasi yang lolos validasi di atas adalah fakta yang ada di dokumen tetapi menjawab pertanyaan lain
(ditanya IELTS dijawab TOEFL; ditanya project computer vision dijawab project lain). `support_ratio` tidak bisa
menangkap ini karena jawabannya memang didukung sumber.

Yang dicoba:

1. Cek kata kunci, yaitu menandai pertanyaan yang memuat istilah yang tidak ada di dokumen mana pun. Cara ini
   ditolak karena akan menandai 14 dari 20 pertanyaan yang bisa dijawab: kata umum seperti "kapan", "sudah",
   dan "rencana" juga tidak ada di dokumen.
2. LLM sebagai pemeriksa (`rag/validate.py: check_relevance`). Model yang sama dipanggil sekali lagi dan hanya
   melihat pertanyaan dan jawaban. Kalau jawabannya membahas hal lain, jawaban ditandai `jawaban_tidak_relevan`
   (status `needs_review`). Contoh di prompt pemeriksa sengaja dipilih yang tidak ada di set evaluasi.

Temuan tak terduga: di run pertama (`20261019-213650`), cek relevansi dijalankan langsung setelah tiap jawaban,
dan 6/26 (`qwen3`), 7/26 (`qwen3.5`), dan 2/26 (`gemma4`) jawaban berubah teksnya dibanding run tanpa cek,
walaupun cek itu sendiri tidak menyentuh jawaban. Untuk menguji dugaan bahwa panggilan pemeriksa di antara soal
memengaruhi jawaban berikutnya, evaluasi dijalankan dua tahap (`--relevance-stage separate`, default): semua soal
di-generate dulu, baru semua jawaban diperiksa. Di run `20261020-004417`, teks jawaban ketiga model sama persis
dengan run tanpa cek (0/26 berubah). Penyebabnya terisolasi pada panggilan yang diselipkan, walaupun mekanisme
pastinya (dugaan: cache prompt di Ollama) belum dibuktikan. Autofill di aplikasi memakai urutan dua tahap yang sama.

Hasil cek relevansi dari run dua tahap `20261020-004417`:

| Model | Jawaban salah yang tertangkap | Jawaban benar yang ikut ditandai | Median detik per jawaban (tanpa -> dengan cek) |
|---|---|---|---|
| `qwen3:4b-instruct` | 0/3 | 3/18 (f07, f08, f12) | 8,5 -> 14,4 |
| `qwen3.5:4b` | 2/3 (c01, u02) | 1/18 (f12) | 9,2 -> 15,1 |
| `gemma4:e2b` | 2/4 (u01, u03) | 2/19 (f05, f12) | 6,0 -> 9,3 |

"Jawaban salah" berarti jawaban yang dinilai salah oleh grader, baik untuk soal yang bisa dijawab maupun yang tidak.

- Pemeriksa menangkap kasus yang paling berbahaya: "TOEFL ITP 540" untuk skor IELTS (`gemma4`), dan pengalaman
  asisten lab yang disajikan sebagai magang (`qwen3.5`).
- Ketiga jawaban salah `qwen3` (c02 dan o03 tidak lengkap; u02 "tidak pernah magang") memang menjawab
  pertanyaannya, jadi wajar tidak ditandai. u02 tetap tertangkap oleh cek dukungan sumber.
- Semua jawaban benar yang ikut ditandai adalah jawaban singkat tanpa konteks, misalnya "40" tanpa kata "jam"
  atau "Kelas Koding Indonesia" yang tidak dikenali sebagai nama lembaga.
- Biayanya sekitar 3,5-6 detik tambahan per jawaban.

Keputusan: `RELEVANCE_CHECK=true` jadi default. Untuk form lamaran, isian yang salah tetapi terlihat yakin lebih
berbahaya daripada satu isian tambahan yang perlu dicek. Di PC yang lebih lambat, cek ini bisa dimatikan.
Perbaikan prompt pemeriksa untuk jawaban singkat sebaiknya diuji pada set pertanyaan baru, supaya prompt-nya tidak
disesuaikan dengan soal ujian.

### Temuan dari uji manual

Dari uji mode tempel pertanyaan dan form simulasi dengan model asli:

- "Apakah kamu bersedia magang onsite di Jakarta selama 6 bulan?" sempat dijawab "Ya" tanpa bukti. Cek relevansi
  menandainya, dan sekarang pertanyaan seperti ini dikenali sebagai "isi sendiri" dan tidak dikirim ke LLM.
- "Jurusan" dijawab dengan nama universitas, dan cek relevansi menandainya `jawaban_tidak_relevan`.
- "Asal universitas" dijawab benar tetapi ikut ditandai tidak relevan (pola salah tandai yang sama dengan evaluasi).
- Jawaban singkat kadang menambahkan informasi lain, misalnya jurusan diisi "S1 Teknik Informatika, Universitas
  Nusantara Digital, 3,62". Dugaan penyebab: petunjuk "daftar dipisah koma" di prompt jawaban singkat. Prompt belum
  diubah karena hasil evaluasi di atas terikat pada prompt yang sekarang.

## Keputusan desain

- Chunking mengikuti tipe dokumen. CV dan laporan project punya heading yang jelas, jadi dipotong per section.
  Sertifikat pendek, jadi satu dokumen menjadi satu chunk. Judul dokumen dan nama section ikut di-index.
- Chunk tidak pernah menyeberang halaman, supaya nomor halaman di sitasi tepat.
- Isian tetap (nama, email, HP) diambil dengan regex, bukan LLM: lebih cepat, hasilnya selalu sama, dan tidak
  bisa berhalusinasi.
- Pertanyaan kesediaan, gaji, dan jadwal tidak dikirim ke LLM, karena jawabannya tidak ada di dokumen mana pun.
- Jenis pertanyaan ditebak dengan aturan, bukan LLM, supaya cepat dan bisa ditelusuri.
- LLM wajib menyebut `chunk_id` sumbernya. Sumber yang tidak ada di hasil retrieve ditandai `sumber_tidak_valid`.
- Dukungan sumber dicek secara heuristik: jawaban dibandingkan dengan teks sumber (substring / overlap token), dan
  jawaban di bawah ambang ditandai `needs_review`.
- INFO POSISI dipisah dari KONTEKS supaya deskripsi lowongan tidak dianggap bukti tentang pelamar.
- Cek relevansi dijalankan setelah semua jawaban selesai (dua tahap), supaya tidak mengubah jawaban.

## Keterbatasan

Data dan evaluasi:

- Semua angka berasal dari data dummy yang rapi dan ditulis sendiri. Dokumen asli belum diuji dengan set berlabel.
- Set uji kecil (26 soal), satu run per model, dan grader berbasis kata kunci tidak menangkap fakta yang tercampur
  (c03).
- Ambang dukungan sumber (0,8 / 0,6) belum dievaluasi.
- Tes offline memakai stub. Tes itu hanya membuktikan alurnya berjalan, bukan kualitas retrieval atau jawaban.

Pipeline:

- PDF hasil scan tidak punya teks. File seperti ini dilewati dengan peringatan (perlu OCR atau versi .md).
- CV dua kolom bisa tercampur urutannya saat diekstrak; cek dengan `python cli.py chunks`.
- Deteksi heading berbasis kata kunci dan huruf kapital, jadi heading dengan nama tak lazim tidak terdeteksi.
- Nama diambil dari 10 baris pertama CV dengan heuristik (2-5 kata huruf, tanpa angka atau email). Kalau tidak
  ada yang cocok, isian nama dikosongkan, bukan ditebak.
- Jawaban singkat kadang menambahkan informasi yang tidak ditanya.
- Retrieval butuh 2-3,5 detik, dan 34% LLM masih berjalan di CPU.
- Jenis pertanyaan di mode tempel ditebak dengan pola kata, jadi bisa salah untuk kalimat yang tidak biasa.
- Chat belum punya ingatan percakapan; pertanyaan lanjutan seperti "yang kedua tadi apa?" tidak bisa dijawab.

Extension:

- Label ditebak dengan aturan, jadi bisa meleset di form yang strukturnya tidak biasa.
- Radio button, checkbox, dan form di dalam iframe (mis. Google Form yang di-embed) belum didukung.
- Dropdown hanya diisi kalau teks opsinya cocok dengan jawaban.
- Isi otomatis setelah Next hanya untuk form tanpa reload.
- Penanda wajib dibaca dari tanda `*` dan atribut `required`. Form yang menandai kolom wajib dengan cara lain
  (mis. hanya warna) dianggap tidak menandai kolom wajib, jadi semua kolomnya ditanyakan lewat panel.

## Rencana selanjutnya

Urutan berdasarkan dampak untuk pemakaian nyata:

1. Uji dengan dokumen asli (PDF CV dua kolom, transkrip berbentuk tabel, sertifikat hasil scan) dan buat set
   pertanyaan berlabel kedua dari dokumen itu. Ini cara paling langsung untuk tahu seberapa jauh hasilnya turun.
2. Perbaiki prompt jawaban singkat supaya tidak menambahkan informasi lain, lalu jalankan ulang evaluasi.
3. Bank jawaban dari `storage/qc_log.jsonl`: jawaban yang sudah di-QC dipakai ulang untuk pertanyaan yang mirip di
   form lain. Pertanyaan yang berulang bisa dijawab hampir instan, dan hasilnya sudah dicek manusia.
4. Lanjutan extension: radio button, checkbox, iframe, isi otomatis untuk form yang reload ke alamat baru (butuh
   izin lebih luas, jadi harus opsional), dan token sederhana antara extension dan server. Pengguna tetap menekan
   Next dan Submit sendiri, karena di banyak form menekan Next sudah mengirim isian ke server perusahaan.
5. Jalankan model embedding di CPU supaya LLM mendapat seluruh VRAM, lalu ukur pengaruhnya ke waktu jawaban dan
   kualitas retrieval.
6. Ingatan percakapan di mode chat untuk pertanyaan lanjutan.

Eksperimen yang belum dijalankan:

- Autofill full-context (semua dokumen masuk prompt) vs retrieval per kolom.
- Tingkat edit setelah QC dari `storage/qc_log.jsonl`.
- Instruksi query embedding: dengan vs tanpa `EMBED_QUERY_INSTRUCTION`.
- Klasifikasi pertanyaan di mode tempel: aturan (sekarang) vs LLM, diukur dari ketepatan dan waktu.
- Mekanisme pasti perubahan jawaban saat cek relevansi diselipkan (cache prompt Ollama?).

## Penggunaan AI

Proyek ini saya kerjakan dengan bantuan Claude Code (Anthropic) sebagai asisten coding.

Yang saya kerjakan sendiri:

- Ide dan konsep: masalahnya berasal dari pengalaman saya mengisi form lamaran berulang kali. Saya juga yang
  menentukan batasan sistemnya: di sisi pelamar, berjalan lokal, hanya membuat draf, dan tidak pernah submit.
- Alur kerja dan keputusan desain: alur dari dokumen sampai draf yang saya cek, model harus muat di laptop saya
  (VRAM 4 GB), klasifikasi pertanyaan memakai aturan supaya cepat, kolom wajib langsung diisi sedangkan kolom
  tidak wajib ditanyakan dulu, dan pengalaman kerja dipisah dari pengalaman organisasi.
- Arah evaluasi: membandingkan tiga model pada set uji berlabel, menambahkan cek relevansi untuk menangkap
  halusinasi, dan menguji dugaan penyebab jawaban berubah lewat evaluasi dua tahap.
- QC hasil akhir: mencoba sistem dan extension di Chrome sendiri, mengecek apakah hasilnya sesuai dengan input
  yang saya harapkan, dan memberi masukan supaya nyaman dipakai sehari-hari. Contohnya, extension yang langsung
  mengisi langkah berikutnya setelah Next berasal dari uji coba saya.

Yang dibantu AI:

- Menulis sebagian besar kode (pipeline, evaluasi, halaman web, extension) berdasarkan alur dan keputusan di atas.
- Riset kandidat model yang muat di VRAM 4 GB.
- Menjalankan evaluasi, merangkum hasilnya, dan mencari bug.
- Menyusun draf README dan slide presentasi.
