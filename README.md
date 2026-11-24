# AL-MIZAN: Chatbot Retrieval islamqa.info + Al-Qur'an + Hadis

Chatbot yang mencari:
- **Pertanyaan serupa** di islamqa.info (beserta jawabannya), dan
- **Ayat Al-Qur'an & hadis** yang relevan,

lalu (opsional) meringkas jawabannya dengan LLM lokal (RAG) lengkap dengan sitasi sumber.

## Struktur folder

```
final_project_2/
├── alquran/, hadist/        # data mentah (JSON) Al-Qur'an & 5 kitab hadis
├── data/                    # corpus & ground truth hasil notebook 01
│   ├── islamqa.csv, quran.csv, hadith.csv
│   └── qrels_question.csv, qrels_quran.csv, qrels_hadith.csv
├── notebooks/
│   ├── 01_corpus_groundtruth.ipynb   # membangun corpus & ground truth
│   ├── 02_retrieval_rag.ipynb        # eksperimen BM25 + RAG + evaluasi (corpus lama)
│   └── 03_evaluasi_metode.ipynb      # evaluasi perbandingan 5 metode retrieval
├── outputs/                 # hasil evaluasi (CSV, grafik)
├── embeddings/              # cache embedding dense (tidak di-push, buat dengan build_index.py)
├── kaggle/                  # notebook untuk membuat embedding dengan GPU Kaggle
├── islamqa_rag/             # paket chatbot (modular)
│   ├── config.py            # path, nama model, jumlah hasil, parameter hybrid/rerank
│   ├── data.py              # load corpus
│   ├── retrievers.py        # TF-IDF, BM25, Dense, Hybrid, Rerank
│   ├── generator.py         # prompt, LLM (Ollama), cek sitasi
│   └── pipeline.py          # IslamQAChatbot: retrieval 2 tombol + RAG
├── build_index.py           # membuat embedding dense
└── app.py                   # UI Streamlit
```

## Metode retrieval

| Metode | Jenis | Cara kerja |
|---|---|---|
| `tfidf` | leksikal | bobot TF-IDF (sublinear tf) + cosine similarity |
| `bm25` | leksikal | BM25Okapi (saturasi tf + normalisasi panjang dokumen) |
| `dense` | semantik | embedding `nomic-embed-text-v2-moe` lewat Ollama + cosine similarity |
| `hybrid` | gabungan | top-50 BM25 + top-50 Dense digabung dengan Reciprocal Rank Fusion (k = 60) |
| `rerank` | dua tahap | top-20 Hybrid diurutkan ulang oleh cross-encoder `ms-marco-MiniLM-L-6-v2` |

TF-IDF dan BM25 memakai preprocessing yang sama (huruf kecil, buang stopword, Porter stemming).
Parameter hybrid & rerank ada di `islamqa_rag/config.py`.

### Hasil evaluasi (MRR, notebook 03)

Soal pertanyaan: 25 asli + 25 parafrase. Ayat & hadis: sampel acak 1.000 query (seed 42).

| Tugas | TF-IDF | BM25 | Dense | Hybrid | Rerank |
|---|---|---|---|---|---|
| pertanyaan (asli) | 0,824 | 0,947 | **1,000** | 0,980 | **1,000** |
| pertanyaan (parafrase) | 0,203 | 0,273 | **0,525** | 0,381 | 0,486 |
| ayat | 0,066 | 0,070 | **0,141** | 0,122 | 0,131 |
| hadis (semua) | 0,055 | 0,055 | 0,069 | **0,073** | 0,070 |
| hadis (terverifikasi) | 0,055 | 0,056 | 0,075 | **0,079** | 0,078 |

Dense terbaik secara keseluruhan; Hybrid dan Rerank belum melampauinya. Metrik lengkap
(Hit/P/R@1,3,5,10, waktu per query) ada di `outputs/eval_retrieval_metode.csv`.

## Cara menjalankan

1. Install paket: `pip install -r requirements.txt`
2. Install [Ollama](https://ollama.com), lalu unduh model:
   ```
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text-v2-moe
   ```
   LLM lain (misal `qwen2.5:3b`, `gemma3:4b`) otomatis muncul di pilihan LLM setelah di-pull.
3. (Untuk retriever dense, hybrid, rerank) buat embedding sekali: `python build_index.py`.
   Di CPU ±4 jam; dengan GPU Kaggle ±10 menit, lihat `kaggle/kaggle_build_embeddings.ipynb`.
   Hasilnya (3 file `.npy`) ditaruh di `embeddings/`.
4. Jalankan UI dari folder ini: `streamlit run app.py`

Retriever `rerank` mengunduh model cross-encoder (±90 MB) ke `hf_cache/` saat pertama dipakai.

## Pemakaian dari Python

```python
from islamqa_rag import IslamQAChatbot

bot = IslamQAChatbot("dense")                      # "tfidf", "bm25", "dense", "hybrid", "rerank"
q = "Is it permissible to backbite someone if what I say is true?"

res = bot.retrieve_questions(q)                    # {"islamqa": [...]}
print(bot.answer(q, res, "llama3.2:3b"))

res = bot.retrieve_dalil(q)                        # {"quran": [...], "hadith": [...]}
print(bot.answer(q, res, "llama3.2:3b"))
```

## Sumber data

- islamqa.info (bahasa Inggris), semua kategori: 15.750 tanya-jawab dari 11 kategori
  (Fiqh & its Principles 9.428, Family Fiqh 1.627, Basic Tenets of Faith 1.506, Character & Heart-Softeners 1.379, dll.),
  beserta subkategori `sub_1`–`sub_5`. File sumbernya `islamqa_semua_kategori_en_final.csv` (±66 MB) tidak ikut di repo;
  lokasinya diatur lewat `ISLAMQA_FILE` di notebook 01.
- Al-Qur'an: teks Uthmani & terjemahan Sahih International (api.alquran.cloud)
- Hadis: Bukhari, Tirmidhi, Abu Dawud, Nasa'i, Ibn Majah (github.com/fawazahmed0/hadith-api)

Jawaban chatbot bukan fatwa; selalu rujuk ke ulama yang kompeten.
