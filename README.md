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
│   └── 02_retrieval_rag.ipynb        # eksperimen BM25 + RAG + evaluasi
├── outputs/                 # hasil evaluasi (CSV)
├── embeddings/              # cache embedding dense (tidak di-push, buat dengan build_index.py)
├── islamqa_rag/             # paket chatbot (modular)
│   ├── config.py            # path, nama model, jumlah hasil
│   ├── data.py              # load corpus
│   ├── retrievers.py        # BM25Retriever & DenseRetriever
│   ├── generator.py         # prompt, LLM (Ollama), cek sitasi
│   └── pipeline.py          # IslamQAChatbot: retrieval 2 tombol + RAG
├── build_index.py           # membuat embedding dense
└── app.py                   # UI Streamlit
```

## Cara menjalankan

1. Install paket: `pip install -r requirements.txt`
2. Install [Ollama](https://ollama.com), lalu unduh model:
   ```
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text-v2-moe
   ```
   LLM lain (misal `qwen2.5:3b`, `gemma3:4b`) otomatis muncul di pilihan LLM setelah di-pull.
3. (Hanya untuk retriever dense) buat embedding sekali: `python build_index.py`
4. Jalankan UI dari folder ini: `streamlit run app.py`

## Pemakaian dari Python

```python
from islamqa_rag import IslamQAChatbot

bot = IslamQAChatbot("bm25")                       # atau "dense"
q = "Is it permissible to backbite someone if what I say is true?"

res = bot.retrieve_questions(q)                    # {"islamqa": [...]}
print(bot.answer(q, res, "llama3.2:3b"))

res = bot.retrieve_dalil(q)                        # {"quran": [...], "hadith": [...]}
print(bot.answer(q, res, "llama3.2:3b"))
```

## Sumber data

- islamqa.info (bahasa Inggris), kategori *Character & Heart-Softeners* (2.065 tanya-jawab)
- Al-Qur'an: teks Uthmani & terjemahan Sahih International (api.alquran.cloud)
- Hadis: Bukhari, Tirmidhi, Abu Dawud, Nasa'i, Ibn Majah (github.com/fawazahmed0/hadith-api)

Jawaban chatbot bukan fatwa; selalu rujuk ke ulama yang kompeten.
