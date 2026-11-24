"""Pengaturan utama chatbot (path, model, jumlah hasil)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # folder final_project_2
DATA_DIR = ROOT / "data"
EMB_DIR = ROOT / "embeddings"
HF_CACHE = ROOT / "hf_cache"                       # model cross-encoder dari HuggingFace

# Model lewat Ollama
EMBED_MODEL = "nomic-embed-text-v2-moe"            # untuk retriever dense
DEFAULT_LLM = "llama3.2:3b"
QUERY_PREFIX = "search_query: "                    # awalan wajib model nomic
DOC_PREFIX = "search_document: "

# Hybrid (Reciprocal Rank Fusion) & rerank
RRF_K = 60             # konstanta RRF: skor = sum 1 / (RRF_K + peringkat)
HYBRID_DEPTH = 50      # jumlah kandidat dari BM25 dan dari Dense yang digabung
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_DEPTH = 20      # jumlah kandidat hybrid yang diurutkan ulang oleh cross-encoder

# Jumlah hasil yang ditampilkan
K_QUESTIONS = 5        # tombol "Retrieve Pertanyaan"
K_QURAN = 5            # tombol "Retrieve Al-Qur'an & Hadis"
K_HADITH = 5
N_CONTEXT = 3          # jumlah sumber per jenis yang dikirim ke LLM

# Pengaturan LLM
LLM_OPTIONS = {"temperature": 0.1, "num_ctx": 4096, "num_predict": 500}
