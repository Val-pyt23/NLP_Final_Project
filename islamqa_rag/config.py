"""Pengaturan utama chatbot (path, model, jumlah hasil)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # folder final_project_2
DATA_DIR = ROOT / "data"
EMB_DIR = ROOT / "embeddings"

# Model lewat Ollama
EMBED_MODEL = "nomic-embed-text-v2-moe"            # untuk retriever dense
DEFAULT_LLM = "llama3.2:3b"
QUERY_PREFIX = "search_query: "                    # awalan wajib model nomic
DOC_PREFIX = "search_document: "

# Jumlah hasil yang ditampilkan
K_QUESTIONS = 5        # tombol "Retrieve Pertanyaan"
K_QURAN = 5            # tombol "Retrieve Al-Qur'an & Hadis"
K_HADITH = 5
N_CONTEXT = 3          # jumlah sumber per jenis yang dikirim ke LLM

# Pengaturan LLM
LLM_OPTIONS = {"temperature": 0.1, "num_ctx": 4096, "num_predict": 500}
