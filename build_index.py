"""Membuat embedding dokumen untuk retriever dense (sekali saja, hasil di embeddings/*.npy).

Pakai (Ollama harus menyala):
    python build_index.py               # semua corpus yang belum punya embedding
    python build_index.py hadith        # corpus tertentu (dibuat ulang)
"""
import sys
import time

from islamqa_rag.data import load_corpora
from islamqa_rag.retrievers import build_embeddings, embedding_path

corpora = load_corpora()
names = sys.argv[1:] or [n for n in corpora if not embedding_path(n).exists()]
if not names:
    print("Semua embedding sudah ada di folder embeddings/.")

for name in names:
    start = time.time()
    vectors = build_embeddings(name, corpora[name]["search_text"].tolist())
    print(f"{name}: {vectors.shape} selesai dalam {(time.time() - start) / 60:.1f} menit -> {embedding_path(name).name}")
