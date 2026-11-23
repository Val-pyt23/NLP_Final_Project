"""Dua retriever dengan antarmuka yang sama: search(query, k) -> [(doc_id, skor), ...]

- BM25Retriever  : leksikal (kata yang sama persis setelah stemming)
- DenseRetriever : semantik (embedding nomic lewat Ollama, cosine similarity)
"""
import re

import numpy as np
import ollama
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import DOC_PREFIX, EMB_DIR, EMBED_MODEL, QUERY_PREFIX

_stemmer = PorterStemmer()


def preprocess(text):
    """huruf kecil -> kata [a-z0-9]+ -> buang stopword & kata < 2 huruf -> stemming."""
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return [_stemmer.stem(w) for w in words if len(w) >= 2 and w not in ENGLISH_STOP_WORDS]


def _top_k(scores, ids, k):
    """k dokumen dengan skor tertinggi; skor <= 0 (tidak ada kata yang cocok di BM25) dibuang."""
    idx = np.argsort(-scores)[:k]
    return [(ids[i], float(scores[i])) for i in idx if scores[i] > 0]


class BM25Retriever:
    def __init__(self, ids, texts):
        self.ids = list(ids)
        self.bm25 = BM25Okapi([preprocess(t) for t in texts])

    def search(self, query, k=5):
        return _top_k(self.bm25.get_scores(preprocess(query)), self.ids, k)


# ---------------------------------------------------------------- dense
def embed(texts, prefix, batch_size=32, progress=False):
    """Teks -> vektor ternormalisasi (sehingga cosine similarity = dot product)."""
    batches = range(0, len(texts), batch_size)
    if progress:
        from tqdm.auto import tqdm
        batches = tqdm(batches, desc="embedding")
    vectors = []
    for i in batches:
        batch = [prefix + str(t) for t in texts[i:i + batch_size]]
        vectors += ollama.embed(model=EMBED_MODEL, input=batch, truncate=True)["embeddings"]
    vectors = np.array(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embedding_path(corpus_name):
    return EMB_DIR / f"{corpus_name}_{EMBED_MODEL.replace(':', '-')}.npy"


def build_embeddings(corpus_name, texts):
    """Membuat embedding dokumen sekali, lalu disimpan ke embeddings/*.npy."""
    EMB_DIR.mkdir(exist_ok=True)
    vectors = embed(list(texts), DOC_PREFIX, progress=True)
    np.save(embedding_path(corpus_name), vectors)
    return vectors


class DenseRetriever:
    def __init__(self, corpus_name, ids):
        path = embedding_path(corpus_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding {path.name} belum ada. Jalankan dulu: python build_index.py {corpus_name}")
        self.ids = list(ids)
        self.vectors = np.load(path)
        if len(self.vectors) != len(self.ids):
            raise ValueError(f"{path.name} berisi {len(self.vectors)} vektor, corpus {len(self.ids)} dokumen. "
                             f"Buat ulang: python build_index.py {corpus_name}")

    def search(self, query, k=5):
        q = embed([query], QUERY_PREFIX)[0]
        return _top_k(self.vectors @ q, self.ids, k)
