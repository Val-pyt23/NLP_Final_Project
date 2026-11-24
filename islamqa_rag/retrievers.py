"""Retriever dengan antarmuka yang sama: search(query, k) -> [(doc_id, skor), ...]

- TfidfRetriever  : leksikal (bobot TF-IDF, cosine similarity)
- BM25Retriever   : leksikal (kata yang sama persis setelah stemming)
- DenseRetriever  : semantik (embedding nomic lewat Ollama, cosine similarity)
- HybridRetriever : gabungan peringkat beberapa retriever (Reciprocal Rank Fusion)
- RerankRetriever : kandidat dari retriever lain diurutkan ulang oleh cross-encoder
"""
import os
import re
from collections import OrderedDict

import numpy as np
import ollama
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .config import (DOC_PREFIX, EMB_DIR, EMBED_MODEL, HF_CACHE, HYBRID_DEPTH, QUERY_PREFIX, RERANK_DEPTH,
                     RERANK_MODEL, RRF_K)

_stemmer = PorterStemmer()


def preprocess(text):
    """huruf kecil -> kata [a-z0-9]+ -> buang stopword & kata < 2 huruf -> stemming."""
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return [_stemmer.stem(w) for w in words if len(w) >= 2 and w not in ENGLISH_STOP_WORDS]


def _top_k(scores, ids, k):
    """k dokumen dengan skor tertinggi; skor <= 0 (tidak ada kata yang cocok di BM25) dibuang."""
    idx = np.argsort(-scores)[:k]
    return [(ids[i], float(scores[i])) for i in idx if scores[i] > 0]


class _CachedSearch:
    """Menyimpan top-CACHE_DEPTH hasil per query (maks. CACHE_SIZE query terakhir).

    Hybrid dan Rerank memanggil BM25 & Dense dengan query yang sama, begitu juga UI saat
    pindah tab. Dengan cache, skor tidak dihitung ulang. Subclass cukup membuat _search().
    """
    CACHE_DEPTH = 100
    CACHE_SIZE = 5000

    def search(self, query, k=5):
        if k > self.CACHE_DEPTH:
            return self._search(query, k)
        cache = self.__dict__.setdefault("_cache", OrderedDict())
        if query in cache:
            cache.move_to_end(query)
        else:
            cache[query] = self._search(query, self.CACHE_DEPTH)
            if len(cache) > self.CACHE_SIZE:
                cache.popitem(last=False)
        return cache[query][:k]


class TfidfRetriever(_CachedSearch):
    """Preprocessing sama dengan BM25, jadi bedanya hanya cara pembobotan kata.

    sublinear_tf: tf diganti 1 + log(tf), agar kata yang diulang-ulang tidak terlalu dominan.
    Vektor dinormalisasi L2, sehingga dot product = cosine similarity.
    """

    def __init__(self, ids, texts):
        self.ids = list(ids)
        self.vectorizer = TfidfVectorizer(analyzer=preprocess, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)

    def _search(self, query, k):
        q = self.vectorizer.transform([query])
        return _top_k((self.matrix @ q.T).toarray().ravel(), self.ids, k)


class BM25Retriever(_CachedSearch):
    def __init__(self, ids, texts):
        self.ids = list(ids)
        self.bm25 = BM25Okapi([preprocess(t) for t in texts])

    def _search(self, query, k):
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


class DenseRetriever(_CachedSearch):
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

    def _search(self, query, k):
        q = embed([query], QUERY_PREFIX)[0]
        return _top_k(self.vectors @ q, self.ids, k)


# ---------------------------------------------------------------- hybrid & rerank
class HybridRetriever:
    """Reciprocal Rank Fusion: skor dokumen = sum 1 / (RRF_K + peringkat di tiap retriever).

    Yang digabung peringkatnya, bukan skornya, karena skala skor BM25 (0..puluhan) dan
    cosine Dense (-1..1) tidak bisa dibandingkan langsung.
    """

    def __init__(self, retrievers, depth=HYBRID_DEPTH, rrf_k=RRF_K):
        self.retrievers = retrievers
        self.depth = depth
        self.rrf_k = rrf_k

    def search(self, query, k=5):
        scores = {}
        for retriever in self.retrievers:
            for rank, (doc_id, _) in enumerate(retriever.search(query, max(self.depth, k)), start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (self.rrf_k + rank)
        return sorted(scores.items(), key=lambda item: -item[1])[:k]


_cross_encoder = None


def get_cross_encoder():
    """Model cross-encoder dimuat sekali (±90 MB, disimpan di hf_cache/)."""
    global _cross_encoder
    if _cross_encoder is None:
        os.environ.setdefault("HF_HOME", str(HF_CACHE))     # semua cache HuggingFace di folder proyek
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError("Retriever rerank butuh paket sentence-transformers: "
                              "pip install sentence-transformers") from e
        _cross_encoder = CrossEncoder(RERANK_MODEL, max_length=512, cache_folder=str(HF_CACHE))
    return _cross_encoder


class RerankRetriever:
    """Dua tahap: ambil `depth` kandidat dari retriever dasar, lalu cross-encoder membaca
    pasangan (query, dokumen) sekaligus dan memberi skor relevansi baru.

    Skor cross-encoder bisa negatif, jadi tidak memakai _top_k (yang membuang skor <= 0).
    """

    def __init__(self, base, texts, depth=RERANK_DEPTH):
        self.base = base
        self.texts = texts                  # Series: doc_id -> teks dokumen
        self.depth = depth
        self.model = get_cross_encoder()

    def search(self, query, k=5):
        candidates = [doc_id for doc_id, _ in self.base.search(query, max(self.depth, k))]
        if not candidates:
            return []
        scores = self.model.predict([(query, str(self.texts[d])) for d in candidates], show_progress_bar=False)
        order = np.argsort(-scores)[:k]
        return [(candidates[i], float(scores[i])) for i in order]
