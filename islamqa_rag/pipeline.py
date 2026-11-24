"""Menggabungkan data + retriever + LLM untuk 2 tombol chatbot.

- Retrieve Pertanyaan      : 5 pertanyaan islamqa yang mirip (+ jawabannya)
- Retrieve Al-Qur'an/Hadis : ayat & hadis yang relevan
Keduanya bisa diberi ringkasan jawaban dari LLM (RAG).
"""
from .config import K_HADITH, K_QUESTIONS, K_QURAN, N_CONTEXT
from .data import load_corpora
from .generator import build_prompt, check_citations, generate
from .retrievers import (BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever,
                         TfidfRetriever)

METHODS = ["tfidf", "bm25", "dense", "hybrid", "rerank"]

_built = {}     # (metode, corpus, jumlah dokumen) -> retriever, agar hybrid/rerank memakai ulang BM25 & Dense


def build_retriever(method, corpus_name, df):
    """Membuat (atau memakai ulang) retriever `method` untuk satu corpus (df dari load_corpora)."""
    key = (method, corpus_name, len(df))
    if key not in _built:
        _built[key] = _make_retriever(method, corpus_name, df)
    return _built[key]


def _make_retriever(method, corpus_name, df):
    if method == "tfidf":
        return TfidfRetriever(df.index, df["search_text"])
    if method == "bm25":
        return BM25Retriever(df.index, df["search_text"])
    if method == "dense":
        return DenseRetriever(corpus_name, df.index)
    if method == "hybrid":
        return HybridRetriever([build_retriever("bm25", corpus_name, df),
                                build_retriever("dense", corpus_name, df)])
    if method == "rerank":
        return RerankRetriever(build_retriever("hybrid", corpus_name, df), df["search_text"])
    raise ValueError(f"method harus salah satu dari {METHODS}")


def shorten(text, n_words):
    words = str(text).split()
    return " ".join(words[:n_words]) + (" ..." if len(words) > n_words else "")


class IslamQAChatbot:
    def __init__(self, method="bm25", corpora=None):
        if method not in METHODS:
            raise ValueError(f"method harus salah satu dari {METHODS}")
        self.method = method
        self.corpora = corpora if corpora is not None else load_corpora()
        self.retrievers = {name: build_retriever(method, name, df) for name, df in self.corpora.items()}

    # ------------------------------------------------------------ retrieval
    def search(self, corpus, query, k):
        """-> list of dict: baris corpus + kolom 'score'."""
        df = self.corpora[corpus]
        return [{**df.loc[doc_id].to_dict(), "score": score}
                for doc_id, score in self.retrievers[corpus].search(query, k)]

    def retrieve_questions(self, query, k=K_QUESTIONS):
        return {"islamqa": self.search("islamqa", query, k)}

    def retrieve_dalil(self, query, k_quran=K_QURAN, k_hadith=K_HADITH):
        return {"quran": self.search("quran", query, k_quran),
                "hadith": self.search("hadith", query, k_hadith)}

    # ------------------------------------------------------------ RAG
    @staticmethod
    def sources_for_llm(results, n=N_CONTEXT):
        """Mengubah hasil retrieval menjadi [(source_id, teks)] untuk prompt."""
        sources = []
        for r in results.get("islamqa", [])[:n]:
            sources.append((f"QA-{r['doc_id']}", f"Question: {r['judul']}\nAnswer: {shorten(r['jawaban'], 150)}"))
        for r in results.get("quran", [])[:n]:
            sources.append((f"QS {r['doc_id']}", r["text_en"]))
        for r in results.get("hadith", [])[:n]:
            sources.append((r["doc_id"], shorten(r["text_en"], 100)))
        return sources

    def answer(self, query, results, model, stream=False):
        prompt = build_prompt(query, self.sources_for_llm(results))
        return generate(prompt, model, stream=stream)

    def citations(self, answer, results):
        ids = [sid for sid, _ in self.sources_for_llm(results)]
        return check_citations(answer, ids)
