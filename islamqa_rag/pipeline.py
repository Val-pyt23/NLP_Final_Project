"""Menggabungkan data + retriever + LLM untuk 2 tombol chatbot.

- Retrieve Pertanyaan      : 5 pertanyaan islamqa yang mirip (+ jawabannya)
- Retrieve Al-Qur'an/Hadis : ayat & hadis yang relevan
Keduanya bisa diberi ringkasan jawaban dari LLM (RAG).
"""
from .config import K_HADITH, K_QUESTIONS, K_QURAN, N_CONTEXT
from .data import load_corpora
from .generator import build_prompt, check_citations, generate
from .retrievers import BM25Retriever, DenseRetriever

METHODS = ["bm25", "dense"]


def shorten(text, n_words):
    words = str(text).split()
    return " ".join(words[:n_words]) + (" ..." if len(words) > n_words else "")


class IslamQAChatbot:
    def __init__(self, method="bm25", corpora=None):
        if method not in METHODS:
            raise ValueError(f"method harus salah satu dari {METHODS}")
        self.method = method
        self.corpora = corpora if corpora is not None else load_corpora()
        self.retrievers = {}
        for name, df in self.corpora.items():
            if method == "bm25":
                self.retrievers[name] = BM25Retriever(df.index, df["search_text"])
            else:
                self.retrievers[name] = DenseRetriever(name, df.index)

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
