"""Evaluasi per query di website: mencocokkan query dengan ground truth (data/qrels_*.csv).

Query dianggap punya ground truth jika teksnya sama dengan:
1. salah satu dari 50 soal uji di qrels_question.csv (asli / parafrase), atau
2. judul pertanyaan islamqa, atau judul + ". " + pertanyaan (query yang dipakai di notebook 02).

Dari situ didapat 1 dokumen islamqa kunci, lalu:
- GT Pertanyaan = dokumen islamqa kunci itu
- GT Al-Qur'an  = ayat yang dikutip di jawabannya (qrels_quran.csv)
- GT Hadis      = hadis yang dikutip di jawabannya (qrels_hadith.csv, ada flag terverifikasi)
"""
import re

import pandas as pd

from .config import DATA_DIR


def _norm(text):
    """huruf kecil, spasi dirapikan, tanda kutip miring (’ “ ” `) disamakan, tanda baca di akhir dibuang."""
    text = str(text).translate(str.maketrans({"’": "'", "‘": "'", "`": "'", "“": '"', "”": '"'}))
    return re.sub(r"\s+", " ", text).strip().rstrip("?.!").strip().lower()


class GroundTruth:
    def __init__(self, islamqa):
        self.test_questions = pd.read_csv(DATA_DIR / "qrels_question.csv", dtype=str)
        quran = pd.read_csv(DATA_DIR / "qrels_quran.csv", dtype=str)
        hadith = pd.read_csv(DATA_DIR / "qrels_hadith.csv", dtype={"query_id": str, "doc_id": str})
        self.quran = quran.groupby("query_id")["doc_id"].apply(list).to_dict()
        self.hadith = hadith.groupby("query_id")["doc_id"].apply(list).to_dict()
        self.hadith_verified = set(hadith.loc[hadith["terverifikasi"], "doc_id"] + "|" +
                                   hadith.loc[hadith["terverifikasi"], "query_id"])

        # teks query -> (doc_id islamqa kunci, keterangan sumber query)
        self.lookup = {}
        for _, r in islamqa.iterrows():
            self.lookup[_norm(r["judul"])] = (r["doc_id"], "judul islamqa")
            self.lookup[_norm(r["search_text"])] = (r["doc_id"], "judul + pertanyaan islamqa")
        for _, r in self.test_questions.iterrows():
            self.lookup[_norm(r["query"])] = (r["doc_id"], f"soal uji {r['query_id']} ({r['jenis']})")

    def find(self, query):
        """-> dict ground truth, atau None jika query bukan query uji."""
        hit = self.lookup.get(_norm(query))
        if hit is None:
            return None
        doc_id, source = hit
        return {"doc_id": doc_id, "source": source,
                "islamqa": [doc_id],
                "quran": self.quran.get(doc_id, []),
                "hadith": self.hadith.get(doc_id, [])}

    def is_verified(self, gt, hadith_id):
        return f"{hadith_id}|{gt['doc_id']}" in self.hadith_verified

    def options(self, islamqa):
        """Pilihan query uji untuk dropdown -> {label: teks query}.
        Berisi 50 soal uji + judul islamqa yang jawabannya mengutip ayat/hadis."""
        opts = {f"{r['query_id']} ({r['jenis']}): {r['query']}": r["query"]
                for _, r in self.test_questions.iterrows()}
        for doc_id, judul in zip(islamqa["doc_id"], islamqa["judul"]):
            n_q, n_h = len(self.quran.get(doc_id, [])), len(self.hadith.get(doc_id, []))
            if n_q or n_h:
                opts[f"islamqa {doc_id}: {judul}  [{n_q} ayat, {n_h} hadis]"] = judul
        return opts


def metrics(ranked_ids, relevant):
    """Metrik untuk 1 query pada K = jumlah hasil yang ditampilkan."""
    relevant = set(relevant)
    k = len(ranked_ids)
    hits = [d for d in ranked_ids if d in relevant]
    first = next((i + 1 for i, d in enumerate(ranked_ids) if d in relevant), None)
    return {"K": k,
            "Hit": float(bool(hits)),
            "Precision": len(hits) / k if k else 0.0,
            "Recall": len(hits) / len(relevant) if relevant else 0.0,
            "MRR": 1 / first if first else 0.0}
