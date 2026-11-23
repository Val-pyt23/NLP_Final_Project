"""Memuat 3 corpus hasil notebook 01 (data/*.csv)."""
import pandas as pd

from .config import DATA_DIR


def load_corpora():
    """Mengembalikan dict: nama corpus -> DataFrame (index = doc_id).

    Kolom `search_text` adalah teks yang dipakai retriever:
    - islamqa: judul + pertanyaan (jawaban tidak ikut, karena yang dicari pertanyaan yang mirip)
    - quran / hadith: terjemahan Inggris
    """
    islamqa = pd.read_csv(DATA_DIR / "islamqa.csv", dtype={"doc_id": str})
    quran = pd.read_csv(DATA_DIR / "quran.csv")
    hadith = pd.read_csv(DATA_DIR / "hadith.csv")

    islamqa["search_text"] = islamqa["judul"] + ". " + islamqa["pertanyaan"]
    quran["search_text"] = quran["text_en"]
    hadith["search_text"] = hadith["text_en"]

    return {
        "islamqa": islamqa.set_index("doc_id", drop=False),
        "quran": quran.set_index("doc_id", drop=False),
        "hadith": hadith.set_index("doc_id", drop=False),
    }
