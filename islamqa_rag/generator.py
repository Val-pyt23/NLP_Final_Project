"""LLM lewat Ollama: prompt, pembuatan jawaban, dan pengecekan sitasi."""
import re

import ollama

from .config import LLM_OPTIONS

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about Islam using ONLY the sources provided. "
    "Do not use outside knowledge and do not issue your own fatwa. "
    "Some sources may be irrelevant: ignore them and never force a connection. "
    "Cite every statement with the exact source ID in square brackets, one ID per bracket, "
    "e.g. [QA-12345], [QS 2:275], [bukhari:6018]. Only cite IDs that appear in the sources. "
    "If the sources do not answer the question, say so clearly. "
    "End with a short reminder to consult a qualified scholar."
)

USER_TEMPLATE = """Question: {question}

Sources:
{sources}

Write a concise answer (3-6 sentences) based only on the relevant sources above, with citations."""


def build_prompt(question, sources):
    """sources: list of (source_id, text), contoh ("QS 2:275", "Those who consume interest ...")."""
    blocks = [f"[{sid}] {text}" for sid, text in sources]
    return USER_TEMPLATE.format(question=question, sources="\n\n".join(blocks))


def list_llms():
    """Model chat yang terpasang di Ollama (model embedding tidak ikut)."""
    names = [m.model for m in ollama.list().models]
    return sorted(n for n in names if "embed" not in n)


def generate(prompt, model, stream=False):
    """stream=False -> string jawaban; stream=True -> generator potongan teks (untuk Streamlit)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    if not stream:
        response = ollama.chat(model=model, messages=messages, options=LLM_OPTIONS)
        return response["message"]["content"].strip()
    chunks = ollama.chat(model=model, messages=messages, options=LLM_OPTIONS, stream=True)
    return (chunk["message"]["content"] for chunk in chunks)


# ---------------------------------------------------------------- sitasi
_BRACKET = re.compile(r"[\[(]([^\])]{2,80})[\])]")
_ID = re.compile(r"QA\s*-\s*\d+|QS\.?\s*\d+\s*:\s*\d+|[A-Za-z][A-Za-z' ]{1,20}:\s*\d+", re.I)


def normalize_id(source_id):
    """'ibn Majah: 3672' -> 'ibnmajah:3672', 'QS 2:275' -> 'qs2:275'."""
    return re.sub(r"[\s.']", "", source_id).lower()


def check_citations(answer, source_ids):
    """Mencari sitasi di jawaban (format [..] maupun (..), boleh beberapa ID dalam satu kurung).

    Mengembalikan (sitasi_valid, sitasi_tidak_valid). Tidak valid = ID yang tidak ada di sumber.
    """
    given = {normalize_id(s) for s in source_ids}
    cited = {normalize_id(m) for inside in _BRACKET.findall(answer) for m in _ID.findall(inside)}
    return sorted(cited & given), sorted(cited - given)
