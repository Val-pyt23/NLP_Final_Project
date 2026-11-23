import html

import streamlit as st

from islamqa_rag import METHODS, IslamQAChatbot
from islamqa_rag.config import DEFAULT_LLM, K_HADITH, K_QUESTIONS, K_QURAN
from islamqa_rag.data import load_corpora
from islamqa_rag.evaluation import GroundTruth, metrics
from islamqa_rag.generator import list_llms

st.set_page_config(page_title="AL-MIZAN", page_icon="📖", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri&family=Playfair+Display:ital@0;1&display=swap');
.title {font-family: 'Playfair Display', serif; font-size: 3rem; line-height: 1.15; margin: 1rem 0 .5rem;}
.title em {color: #c9a96e;}
.subtitle {opacity: .6; margin-bottom: 2rem;}
.ref {font-family: monospace; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase;
      display: flex; justify-content: space-between; margin-top: 1.2rem;}
.ref .name {color: #c9a96e;}
.ref .score {opacity: .6; text-transform: none;}
.arabic {direction: rtl; text-align: right; font-family: 'Amiri', serif; font-size: 1.9rem;
         line-height: 3.2rem; margin: .6rem 0;}
.translation {opacity: .85; line-height: 1.6;}
.section {font-family: 'Playfair Display', serif; font-size: 1.4rem; margin-top: 1.5rem;}
</style>""", unsafe_allow_html=True)

BOOKS = {"bukhari": "Sahih al-Bukhari", "tirmidhi": "Jami' at-Tirmidhi", "abudawud": "Sunan Abu Dawud",
         "nasai": "Sunan an-Nasa'i", "ibnmajah": "Sunan Ibn Majah"}
METHOD_LABEL = {"bm25": "BM25 (leksikal)", "dense": "Dense (nomic-embed-text-v2-moe)"}


# ---------------------------------------------------------------- load (sekali saja)
@st.cache_resource(show_spinner="Memuat corpus ...")
def get_corpora():
    return load_corpora()


@st.cache_resource(show_spinner="Menyiapkan retriever (BM25 hadis butuh ±1 menit saat pertama) ...")
def get_bot(method):
    return IslamQAChatbot(method, corpora=get_corpora())


@st.cache_resource(show_spinner="Memuat ground truth ...")
def get_gt():
    return GroundTruth(get_corpora()["islamqa"])


@st.cache_data(ttl=60)
def get_llms():
    try:
        return list_llms()
    except Exception:          # Ollama belum menyala
        return []


# ---------------------------------------------------------------- sidebar: pengaturan model
with st.sidebar:
    st.header("Model")
    method = st.radio("Retriever", METHODS, format_func=METHOD_LABEL.get)
    llms = get_llms()
    if llms:
        llm = st.selectbox("LLM untuk ringkasan (RAG)", llms,
                           index=llms.index(DEFAULT_LLM) if DEFAULT_LLM in llms else 0)
    else:
        st.warning("Ollama tidak terhubung / belum ada model LLM. Ringkasan LLM dinonaktifkan.")
        llm = None
    st.divider()
    st.header("Query uji")
    test_options = get_gt().options(get_corpora()["islamqa"])

    def pick_test_query():
        label = st.session_state.get("test_pick")
        if label:
            st.session_state["query"] = test_options[label]

    st.selectbox("Pilih query yang punya ground truth", list(test_options), index=None, key="test_pick",
                 on_change=pick_test_query, placeholder="Cari soal uji / judul islamqa ...")
    st.caption("Query yang diketik bebas tidak punya ground truth, jadi tidak dievaluasi.")
    st.divider()
    k_q = st.slider("Jumlah pertanyaan islamqa", 1, 10, K_QUESTIONS)
    k_quran = st.slider("Jumlah ayat", 1, 10, K_QURAN)
    k_hadith = st.slider("Jumlah hadis", 1, 10, K_HADITH)
    st.caption("Sumber: islamqa.info (Character & Heart-Softeners), Al-Qur'an (Sahih International), "
               "5 kitab hadis (fawazahmed0/hadith-api).")

try:
    bot = get_bot(method)
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()


# ---------------------------------------------------------------- header + kotak pertanyaan
st.markdown("<div class='title'>Timbangan Dalil &amp; <em>Tanya-Jawab Islami</em></div>"
            "<div class='subtitle'>Penelusuran ayat Al-Qur'an, hadis, dan tanya-jawab islamqa.info "
            "berbasis information retrieval, dengan ringkasan RAG opsional.</div>", unsafe_allow_html=True)


def set_query(text):
    st.session_state["query"] = text
    st.session_state["test_pick"] = None      # pilihan query uji di sidebar dikosongkan


def typed_query():
    st.session_state["test_pick"] = None      # query diketik sendiri -> bukan dari dropdown


col_input, col_clear = st.columns([6, 1], vertical_alignment="bottom")
query = col_input.text_input("Pertanyaan", key="query", label_visibility="collapsed", on_change=typed_query,
                             placeholder="Tulis pertanyaan (bahasa Inggris), lalu Enter ...").strip()
col_clear.button("Hapus", on_click=set_query, args=("",), use_container_width=True)

if not query:
    st.info("Tulis pertanyaan di atas, lalu tekan Enter.")
    st.stop()


# ---------------------------------------------------------------- tampilan hasil
def show_summary(tab_key, results):
    """Ringkasan LLM dibuat hanya saat tombol ditekan, lalu disimpan agar tidak hilang saat ganti tab."""
    if llm is None:
        return
    key = (tab_key, query, method, llm)
    saved = st.session_state.setdefault("answers", {})
    if key not in saved:
        if not st.button(f"Buat ringkasan jawaban dengan {llm}", key=f"btn_{tab_key}"):
            return
        with st.container(border=True):
            saved[key] = st.write_stream(bot.answer(query, results, llm, stream=True))
    else:
        with st.container(border=True):
            st.markdown(saved[key])
    valid, invalid = bot.citations(saved[key], results)
    note = f"Ringkasan RAG · {llm} · sitasi valid: {len(valid)}"
    if invalid:
        note += f" ·tidak ada di sumber: {', '.join(invalid)}"
    st.caption(note)


def plain(text):
    """Teks islamqa ditampilkan apa adanya: ` (misal An-Nazi`at) dan $ tidak dibaca sebagai format markdown."""
    return str(text).replace("`", "‘").replace("$", "\\$")


def mark(doc_id, relevant):
    if relevant is None:
        return ""
    return "🟢 " if doc_id in relevant else "🔴 "


def show_eval(title, ranked_ids, relevant, describe):
    """Kotak evaluasi 1 query: metrik + daftar ground truth yang tidak ditemukan."""
    with st.container(border=True):
        st.markdown(f"**Evaluasi {title}** · ground truth: {len(relevant)} dokumen")
        if not relevant:
            st.caption("Jawaban islamqa kunci tidak mengutip dalil jenis ini, jadi tidak bisa dievaluasi.")
            return
        m = metrics(ranked_ids, relevant)
        k = m["K"]
        cols = st.columns(4)
        cols[0].metric(f"Hit@{k}", "Ya" if m["Hit"] else "Tidak")
        cols[1].metric(f"Precision@{k}", f"{m['Precision']:.2f}")
        cols[2].metric(f"Recall@{k}", f"{m['Recall']:.2f}")
        cols[3].metric("MRR", f"{m['MRR']:.2f}")
        missing = [d for d in relevant if d not in ranked_ids]
        if missing:
            with st.expander(f"Ground truth yang tidak ditemukan ({len(missing)})"):
                for d in missing:
                    st.markdown(f"- {describe(d)}")
        else:
            st.caption("Semua ground truth ditemukan.")


def dalil_card(label, score, arabic, translation):
    st.markdown(f"<div class='ref'><span class='name'>{html.escape(label)}</span>"
                f"<span class='score'>skor {method} {score:.3f}</span></div>"
                f"<div class='arabic'>{html.escape(str(arabic))}</div>"
                f"<div class='translation'>“{html.escape(str(translation))}”</div>", unsafe_allow_html=True)
    st.divider()


dalil = bot.retrieve_dalil(query, k_quran, k_hadith)
questions = bot.retrieve_questions(query, k_q)
n_dalil = len(dalil["quran"]) + len(dalil["hadith"])
gt = get_gt().find(query)
corpora = get_corpora()
gt_quran = set(gt["quran"]) if gt else None
gt_hadith = set(gt["hadith"]) if gt else None
gt_question = set(gt["islamqa"]) if gt else None

if gt:
    key_doc = corpora["islamqa"].loc[gt["doc_id"]]
    st.success(f"Query uji ({gt['source']}) · dokumen kunci: [{gt['doc_id']}] {key_doc['judul']}")
else:
    st.warning("Query ini bukan query uji, jadi tidak ada ground truth dan tidak dievaluasi. "
               "Untuk melihat evaluasi, pilih query di sidebar (Query uji) atau ketik teks yang persis sama "
               "dengan soal uji / judul islamqa.")


def describe_quran(d):
    if d not in corpora["quran"].index:
        return f"QS {d} (tidak ada di corpus)"
    r = corpora["quran"].loc[d]
    return f"QS {d} ({r['surah_name']}): {r['text_en'][:120]}"


def describe_hadith(d):
    status = "terverifikasi" if get_gt().is_verified(gt, d) else "belum terverifikasi"
    text = corpora["hadith"].loc[d, "text_en"] if d in corpora["hadith"].index else "(tidak ada di corpus)"
    return f"{d} ({status}): {str(text)[:120]}"


def describe_question(d):
    return f"[{d}] {corpora['islamqa'].loc[d, 'judul']}"


tab_d, tab_q = st.tabs([f"Retrieve Al-Qur'an & Hadis ({n_dalil} dalil)",
                        f"Retrieve Pertanyaan ({len(questions['islamqa'])} data)"])

with tab_d:
    show_summary("dalil", dalil)
    if gt:
        show_eval("Al-Qur'an", [r["doc_id"] for r in dalil["quran"]], gt["quran"], describe_quran)
        show_eval("Hadis", [r["doc_id"] for r in dalil["hadith"]], gt["hadith"], describe_hadith)
    if not n_dalil:
        st.info("Tidak ada ayat/hadis yang cocok. Coba kata lain atau retriever dense.")
    if dalil["quran"]:
        st.markdown("<div class='section'>Al-Qur'an</div>", unsafe_allow_html=True)
    for r in dalil["quran"]:
        dalil_card(f"{mark(r['doc_id'], gt_quran)}QS. {r['surah_name']} : {r['ayah']}  ({r['doc_id']})", r["score"], r["text_ar"], r["text_en"])
    if dalil["hadith"]:
        st.markdown("<div class='section'>Hadis</div>", unsafe_allow_html=True)
    for r in dalil["hadith"]:
        grade = f" · {r['grade']}" if isinstance(r.get("grade"), str) else ""
        dalil_card(f"{mark(r['doc_id'], gt_hadith)}{BOOKS.get(r['book'], r['book'])} no. {r['number']}{grade}", r["score"],
                   r["text_ar"], r["text_en"])

with tab_q:
    show_summary("pertanyaan", questions)
    if gt:
        show_eval("Pertanyaan", [r["doc_id"] for r in questions["islamqa"]], gt["islamqa"], describe_question)
    if not questions["islamqa"]:
        st.info("Tidak ada pertanyaan yang cocok. Coba kata lain atau retriever dense.")
    for i, r in enumerate(questions["islamqa"], 1):
        with st.expander(f"{mark(r['doc_id'], gt_question)}{i}. {r['judul']}  ·  skor {method} {r['score']:.3f}"):
            st.markdown(f"**Pertanyaan:** {plain(r['pertanyaan'])}")
            st.markdown(f"**Jawaban:**\n\n{plain(r['jawaban'])}")
            st.markdown(f"[Buka di islamqa.info ({r['doc_id']})]({r['url']})")
