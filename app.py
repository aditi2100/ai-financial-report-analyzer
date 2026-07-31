import re
import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Financial Report Analyzer",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# Minimal UI theme styling
# ----------------------------
st.markdown("""
<style>
/* Primary button */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #0F172A;
    background: #0F172A;
    color: #FFFFFF !important;
    font-weight: 600;
    padding: 0.55rem 1rem;
}

/* Hover */
.stButton > button:hover {
    background: #1E293B;
    border-color: #1E293B;
    color: #FFFFFF !important;
}

/* Focus/active */
.stButton > button:focus,
.stButton > button:active {
    color: #FFFFFF !important;
    box-shadow: 0 0 0 0.2rem rgba(15, 23, 42, 0.18);
    outline: none;
}

/* Disabled */
.stButton > button:disabled {
    background: #CBD5E1 !important;
    border-color: #CBD5E1 !important;
    color: #475569 !important;
    cursor: not-allowed;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Financial Report Analyzer")
st.caption(
    "Upload financial PDFs, generate summaries, extract key metrics, and ask questions with citation-style retrieved context."
)

# ----------------------------
# Built-in demo text
# ----------------------------
DEMO_REPORT_PAGES = [
    {
        "page": 1,
        "text": """
        Company XYZ Annual Report 2025.
        Revenue increased to $12,450 million from $10,980 million in 2024, driven by strong cloud and subscription growth.
        Net income was $2,130 million compared to $1,740 million last year.
        Operating income reached $2,980 million.
        Management highlights continued investment in AI products and international expansion.
        """
    },
    {
        "page": 2,
        "text": """
        Risk Factors.
        The company faces foreign exchange volatility, cybersecurity threats, and supply chain disruptions.
        Macroeconomic uncertainty and changing data privacy regulations may impact future performance.
        Competition in AI-enabled software markets may pressure pricing.
        """
    },
    {
        "page": 3,
        "text": """
        Balance Sheet and Cash Flow.
        Total assets were $25,600 million as of December 31, 2025.
        Cash flow from operations was $3,420 million.
        The company repurchased shares worth $600 million and reduced long-term debt by $300 million.
        """
    },
]


# ----------------------------
# Helpers
# ----------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})
    return pages


def chunk_text(pages, chunk_size=1200):
    chunks = []
    for p in pages:
        txt = p["text"].replace("\n", " ").strip()
        if not txt:
            continue
        for i in range(0, len(txt), chunk_size):
            part = txt[i:i + chunk_size]
            if len(part.strip()) > 50:
                chunks.append({"page": p["page"], "chunk": part})
    return chunks


def simple_summary(text, max_sentences=6):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    if not sentences:
        return "No sufficient text found to summarize."

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()
    top_idx = np.argsort(scores)[-max_sentences:]
    top_idx = sorted(top_idx)
    return " ".join([sentences[i] for i in top_idx])


def retrieve_relevant_chunks(chunks, query, top_k=4):
    corpus = [c["chunk"] for c in chunks]
    vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform(corpus + [query])
    chunk_vecs = X[:-1]
    q_vec = X[-1]
    sims = cosine_similarity(chunk_vecs, q_vec).ravel()
    idx = sims.argsort()[-top_k:][::-1]
    results = []
    for i in idx:
        results.append({
            "page": chunks[i]["page"],
            "chunk": chunks[i]["chunk"],
            "score": float(sims[i]),
        })
    return results


def extract_financial_metrics(all_text):
    patterns = {
        "Revenue": r"(revenue|net sales)[^\d$]{0,20}\$?\s?([\d,]+(\.\d+)?)",
        "Net Income": r"(net income|net earnings)[^\d$]{0,20}\$?\s?([\d,]+(\.\d+)?)",
        "Operating Income": r"(operating income)[^\d$]{0,20}\$?\s?([\d,]+(\.\d+)?)",
        "Total Assets": r"(total assets)[^\d$]{0,20}\$?\s?([\d,]+(\.\d+)?)",
        "Cash Flow": r"(cash flow from operations|operating cash flow)[^\d$]{0,20}\$?\s?([\d,]+(\.\d+)?)",
    }

    rows = []
    txt = all_text.lower()
    for metric, pat in patterns.items():
        matches = re.findall(pat, txt, flags=re.IGNORECASE)
        if matches:
            val = matches[0][1]
            rows.append({"Metric": metric, "Extracted Value (raw)": val})
        else:
            rows.append({"Metric": metric, "Extracted Value (raw)": "Not found"})
    return pd.DataFrame(rows)


def run_analysis(documents):
    all_docs = []
    combined_text = ""
    all_chunks = []

    for doc in documents:
        pages = doc["pages"]
        file_name = doc["name"]

        chunks = chunk_text(pages, chunk_size=1200)
        doc_text = " ".join([p["text"] for p in pages])

        combined_text += "\n" + doc_text
        all_chunks.extend([{"doc": file_name, **c} for c in chunks])

        all_docs.append({
            "file": file_name,
            "pages": len(pages),
            "chunks": len(chunks),
            "text_len": len(doc_text),
        })

    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(all_docs))
    c2.metric("Total Chunks", int(sum(d["chunks"] for d in all_docs)))
    c3.metric("Total Characters", f"{int(sum(d['text_len'] for d in all_docs)):,}")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Document Stats")
    st.dataframe(pd.DataFrame(all_docs), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Executive Summary")
    summary = simple_summary(combined_text, max_sentences=6)
    st.write(summary)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Key Financial Metric Extraction")
    metrics_df = extract_financial_metrics(combined_text)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Q&A (Retrieval-Based)")
    question = st.text_input("Ask a question about the uploaded reports", placeholder="What risks are highlighted in the report?")
    ask = st.button("Answer with Retrieved Context", use_container_width=False)

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")
        elif not all_chunks:
            st.warning("No readable text chunks found.")
        else:
            retrieved = retrieve_relevant_chunks(all_chunks, question, top_k=4)

            st.markdown("#### Retrieved Context (Citations)")
            context_text = []
            for i, r in enumerate(retrieved, start=1):
                snippet = r["chunk"][:420].strip().replace("\n", " ")
                st.markdown(f"**[{i}] {r['doc']} — page {r['page']}** · relevance `{r['score']:.3f}`")
                st.write(snippet + "...")
                context_text.append(f"[{i}] {snippet}")

            st.markdown("#### Draft Answer")
            st.write(
                "Based on the most relevant sections:\n\n"
                + " ".join(context_text[:2])
                + "\n\n_Production upgrade: plug in an LLM over these retrieved chunks for synthesized answers._"
            )
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------
# Main UI flow
# ----------------------------
st.markdown("### Quick Start")
col_a, col_b = st.columns([1, 2])
with col_a:
    use_demo = st.button("Load Demo Financial Report", use_container_width=True)
with col_b:
    st.caption("No PDF needed for this mode. Useful for instant demo during interviews.")

uploaded = st.file_uploader(
    "Upload one or more financial PDF reports",
    type=["pdf"],
    accept_multiple_files=True
)

st.markdown("<hr/>", unsafe_allow_html=True)

if use_demo:
    demo_documents = [{"name": "demo_financial_report_2025.pdf", "pages": DEMO_REPORT_PAGES}]
    st.success("Loaded built-in demo report.")
    run_analysis(demo_documents)
elif uploaded:
    docs = []
    for file in uploaded:
        pages = extract_text_from_pdf(file)
        docs.append({"name": file.name, "pages": pages})
    run_analysis(docs)
else:
    st.info("Click **Load Demo Financial Report** or upload PDFs to begin.")
