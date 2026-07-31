import re
import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="AI Financial Report Analyzer", layout="wide")
st.title("📊 AI Financial Report Analyzer")
st.caption("Upload financial PDFs, get summaries, extract key metrics, and ask questions with citation-style context.")


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
                chunks.append({
                    "page": p["page"],
                    "chunk": part
                })
    return chunks


def simple_summary(text, max_sentences=5):
    # lightweight extractive summary
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    if not sentences:
        return "No sufficient text found to summarize."

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()
    top_idx = np.argsort(scores)[-max_sentences:]
    top_idx = sorted(top_idx)
    summary = " ".join([sentences[i] for i in top_idx])
    return summary


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
            # pick first plausible numeric match
            val = matches[0][1]
            rows.append({"Metric": metric, "Extracted Value (raw)": val})
        else:
            rows.append({"Metric": metric, "Extracted Value (raw)": "Not found"})
    return pd.DataFrame(rows)


# ----------------------------
# UI
# ----------------------------
uploaded = st.file_uploader("Upload one or more financial PDF reports", type=["pdf"], accept_multiple_files=True)

if uploaded:
    all_docs = []
    combined_text = ""
    all_chunks = []

    for file in uploaded:
        pages = extract_text_from_pdf(file)
        chunks = chunk_text(pages, chunk_size=1200)

        doc_text = " ".join([p["text"] for p in pages])
        combined_text += "\n" + doc_text
        all_chunks.extend([{"doc": file.name, **c} for c in chunks])

        all_docs.append({
            "file": file.name,
            "pages": len(pages),
            "chunks": len(chunks),
            "text_len": len(doc_text),
        })

    st.subheader("Uploaded Report Stats")
    st.dataframe(pd.DataFrame(all_docs), use_container_width=True)

    # Summary block
    st.subheader("Executive Summary (Extractive)")
    summary = simple_summary(combined_text, max_sentences=6)
    st.write(summary)

    # Metric extraction block
    st.subheader("Key Financial Metric Extraction")
    metrics_df = extract_financial_metrics(combined_text)
    st.dataframe(metrics_df, use_container_width=True)

    # Q&A block (RAG-like retrieval)
    st.subheader("Ask Questions About the Reports")
    question = st.text_input("Example: What are the biggest risks discussed in the filing?")
    if st.button("Answer with retrieved context"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not all_chunks:
            st.warning("No readable text chunks found.")
        else:
            retrieved = retrieve_relevant_chunks(all_chunks, question, top_k=4)

            st.markdown("### Retrieved Context (with citations)")
            context_text = []
            for i, r in enumerate(retrieved, start=1):
                snippet = r["chunk"][:450].strip().replace("\n", " ")
                st.markdown(f"**[{i}] {r['doc']} — page {r['page']}** (score={r['score']:.3f})")
                st.write(snippet + "...")
                context_text.append(f"[{i}] {snippet}")

            # lightweight answer synthesis (non-LLM fallback)
            st.markdown("### Draft Answer")
            st.write(
                "Based on the most relevant report sections, here is a concise answer:\n\n"
                + " ".join(context_text[:2])
                + "\n\n(For production, replace this synthesis with an LLM call using retrieved chunks.)"
            )

else:
    st.info("Upload PDF reports to begin analysis.")