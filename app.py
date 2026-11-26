import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
SAMPLE_PATH = BASE_DIR / "sample_data" / "sample_annual_report.json"

st.set_page_config(
    page_title="Ledger | Financial Report Analyzer",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --ink: #16231d; --muted: #68766e; --green: #136b4b; --line: #e4e9e5; }
    html, body, [class*="css"] { font-family: 'Aptos', 'Segoe UI', sans-serif; }
    .stApp { background: #f7f8f6; }
    h1, h2, h3 { font-family: 'Aptos Display', 'Segoe UI', sans-serif !important; color: var(--ink); }
    h1 { letter-spacing: -0.045em; }
    .block-container { max-width: 1180px; padding-top: 2rem; }
    .eyebrow { color: var(--green); font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero { border: 1px solid var(--line); background: linear-gradient(120deg, #fff 50%, #edf5ef); border-radius: 20px; padding: 2rem 2.2rem; margin: .5rem 0 1.5rem; }
    .hero p { color: var(--muted); font-size: 1.05rem; max-width: 720px; margin-bottom: 0; }
    .panel { border: 1px solid var(--line); background: #fff; border-radius: 16px; padding: 1.25rem 1.35rem; margin: .65rem 0 1.1rem; }
    .note { color: var(--muted); font-size: .9rem; }
    .stButton > button, .stDownloadButton > button { border-radius: 9px; font-weight: 600; }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.15rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_sample_report():
    with SAMPLE_PATH.open(encoding="utf-8") as sample_file:
        return json.load(sample_file)


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    if reader.is_encrypted:
        raise ValueError("This PDF is password-protected and cannot be read.")
    return [
        {"page": page_number, "text": page.extract_text() or ""}
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def chunk_text(pages, chunk_size=1200):
    chunks = []
    for page in pages:
        text = re.sub(r"\s+", " ", page["text"]).strip()
        for start in range(0, len(text), chunk_size):
            part = text[start : start + chunk_size].strip()
            if part:
                chunks.append({"page": page["page"], "chunk": part})
    return chunks


def simple_summary(text, max_sentences=6):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 30]
    if not sentences:
        return "No readable text was found to summarize."

    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:max_sentences])

    scores = np.asarray(matrix.sum(axis=1)).ravel()
    selected = np.argsort(scores)[-max_sentences:]
    return " ".join(sentences[index] for index in sorted(selected))


def retrieve_relevant_chunks(chunks, query, top_k=4):
    query = query.strip()
    if not chunks or not query:
        return []

    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(
            [chunk["chunk"] for chunk in chunks] + [query]
        )
    except ValueError:
        return []

    similarities = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    ranked = similarities.argsort()[::-1]
    return [
        {
            **chunks[index],
            "score": float(similarities[index]),
        }
        for index in ranked[:top_k]
        if similarities[index] > 0
    ]


def extract_financial_metrics(all_text):
    patterns = {
        "Revenue": r"(?:revenue|net sales)[^\d$]{0,40}\$?\s*([\d,]+(?:\.\d+)?)",
        "Net Income": r"(?:net income|net earnings)[^\d$]{0,40}\$?\s*([\d,]+(?:\.\d+)?)",
        "Operating Income": r"operating income[^\d$]{0,40}\$?\s*([\d,]+(?:\.\d+)?)",
        "Total Assets": r"total assets[^\d$]{0,40}\$?\s*([\d,]+(?:\.\d+)?)",
        "Cash Flow from Operations": r"(?:cash flow from operations|operating cash flow)[^\d$]{0,40}\$?\s*([\d,]+(?:\.\d+)?)",
    }

    rows = []
    for metric, pattern in patterns.items():
        match = re.search(pattern, all_text, flags=re.IGNORECASE)
        rows.append(
            {
                "Metric": metric,
                "Extracted Value (USD millions)": match.group(1) if match else "Not found",
            }
        )
    return pd.DataFrame(rows)


def run_analysis(documents):
    all_chunks = []
    document_stats = []
    readable_text = []

    for document in documents:
        pages = document["pages"]
        text = " ".join(page["text"] for page in pages).strip()
        chunks = chunk_text(pages)
        readable_text.append(text)
        all_chunks.extend({"doc": document["name"], **chunk} for chunk in chunks)
        document_stats.append(
            {
                "Document": document["name"],
                "Pages": len(pages),
                "Text chunks": len(chunks),
                "Characters": len(text),
            }
        )

    combined_text = "\n".join(readable_text)
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(document_stats))
    c2.metric("Pages", sum(stat["Pages"] for stat in document_stats))
    c3.metric("Text chunks", len(all_chunks))

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Document overview")
    st.dataframe(pd.DataFrame(document_stats), width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Executive summary")
    if combined_text.strip():
        st.write(simple_summary(combined_text))
    else:
        st.warning("No selectable text was found. Scanned PDFs need OCR before analysis.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Key financial metrics")
    st.caption("Pattern-based extraction · values shown as reported (USD millions in the sample report)")
    st.dataframe(
        extract_financial_metrics(combined_text),
        width="stretch",
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Ask the report")
    st.caption("Searches the report with TF-IDF and returns the most relevant passages with page citations.")
    with st.form("report_question"):
        question = st.text_input(
            "Your question",
            placeholder="What risks could affect next year's performance?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Find cited passages", type="primary")

    if submitted:
        results = retrieve_relevant_chunks(all_chunks, question, top_k=4)
        if not question.strip():
            st.info("Enter a question to search the report.")
        elif not results:
            st.info("No matching passages found. Try a more specific question using terms from the report.")
        else:
            for index, result in enumerate(results, start=1):
                with st.expander(
                    f"[{index}] {result['doc']} · page {result['page']} · match {result['score']:.0%}",
                    expanded=index == 1,
                ):
                    st.write(result["chunk"])
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Ledger · Financial document intelligence</div>
      <h1>Read the numbers.<br/>Find the story.</h1>
      <p>Turn annual reports into a clear summary, key financial metrics, and cited answers. Start with the sample report or bring your own PDF.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

sample_report = load_sample_report()
source = st.radio(
    "Choose a report",
    ["Explore sample report", "Upload PDF reports"],
    horizontal=True,
    label_visibility="collapsed",
)

if source == "Explore sample report":
    st.caption(
        f"Sample dataset: {sample_report['company']} · {sample_report['report_period']} · "
        "fictional data for demonstration only."
    )
    st.download_button(
        "Download sample dataset",
        data=SAMPLE_PATH.read_bytes(),
        file_name="sample_annual_report.json",
        mime="application/json",
    )
    documents = [
        {
            "name": sample_report["filename"],
            "pages": sample_report["pages"],
        }
    ]
    run_analysis(documents)
else:
    uploaded_files = st.file_uploader(
        "Upload one or more annual reports or 10-K PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs work best. Scanned documents require OCR.",
    )
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            try:
                documents.append(
                    {
                        "name": uploaded_file.name,
                        "pages": extract_text_from_pdf(uploaded_file),
                    }
                )
            except (PdfReadError, ValueError, OSError) as error:
                st.error(f"Could not read {uploaded_file.name}: {error}")
        if documents:
            run_analysis(documents)

st.divider()
st.markdown(
    '<p class="note">Analysis runs in this app session. Results are extractive and informational, not investment advice.</p>',
    unsafe_allow_html=True,
)
