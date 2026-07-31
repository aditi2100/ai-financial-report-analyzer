# AI Financial Report Analyzer

A GenAI-style financial document intelligence app that analyzes uploaded PDF reports (e.g., 10-K / annual reports) and provides:

- Executive summary (extractive)
- Key metric extraction (revenue, net income, assets, etc.)
- Retrieval-based Q&A with citation-style context

---

## Why this project matters

This project demonstrates practical AI/ML + data engineering patterns used in document intelligence systems:

- Document ingestion from PDFs
- Text chunking and indexing
- Semantic retrieval (RAG-like workflow)
- Context-grounded Q&A
- Metric extraction from unstructured reports
- Interactive app deployment with Streamlit

---

## Tech stack

- Python
- Streamlit
- PyPDF
- scikit-learn (TF-IDF + cosine similarity)
- Pandas / NumPy

---

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Mac/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

---

## How it works

1. Upload one or more financial PDF reports.
2. App extracts text by page.
3. Text is chunked for retrieval.
4. A TF-IDF retriever finds top relevant chunks for each user question.
5. App displays citation-style sources (document + page).
6. Metrics are extracted via pattern-based parsing.

---

## Suggested production upgrades

- Replace TF-IDF with embedding model + vector DB (FAISS / Pinecone / Weaviate)
- Add OpenAI/LangChain answer synthesis over retrieved chunks
- Add table extraction for more accurate numeric KPI capture
- Add multi-document comparison and trend charts by filing period

---

## Resume bullet

Built a GenAI-powered financial report analyzer using Retrieval-Augmented Generation (RAG) principles to summarize filings, answer natural-language questions with citation-style evidence, and extract key financial metrics from uploaded PDF reports.