# Ledger — Financial Report Analyzer

A Streamlit app for exploring annual reports and 10-Ks. It provides:

- [Open the interactive sample demo](https://aditi2100.github.io/ai-financial-report-analyzer/)
- An extractive executive summary
- Pattern-based extraction of common financial metrics
- TF-IDF search with cited passages by document and page
- A built-in fictional sample report that loads on the first visit

The sample company, values, and events are entirely fictional. The app does not call an external AI service; answers are retrieved report passages, not generated financial advice.

## Deploy

[![Open interactive GitHub Pages demo](https://img.shields.io/badge/demo-GitHub%20Pages-136b4b?style=for-the-badge)](https://aditi2100.github.io/ai-financial-report-analyzer/)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/aditi2100/ai-financial-report-analyzer)

The Render Blueprint uses the free web-service plan. Create the service from the button above; Render builds the app and provides its public URL. Free services may spin down after inactivity.

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

The sample dataset loads by default. Switch to **Upload PDF reports** to analyze your own files. You can also download the sample JSON directly from the app.

## How it works

1. Select the included sample report or upload one or more text-based PDFs.
2. The app extracts text page by page and creates retrieval chunks.
3. TF-IDF ranks report passages against your question and displays page citations.
4. Pattern-based parsing extracts common financial metrics.

Scanned PDFs need OCR before analysis. Metric extraction is intentionally lightweight and should be checked against the source report.
