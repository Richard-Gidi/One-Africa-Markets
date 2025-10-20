# OneAfrica Market Pulse — Automated Market Intelligence (Streamlit Demo)

A lightweight, executive-ready demo that **scans news sources**, **summarizes insights**, and **ranks relevance** for **tree commodities** (cashew, shea, cocoa, palm kernel) and policy updates across Africa.

> “Ask your data why, until it has nothing else to say.” — Richard Gidi

---

## Features
- Curated RSS feeds (Reuters commodities, AllAfrica Agriculture, FAO, etc.)
- Keyword-based **relevance scoring** using TF‑IDF (no API keys required)
- **Extractive summaries** (2–6 sentences) with a simple TF‑IDF centroid method
- **Impact tags** via rule-based classifier (Supply Risk, Price Upside, FX Pressure, Logistics, General)
- Export **CSV** and **Markdown Digest**
- Fully configurable in the sidebar (feeds, keywords, filters)

## Quickstart

```bash
# 1) Create a virtual env (recommended)
python -m venv .venv
source .venv/bin/activate         # on Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run Streamlit
streamlit run streamlit_app.py
```

## Customization

- **Add/Remove Feeds:** Edit `DEFAULT_SOURCES` inside `streamlit_app.py`.
- **Keywords:** Use the sidebar to modify keywords (comma-separated) during runtime.
- **Impact Rules:** Adjust `IMPACT_RULES` in `streamlit_app.py` to tune the tags.
- **Lookback Window:** Slider in the sidebar controls how far back to fetch from each feed.

## Notes

- This demo intentionally avoids heavyweight models and paid APIs, so it will run anywhere.
- For production, you can:
  - Replace the summarizer with an LLM (OpenAI, Azure, etc.).
  - Add **n8n** to schedule daily runs and push digests to email/WhatsApp/Slack.
  - Plug outputs into **Power BI** or a **FastAPI** backend for multi-user access.
  - Store articles and embeddings in a vector DB (FAISS/Chroma) for semantic search.
  - Add entity extraction and country/commodity tagging for deeper analytics.

## Branding

The app is titled **OneAfrica Market Pulse** and is ready to demo to One Africa Markets.
You can tweak the header or add your logo by editing the Streamlit layout at the top of the file.

---

**Author:** Richard Gidi  
Portfolio: https://www.datascienceportfol.io/richkgidi
