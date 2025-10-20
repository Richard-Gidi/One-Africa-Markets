# OneAfrica Market Pulse — Automated Market Intelligence (Streamlit Demo)
# Author: Richard Gidi
# Purpose: Continuous scanning of global news, trade bulletins, and policy updates for tree commodities.
# Run: streamlit run streamlit_app.py

import os
import re
import time
import html
import json
import random
import requests
import datetime as dt
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Optional: simple extractive summarizer (no heavy ML downloads)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_NAME = "OneAfrica Market Pulse"
TAGLINE = "Automated intelligence for cashew, shea, cocoa & allied markets."
QUOTE = "“Ask your data why, until it has nothing else to say.” — Richard Gidi"

DEFAULT_KEYWORDS = [
    "cashew", "shea", "shea nut", "cocoa", "palm kernel", "agri", "export", "harvest",
    "shipment", "freight", "logistics", "port", "tariff", "ban", "fx", "currency",
    "cedi", "naira", "inflation", "subsidy", "cooperative", "value-addition", "processing",
    "ghana", "nigeria", "cote d’ivoire", "ivory coast", "benin", "togo", "burkina",
    "west africa", "sahel", "trade policy", "commodity", "price", "market"
]

DEFAULT_SOURCES = {
    # General commodity / Africa business
    "Reuters Commodities (Africa)": "https://www.reuters.com/markets/commodities/rss",
    "AllAfrica Agriculture": "https://allafrica.com/tools/headlines/rdf/agriculture/headlines.rdf",
    "FAO News": "https://www.fao.org/newsroom/rss/en/",
    "The Conversation Africa": "https://theconversation.com/africa/articles.atom",
    "Agriculture.com News": "https://www.agriculture.com/rss.xml",
    "World Bank Blogs (Africa)": "https://blogs.worldbank.org/africacan/rss",
    # Cocoa / cashew focused portals (general agri proxies)
    "AgFunderNews": "https://agfundernews.com/feed",
    "DailyFX Commodities": "https://www.dailyfx.com/feeds/commodities",
    # Add more trusted feeds as needed
}

IMPACT_RULES = {
    "Supply Risk": [
        r"\bexport ban\b", r"\brestriction\b", r"\bembargo\b", r"\bdrought\b", r"\bflood\b",
        r"\bpest\b", r"\bdisease\b", r"\bshortage\b", r"\bstrike\b", r"\bport closure\b"
    ],
    "Price Upside": [
        r"\bstrong demand\b", r"\bsurge\b", r"\brise\b", r"\bincrease\b", r"\bgrant\b", r"\bstimulus\b",
        r"\bincentive\b", r"\bsubsidy\b", r"\bvalue[- ]addition\b"
    ],
    "Price Downside": [
        r"\boversupply\b", r"\bglut\b", r"\bdecline\b", r"\bfal(?:l|ling)\b", r"\bdrop\b", r"\bcut price\b"
    ],
    "FX Pressure": [
        r"\bdepreciation\b", r"\bweak(?:ening)? currency\b", r"\bfx\b", r"\bforex\b", r"\bdollar strength\b"
    ],
    "Logistics": [
        r"\bfreight\b", r"\bshipping\b", r"\bport\b", r"\bcontainer\b", r"\bcongestion\b", r"\breroute\b"
    ],
}

def _normalize(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_article_text(url: str, timeout: int = 10) -> str:
    """Fetch and lightly clean article text. Falls back to empty on failure."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; OneAfricaPulse/1.0)"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        # Try to get article body
        candidates = []
        for sel in ["article", "[role='main']", ".article-body", ".story-body", ".post-content", "main", "body"]:
            for node in soup.select(sel):
                candidates.append(node.get_text(separator=" ", strip=True))
        text = max(candidates, key=len) if candidates else soup.get_text(separator=" ", strip=True)
        text = _normalize(text)
        return text
    except Exception:
        return ""

def keyword_relevance(text: str, keywords: List[str]) -> float:
    """Compute a simple relevance score based on TF-IDF similarity to keywords string."""
    if not text:
        return 0.0
    corpus = [text]
    query = " ".join(keywords)
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vec.fit_transform(corpus + [query])
        sim = cosine_similarity(X[0:1], X[-1])[0][0]
        return float(sim)
    except Exception:
        # fallback: keyword hits normalized
        tokens = re.findall(r"[a-zA-Z']{3,}", text.lower())
        hits = sum(1 for t in tokens if t in {k.lower(): None for k in keywords})
        return hits / max(1, len(tokens))

def simple_extractive_summary(text: str, n_sentences: int = 3, keywords: Optional[List[str]] = None) -> str:
    """Extract top sentences by TF-IDF weight close to the doc centroid, with keyword boost."""
    if not text:
        return ""
    # Split sentences
    sents = re.split(r"(?<=[\.\?\!])\s+", text)
    # Limit overly long docs
    sents = [s for s in sents if 30 <= len(s) <= 400][:60]
    if len(sents) <= n_sentences:
        return " ".join(sents)

    try:
        vec = TfidfVectorizer(stop_words="english", max_features=8000)
        X = vec.fit_transform(sents)
        centroid = X.mean(axis=0)
        sims = cosine_similarity(X, centroid)
        sims = np.array(sims).ravel()
        # Keyword boost
        if keywords:
            kw = [k.lower() for k in keywords]
            boost = np.array([sum(1 for w in re.findall(r"[a-z']+", s.lower()) if w in kw) for s in sents], dtype=float)
            sims = sims + 0.05 * boost  # small boost
        idx = sims.argsort()[-n_sentences:][::-1]
        chosen = [sents[i] for i in idx]
        return " ".join(chosen)
    except Exception:
        # Fallback: first n sentences
        return " ".join(sents[:n_sentences])

def classify_impact(text: str) -> List[str]:
    tags = []
    lower = text.lower()
    for label, patterns in IMPACT_RULES.items():
        for p in patterns:
            if re.search(p, lower):
                tags.append(label)
                break
    if not tags:
        tags = ["General"]
    return list(dict.fromkeys(tags))  # unique, preserve order

def fetch_from_feed(url: str, days_back: int = 7) -> List[Dict]:
    feed = feedparser.parse(url)
    items = []
    now = dt.datetime.utcnow()
    for e in feed.entries:
        title = _normalize(getattr(e, "title", ""))
        link = getattr(e, "link", "")
        summary = _normalize(getattr(e, "summary", ""))
        # Date parsing with fallbacks
        published = None
        for attr in ["published_parsed", "updated_parsed"]:
            t = getattr(e, attr, None)
            if t:
                published = dt.datetime(*t[:6])
                break
        # Filter by date if we have it
        if published:
            if (now - published).days > days_back:
                continue
            published_str = published.strftime("%Y-%m-%d %H:%M")
        else:
            published_str = "N/A"

        items.append({
            "source": urlparse(url).netloc,
            "title": title,
            "link": link,
            "published": published_str,
            "summary": summary
        })
    return items

def make_digest(df: pd.DataFrame, top_k: int = 12) -> str:
    header = f"# {APP_NAME} — Daily Digest\n\n*{TAGLINE}*\n\n> {QUOTE}\n\n"
    if df.empty:
        return header + "_No relevant items found for the selected period._"
    parts = [header, f"**Date:** {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"]
    for i, row in df.head(top_k).iterrows():
        parts.append(f"### {row['title']}\n"
                     f"- **Source:** {row['source']}  \n"
                     f"- **Published:** {row['published']}  \n"
                     f"- **Relevance:** {row['relevance']:.3f}  \n"
                     f"- **Impact:** {', '.join(row['impact'])}  \n"
                     f"- **Summary:** {row['auto_summary']}\n"
                     f"[Read more]({row['link']})\n\n---\n")
    return "\n".join(parts)

# ---------------- Streamlit UI -----------------

st.set_page_config(page_title=APP_NAME, page_icon="🌍", layout="wide")

st.title("🌍 OneAfrica Market Pulse")
st.caption(TAGLINE)
st.markdown(f"> {QUOTE}")

with st.sidebar:
    st.header("Configuration")
    st.markdown("**Select sources**")
    chosen_sources = []
    for name, url in DEFAULT_SOURCES.items():
        if st.checkbox(name, value=True):
            chosen_sources.append(url)

    st.markdown("---")
    st.subheader("Keywords")
    custom_kw = st.text_area("Add or edit keywords (comma-separated)", value=", ".join(DEFAULT_KEYWORDS), height=120)
    keywords = [k.strip() for k in custom_kw.split(",") if k.strip()] or DEFAULT_KEYWORDS

    days_back = st.slider("Lookback window (days)", min_value=1, max_value=30, value=7, help="Only include items within the last N days (if the feed provides dates).")

    st.markdown("---")
    st.subheader("Summarization")
    n_sent = st.slider("Sentences per summary", 2, 6, 3)

    st.markdown("---")
    st.subheader("Filters")
    min_relevance = st.slider("Minimum relevance score", 0.0, 1.0, 0.10, 0.01)

    st.markdown("---")
    st.subheader("Output")
    top_k = st.slider("Top items in digest", 5, 30, 12)

    run_btn = st.button("🚀 Scan Now")

placeholder = st.empty()

if run_btn:
    with st.spinner("Scanning feeds, extracting content, and generating summaries..."):
        rows = []
        for src in chosen_sources:
            try:
                entries = fetch_from_feed(src, days_back=days_back)
                for e in entries:
                    text = fetch_article_text(e["link"])
                    base = e["summary"] or ""
                    body = text if len(text) > len(base) else base
                    relevance = keyword_relevance(" ".join([e["title"], body]), keywords)
                    if relevance < min_relevance:
                        continue
                    summary = simple_extractive_summary(body, n_sentences=n_sent, keywords=keywords)
                    impact_tags = classify_impact(" ".join([e["title"], body]))
                    rows.append({
                        "source": e["source"],
                        "title": e["title"],
                        "link": e["link"],
                        "published": e["published"],
                        "relevance": relevance,
                        "impact": impact_tags,
                        "auto_summary": summary
                    })
            except Exception as ex:
                st.warning(f"Failed to process {src}: {ex}")

        df = pd.DataFrame(rows).sort_values(by=["relevance"], ascending=False).reset_index(drop=True)
        st.success(f"Found {len(df)} relevant items.")
        st.dataframe(df[["source", "published", "title", "relevance", "impact", "auto_summary", "link"]])

        # Digest
        digest_md = make_digest(df, top_k=top_k)
        st.markdown("---")
        st.subheader("📬 Daily Digest (Markdown)")
        st.code(digest_md, language="markdown")

        # Downloads
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_name = f"oneafrica_pulse_{ts}.csv"
        md_name = f"oneafrica_pulse_digest_{ts}.md"
        df.to_csv(csv_name, index=False)
        with open(md_name, "w", encoding="utf-8") as f:
            f.write(digest_md)

        st.download_button("Download CSV", data=open(csv_name, "rb"), file_name=csv_name, mime="text/csv")
        st.download_button("Download Digest (Markdown)", data=open(md_name, "rb"), file_name=md_name, mime="text/markdown")

        st.markdown("Tip: paste the Markdown into an email, WhatsApp (as a code block), or your company wiki.")

else:
    st.info("Configure your sources and keywords on the left, then click **Scan Now** to generate an automated market digest.")
    st.markdown("""
**What this demo does:**  
- Scans curated RSS feeds for the last *N* days  
- Fetches full article text where possible  
- Scores relevance against your **commodity & policy keywords**  
- Auto-summarizes into 2–6 sentences  
- Tags each item with impact labels (Supply Risk, Price Upside, FX Pressure, Logistics, etc.)  
- Produces a **downloadable CSV** and **Daily Digest (Markdown)**  
    """)
    st.markdown("This is a lightweight, API-free prototype — perfect for a live demo with One Africa Markets.")
