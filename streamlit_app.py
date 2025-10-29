# OneAfrica Market Pulse — Automated Market Intelligence (Streamlit Demo)
# Author: Richard Gidi
# Run: streamlit run streamlit_app.py

import os
import re
import html
import json
import hashlib
import datetime as dt
from typing import List, Dict, Tuple, Optional, Any
from urllib.parse import urlparse, urljoin
import math
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
import logging
from openai import OpenAI

# ==== .env loader ====
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ==== Logging setup ====
logger = logging.getLogger("oneafrica.pulse")
if not logger.handlers:
    h = logging.StreamHandler()
    f = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    h.setFormatter(f)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ==== Safe imports ====
HAS_SK = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    HAS_SK = False

HAS_VADER = True
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        _ = nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
except Exception:
    HAS_VADER = False

from io import BytesIO
from docx import Document
from docx.shared import Pt, Inches

# ==== App constants ====
APP_NAME = "One Africa Market Pulse"
TAGLINE = "Automated intelligence for cashew, shea, cocoa & allied markets."
QUOTE = "“Ask your data why, until it has nothing else to say.” — Richard Gidi"
FALLBACK_IMG = "https://images.unsplash.com/photo-1519681393784-d120267933ba?q=80&w=1200&auto=format&fit=crop"

DEFAULT_SOURCES = {
    "AllAfrica » Agriculture": "https://allafrica.com/tools/headlines/rdf/agriculture/headlines.rdf",
    "AllAfrica » Business": "https://allafrica.com/tools/headlines/rdf/business/headlines.rdf",
    "CitiNewsroom": "https://citinewsroom.com/feed/",
    "FreshPlaza Africa": "https://www.freshplaza.com/africa/rss.xml",
}

# ==== Basic setup ====
st.set_page_config(page_title=APP_NAME, page_icon="🌍", layout="wide")
st.set_option('client.showErrorDetails', False)

# ==== Session utils ====
def ss_get(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]
def ss_set(key, value):
    st.session_state[key] = value

# ==== Cache + request utils ====
def get_session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
    s.mount("http://", HTTPAdapter(max_retries=r))
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

def _normalize(t: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(t or "")).strip()

@st.cache_data(ttl=600)
def fetch_page(url: str) -> str:
    try:
        h = {"User-Agent":"Mozilla/5.0 OneAfricaPulse/1.0"}
        r = get_session().get(url, headers=h, timeout=10)
        return r.text if r.status_code == 200 else ""
    except Exception:
        return ""

# ==== Simplified feed parsing ====
def parse_feed_xml(content: bytes, base_url: str) -> List[Dict[str,str]]:
    items=[]
    try:
        root=ET.fromstring(content)
        chan=root.find("channel")
        if chan is not None:
            for it in chan.findall("item"):
                t=_normalize(it.findtext("title") or "")
                l=_normalize(it.findtext("link") or "")
                s=_normalize(it.findtext("description") or "")
                d=_normalize(it.findtext("pubDate") or "")
                if l: items.append({"title":t,"link":l,"summary":s,"published_raw":d})
            return items
        for e in root.findall("{http://www.w3.org/2005/Atom}entry"):
            t=_normalize(e.findtext("{http://www.w3.org/2005/Atom}title") or "")
            l=e.find("{http://www.w3.org/2005/Atom}link")
            l=l.attrib.get("href","") if l is not None else ""
            s=_normalize(e.findtext("{http://www.w3.org/2005/Atom}summary") or "")
            d=_normalize(e.findtext("{http://www.w3.org/2005/Atom}updated") or "")
            if l: items.append({"title":t,"link":l,"summary":s,"published_raw":d})
    except Exception: pass
    return items

@st.cache_data(ttl=600)
def fetch_feed_raw(url:str)->bytes:
    try:
        r=get_session().get(url,timeout=10)
        return r.content if r.status_code==200 else b""
    except Exception: return b""

def parse_date_safe(d):
    for f in ("%a, %d %b %Y %H:%M:%S %z","%Y-%m-%d","%d %b %Y"):
        try: return dt.datetime.strptime(d,f)
        except: continue
    return None

# ==== Keyword scoring + summary ====
def keyword_relevance(text,keywords):
    if not text: return 0.0
    if HAS_SK:
        try:
            v=TfidfVectorizer(stop_words="english")
            X=v.fit_transform([text," ".join(keywords)])
            return float(cosine_similarity(X[0:1],X[1:2])[0][0])
        except: pass
    toks=re.findall(r"[a-zA-Z]{3,}",text.lower())
    ks={k.lower() for k in keywords}
    return sum(1 for t in toks if t in ks)/max(1,len(toks))

def simple_summary(text,n=3):
    sents=re.split(r"(?<=[.!?])\s+",text)
    return " ".join(sents[:n])

# ==== Main enrichment ====
def enrich(entry,keywords,min_rel):
    body=_normalize(entry.get("summary",""))
    rel=keyword_relevance(body,keywords)
    if rel<min_rel: return None
    return {
        "source":urlparse(entry.get("link","")).netloc,
        "title":entry.get("title",""),
        "link":entry.get("link",""),
        "published":entry.get("published_raw",""),
        "relevance":rel,
        "summary":simple_summary(body)
    }

# ==== Fetch + process ====
def fetch_from_feed(url,start,end,keywords,min_rel):
    raw=fetch_feed_raw(url)
    its=parse_feed_xml(raw,url)
    out=[]
    for e in its:
        d=parse_date_safe(e.get("published_raw",""))
        if d and (d<start or d>end): continue
        r=enrich(e,keywords,min_rel)
        if r: out.append(r)
    return out

# ================= Sidebar Configurations ====================
with st.sidebar:
    st.header("⚙️ Configurations")

    # --- Default + Custom Sources ---
    st.subheader("📰 RSS/Atom Sources")

    # Initialize persistent custom list
    custom_list = ss_get("custom_sources", [])

    # Default checkboxes
    chosen_sources=[]
    for n,u in DEFAULT_SOURCES.items():
        if st.checkbox(n,True,key=f"src_{n}"): chosen_sources.append(u)

    # --- Custom feed adder ---
    st.markdown("**Add your own feed URL**")
    new_feed = st.text_input("New RSS/Atom URL", placeholder="https://example.com/rss.xml", key="new_feed_input")
    if st.button("➕ Add Feed"):
        if new_feed and new_feed not in custom_list:
            custom_list.append(new_feed)
            st.success(f"Added custom feed: {new_feed}")
        elif new_feed in custom_list:
            st.info("Feed already added.")
        else:
            st.warning("Enter a valid URL first.")
    st.session_state["custom_sources"]=custom_list

    # Display and remove existing custom feeds
    if custom_list:
        st.markdown("**Your Custom Feeds:**")
        for idx, url in enumerate(custom_list):
            col1,col2=st.columns([5,1])
            col1.write(url)
            if col2.button("✖",key=f"del_{idx}"):
                custom_list.pop(idx)
                st.session_state["custom_sources"]=custom_list
                st.rerun()

    st.markdown("---")

    # --- Date Range ---
    st.subheader("📅 Date Range")
    days=st.slider("Days back",1,30,3)
    end=dt.datetime.now()
    start=end-dt.timedelta(days=days)

    # --- Keywords ---
    st.subheader("🔍 Keywords")
    kw_text=st.text_area("Keywords (comma-separated)","cashew,shea,cocoa",height=80)
    keywords=[k.strip() for k in kw_text.split(",") if k.strip()]
    min_rel=st.slider("Min relevance",0.0,1.0,0.05,0.01)

# ================= Main ====================
st.title(APP_NAME)
st.caption(TAGLINE)
st.write(QUOTE)

if st.button("🚀 Scan Now"):
    active_sources = chosen_sources + st.session_state.get("custom_sources", [])
    if not active_sources:
        st.warning("No sources selected or added.")
        st.stop()

    all_rows=[]
    progress=st.progress(0.0)
    for i,u in enumerate(active_sources,1):
        progress.progress(i/len(active_sources))
        st.write(f"Fetching {u}")
        try:
            rows=fetch_from_feed(u,start,end,keywords,min_rel)
            all_rows.extend(rows)
        except Exception as e:
            st.warning(f"Skipped {u}: {e}")
    progress.progress(1.0)

    if not all_rows:
        st.warning("No articles found for your criteria.")
        st.stop()

    df=pd.DataFrame(all_rows)
    st.success(f"Found {len(df)} relevant articles.")
    st.dataframe(df[["source","title","published","relevance"]])

    csv=df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV",csv,"oneafrica_pulse.csv","text/csv")

else:
    st.info("Click **Scan Now** to begin. You can also add your own news feed URLs in the sidebar.")

