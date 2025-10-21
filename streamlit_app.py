# OneAfrica Market Pulse — Automated Market Intelligence (Streamlit Demo)
# Author: Richard Gidi
# Focus: Robust news-only pipeline (RSS/Atom + optional Newsdata.io) with premium UI and friendly error handling.
# Run: streamlit run streamlit_app.py

import re
import os
import html
import json
import time
import hashlib
import datetime as dt
from typing import List, Dict, Tuple, Optional, Any
from functools import lru_cache
from urllib.parse import urlparse, urljoin

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

# ========================= Streamlit safety: hide tracebacks =========================
st.set_option('client.showErrorDetails', False)

# ========================= Logging (server console only) =========================
logger = logging.getLogger("oneafrica.pulse")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ========================= Optional sklearn (graceful fallback) =========================
HAS_SK = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    HAS_SK = False
    logger.info("sklearn not available; falling back to keyword hit scoring.")

# ========================= App Strings / Theme =========================
APP_NAME = "OneAfrica Market Pulse"
TAGLINE = "Automated intelligence for cashew, shea, cocoa & allied markets."
QUOTE = "“Ask your data why, until it has nothing else to say.” — Richard Gidi"

# Fallback image for articles with no thumbnail
FALLBACK_IMG = "https://images.unsplash.com/photo-1519681393784-d120267933ba?q=80&w=1200&auto=format&fit=crop"

DEFAULT_KEYWORDS = [
    "cashew", "shea", "shea nut", "cocoa", "palm kernel", "agri", "export", "harvest",
    "shipment", "freight", "logistics", "port", "tariff", "ban", "fx", "currency",
    "cedi", "naira", "inflation", "subsidy", "cooperative", "value-addition", "processing",
    "ghana", "nigeria", "cote d’ivoire", "ivory coast", "benin", "togo", "burkina",
    "west africa", "sahel", "trade policy", "commodity", "price", "market"
]

# Curated working RSS/Atom sources
DEFAULT_SOURCES = {
    "AllAfrica » Agriculture": "https://allafrica.com/tools/headlines/rdf/agriculture/headlines.rdf",
    "AllAfrica » Business": "https://allafrica.com/tools/headlines/rdf/business/headlines.rdf",
    "The Standard » Business": "https://www.standardmedia.co.ke/rss/business.php",
    "The Standard » Agriculture": "https://www.standardmedia.co.ke/rss/agriculture.php",
    "CitiNewsroom": "https://citinewsroom.com/feed/",
    "FAO News (All topics)": "https://www.fao.org/news/rss/en",
    "FAO GIEWS": "https://www.fao.org/giews/rss/en/",
    "FreshPlaza Africa": "https://www.freshplaza.com/africa/rss.xml",
    "African Arguments": "https://africanarguments.org/feed/",
    "How We Made It In Africa": "https://www.howwemadeitinafrica.com/feed/",
    "Bizcommunity (Africa • Agri+Logistics)": "https://www.bizcommunity.com/GenerateRss.aspx?i=63,76&c=81",
}

IMPACT_RULES = {
    "Supply Risk": [
        r"\bexport (?:ban|restriction|control)\b",
        r"\b(?:import|trade) (?:ban|restriction|control)\b",
        r"\bembargo\b",
        r"\b(?:drought|flood|rainfall|weather)\b",
        r"\b(?:pest|disease|infestation)\b",
        r"\b(?:shortage|scarcity)\b",
        r"\b(?:strike|protest|unrest)\b",
        r"\bport (?:closure|congestion|delay)\b",
        r"\bharvest (?:delay|loss|damage)\b",
        r"\bproduction (?:issue|problem|concern)\b",
    ],
    "Price Upside": [
        r"\bstrong (?:demand|buying|interest)\b",
        r"\b(?:surge|spike|jump|rise|increase)\b",
        r"\b(?:grant|stimulus|support)\b",
        r"\b(?:incentive|subsidy|funding)\b",
        r"\bvalue[- ](?:addition|chain|processing)\b",
        r"\bhigh(?:er)? (?:price|demand|consumption)\b",
        r"\bmarket (?:rally|strength|upturn)\b",
        r"\bsupply (?:squeeze|shortage|tightness)\b",
        r"\bquality premium\b",
    ],
    "Price Downside": [
        r"\b(?:oversupply|surplus|glut)\b",
        r"\b(?:decline|fall|drop|decrease|slump)\b",
        r"\b(?:weak|soft|bearish) (?:price|market|demand)\b",
        r"\bcut (?:price|rate|cost)\b",
        r"\blow(?:er)? (?:price|demand|consumption)\b",
        r"\bmarket (?:weakness|downturn)\b",
        r"\bcompetitive pressure\b",
    ],
    "FX & Policy": [
        r"\b(?:depreciation|devaluation)\b",
        r"\bweak(?:ening)? (?:currency|exchange)\b",
        r"\b(?:fx|forex|dollar|euro|yuan)\b",
        r"\b(?:monetary|fiscal|trade) policy\b",
        r"\b(?:interest|exchange) rate\b",
        r"\b(?:tariff|duty|levy|tax)\b",
        r"\bregulatory (?:change|update|requirement)\b",
        r"\bpolicy (?:change|update|reform)\b",
    ],
    "Logistics & Trade": [
        r"\b(?:freight|shipping|transport)\b",
        r"\b(?:port|container|vessel|cargo)\b",
        r"\b(?:congestion|delay|bottleneck)\b",
        r"\b(?:reroute|divert|alternative route)\b",
        r"\b(?:cost|rate) (?:increase|surge|rise)\b",
        r"\btrade (?:flow|route|pattern)\b",
        r"\b(?:export|import) (?:volume|data|figure)\b",
    ],
    "Market Structure": [
        r"\b(?:merger|acquisition|takeover)\b",
        r"\b(?:investment|expansion|capacity)\b",
        r"\b(?:processing|factory|facility)\b",
        r"\b(?:certification|standard|quality)\b",
        r"\b(?:cooperative|association|group)\b",
        r"\b(?:contract|agreement|deal)\b",
        r"\b(?:partnership|collaboration)\b",
        r"\bmarket (?:structure|reform|development)\b",
    ],
    "Tech & Innovation": [
        r"\b(?:technology|innovation|digital)\b",
        r"\b(?:blockchain|traceability|tracking)\b",
        r"\b(?:sustainability|sustainable)\b",
        r"\b(?:efficiency|optimization)\b",
        r"\b(?:automation|mechanization)\b",
        r"\b(?:research|development|r&d)\b",
        r"\b(?:startup|fintech|agtech)\b",
    ],
}

# ========================= Streamlit UI CSS =========================
CARD_CSS = """
<style>
/* Hero header */
.hero {
  position: relative;
  border-radius: 16px;
  padding: 28px 28px;
  background: linear-gradient(135deg, #0ea5e9, #7c3aed 60%);
  color: white;
  box-shadow: 0 14px 40px rgba(0,0,0,0.18);
}
.hero h1 { margin: 0 0 6px 0; font-size: 28px; font-weight: 800; }
.hero p { margin: 0; opacity: .95; }

/* Section title pill */
.pill {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 12px; border-radius: 999px;
  background: rgba(255,255,255,0.15); color:#fff; font-weight:600; font-size: 13px;
}

/* Article cards */
.grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 16px;
}
.card {
  background: var(--card-bg, #ffffff);
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 14px; overflow: hidden;
  transition: transform .15s ease, box-shadow .15s ease;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(0,0,0,0.08); }
.thumb { width: 100%; height: 180px; object-fit: cover; background:#f6f7f9; }
.card-body { padding: 14px; }
.title { font-weight: 700; font-size: 16px; margin: 4px 0 8px 0; line-height: 1.25; }
.meta { font-size: 12px; color: #6b7280; display:flex; gap:10px; flex-wrap:wrap; margin-bottom:8px; }
.badges { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }
.badge {
  font-size: 11px; font-weight:700; padding:4px 8px; border-radius:999px;
  background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe;
}
.summary { font-size: 13px; color:#374151; line-height:1.5; margin-top: 6px;}
.link { text-decoration: none; font-weight:700; color:#2563eb; }
</style>
"""

# ========================= HTTP utils + safe wrappers =========================
def get_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

def _normalize(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Global bag to collect soft errors (shown as a friendly banner later)
SOFT_ERRORS: List[str] = []

def soft_fail(msg: str, detail: Optional[str] = None):
    """Collect friendly error notes; log details to console only."""
    if msg:
        SOFT_ERRORS.append(msg)
    if detail:
        logger.warning(detail)

# ========================= Content helpers =========================
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_page(url: str, timeout: int = 12) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; OneAfricaPulse/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        r = get_session().get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            soft_fail("Skipped a page that didn’t load cleanly.", f"fetch_page {url} -> {r.status_code}")
            return ""
        return r.text
    except Exception as e:
        soft_fail("Skipped one page due to connectivity.", f"fetch_page EXC {url}: {e}")
        return ""

def get_og_image(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    try:
        candidates = [
            ("meta", {"property": "og:image"}),
            ("meta", {"name": "twitter:image"}),
            ("meta", {"property": "twitter:image"}),
            ("link", {"rel": "image_src"}),
        ]
        for tag, attrs in candidates:
            el = soup.find(tag, attrs=attrs)
            if el:
                src = el.get("content") or el.get("href")
                if src:
                    if src.startswith("//"):
                        return "https:" + src
                    if src.startswith("/"):
                        return urljoin(base_url, src)
                    return src
    except Exception as e:
        logger.info(f"get_og_image: {e}")
    return None

def get_favicon_url(domain_url: str) -> str:
    parsed = urlparse(domain_url)
    return f"{parsed.scheme}://{parsed.netloc}/favicon.ico"

@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_article_text_and_image(url: str) -> Tuple[str, str]:
    """Returns (text, image_url). Never raises to UI."""
    if not url:
        return "", FALLBACK_IMG
    html_text = fetch_page(url)
    if not html_text:
        return "", FALLBACK_IMG
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "iframe", "form"]):
            tag.decompose()

        candidates = []
        selectors = [
            "article", "[role='main']", ".article-body", ".story-body",
            ".post-content", "main", ".content", ".entry-content",
            "#article-body", ".article-content", ".story-content",
            ".news-content", ".page-content", "body",
        ]
        for sel in selectors:
            for node in soup.select(sel):
                text = node.get_text(separator=" ", strip=True)
                if len(text) > 100:
                    candidates.append(text)
        text = max(candidates, key=len) if candidates else soup.get_text(separator=" ", strip=True)
        text = _normalize(text)
        if len(text) < 50:
            text = ""

        img = get_og_image(soup, url) or get_favicon_url(url) or FALLBACK_IMG
        return text, img
    except Exception as e:
        soft_fail("Used a fallback image for one article.", f"fetch_article_text_and_image EXC {url}: {e}")
        return "", FALLBACK_IMG

# ========================= Relevance & Summary =========================
def keyword_relevance(text: str, keywords: List[str]) -> float:
    if not text:
        return 0.0
    if HAS_SK:
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=5000)
            X = vec.fit_transform([text, " ".join(keywords)])
            sim = cosine_similarity(X[0:1], X[1:2])[0][0]
            return float(sim)
        except Exception as e:
            logger.info(f"tfidf relevance fallback: {e}")
    tokens = re.findall(r"[a-zA-Z']{3,}", text.lower())
    kwset = {k.lower() for k in keywords}
    hits = sum(1 for t in tokens if t in kwset)
    return hits / max(1, len(tokens))

def simple_extractive_summary(text: str, n_sentences: int = 3, keywords: Optional[List[str]] = None) -> str:
    if not text:
        return ""
    sents = re.split(r"(?<=[\.\?\!])\s+", text)
    sents = [s for s in sents if 30 <= len(s) <= 400][:60]
    if len(sents) <= n_sentences:
        return " ".join(sents)
    if HAS_SK:
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=8000)
            X = vec.fit_transform(sents)
            centroid = X.mean(axis=0)
            sims = cosine_similarity(X, centroid).ravel()
            if keywords:
                kw = [k.lower() for k in keywords]
                boost = np.array([sum(1 for w in re.findall(r"[a-z']+", s.lower()) if w in kw) for s in sents], dtype=float)
                sims = sims + 0.05 * boost
            idx = sims.argsort()[-n_sentences:][::-1]
            return " ".join([sents[i] for i in idx])
        except Exception as e:
            logger.info(f"summary fallback: {e}")
    return " ".join(sents[:n_sentences])

def classify_impact(text: str) -> List[str]:
    tags = []
    lower = text.lower()
    for label, patterns in IMPACT_RULES.items():
        try:
            if any(re.search(p, lower) for p in patterns):
                tags.append(label)
        except Exception:
            continue
    return list(dict.fromkeys(tags)) or ["General"]

def parse_date(date_str: str) -> Optional[dt.datetime]:
    try:
        fmts = [
            "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S", "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d",
            "%d %b %Y", "%B %d, %Y",
        ]
        for fmt in fmts:
            try:
                return dt.datetime.strptime(date_str, fmt)
            except ValueError:
                continue
    except Exception as e:
        logger.info(f"parse_date: {e}")
    return None

# ========================= RSS/Atom parsing (no feedparser) =========================
ATOM_NS = "{http://www.w3.org/2005/Atom}"

def _text(elem: Optional[ET.Element]) -> str:
    return _normalize(elem.text if elem is not None and elem.text else "")

def _find(elem: ET.Element, tag: str) -> Optional[ET.Element]:
    e = elem.find(tag)
    if e is not None:
        return e
    if not tag.startswith("{"):
        e = elem.find(ATOM_NS + tag)
    return e

def _findall(elem: ET.Element, tag: str) -> List[ET.Element]:
    return list(elem.findall(tag)) + list(elem.findall(ATOM_NS + tag))

@st.cache_data(ttl=60*10, show_spinner=False)
def fetch_feed_raw(url: str, timeout: int = 20) -> bytes:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 OneAfricaPulse/1.0",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        }
        r = get_session().get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            soft_fail("Skipped a source that returned a non-200 response.", f"fetch_feed_raw {url} -> {r.status_code}")
        return r.content if r.status_code == 200 else (r.content or b"")
    except Exception as e:
        soft_fail("Temporarily skipped one source due to connectivity.", f"fetch_feed_raw EXC {url}: {e}")
        return b""

def parse_feed_xml(content: bytes, base_url: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not content:
        return items
    try:
        root = ET.fromstring(content)
        channel = root.find("channel")
        if channel is not None:  # RSS
            for it in channel.findall("item"):
                title = _text(_find(it, "title")) or "(untitled)"
                link = _text(_find(it, "link"))
                if not link:
                    guid = _find(it, "guid")
                    link = _text(guid)
                if link and link.startswith("/"):
                    link = urljoin(base_url, link)
                summary = _text(_find(it, "description"))
                pub = _text(_find(it, "pubDate"))
                if title or link:
                    items.append({"title": title, "link": link, "summary": summary, "published_raw": pub})
            return items

        # Atom
        for entry in _findall(root, "entry"):
            title = _text(_find(entry, "title")) or "(untitled)"
            link_el = _find(entry, "link")
            link = ""
            if link_el is not None:
                link = link_el.attrib.get("href", "") or _text(link_el)
            if link and link.startswith("/"):
                link = urljoin(base_url, link)
            summary = _text(_find(entry, "summary")) or _text(_find(entry, "content"))
            pub = _text(_find(entry, "updated")) or _text(_find(entry, "published"))
            if title or link:
                items.append({"title": title, "link": link, "summary": summary, "published_raw": pub})
        return items
    except Exception as e:
        soft_fail("Skipped one feed that had invalid XML.", f"parse_feed_xml EXC {e}")
        return items

def validate_feed(url: str, ignore_recency_check: bool = False) -> Tuple[bool, str]:
    try:
        content = fetch_feed_raw(url)
        items = parse_feed_xml(content, base_url=url)
        if not items:
            return False, "No entries found"
        if ignore_recency_check:
            return True, "OK"
        now = dt.datetime.now(dt.timezone.utc)
        for it in items[:10]:
            d = parse_date(it.get("published_raw", "") or "")
            if d:
                d = d.astimezone(dt.timezone.utc) if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
                if (now - d).days <= 60:
                    return True, "OK"
        return False, "No recent entries (≤60 days)"
    except Exception as ex:
        soft_fail("Skipped one feed due to a validation issue.", f"validate_feed EXC {url}: {ex}")
        return False, "Validation error"

def fetch_from_feed(url: str, start_date: dt.datetime, end_date: dt.datetime,
                    force_fetch: bool, ignore_recency: bool) -> List[Dict[str, Any]]:
    ok, status = validate_feed(url, ignore_recency_check=ignore_recency)
    if not ok and not force_fetch:
        soft_fail(f"Skipped {urlparse(url).netloc} (feed not recent/valid).", f"validate -> {status}")
        return []
    raw = fetch_feed_raw(url)
    raw_items = parse_feed_xml(raw, base_url=url)

    items: List[Dict[str, Any]] = []
    for e in raw_items:
        title = _normalize(e.get("title", ""))
        link = e.get("link", "")
        summary = _normalize(e.get("summary", ""))
        published = None
        if e.get("published_raw"):
            published = parse_date(e["published_raw"])
        if published:
            published = published.astimezone(dt.timezone.utc) if published.tzinfo else published.replace(tzinfo=dt.timezone.utc)
            if not (start_date <= published <= end_date):
                continue
            published_str = published.strftime("%Y-%m-%d %H:%M UTC")
        else:
            published_str = "Date unknown"
        items.append({
            "source": urlparse(url).netloc,
            "title": title,
            "link": link,
            "published": published_str,
            "summary": summary,
        })
    return items

# ========================= Newsdata.io =========================
NEWSDATA_BASE = "https://newsdata.io/api/1/latest"

@st.cache_data(ttl=60*10, show_spinner=False)
def fetch_from_newsdata_cached(params: Dict[str, Any], max_pages: int) -> List[Dict[str, Any]]:
    session = get_session()
    items: List[Dict[str, Any]] = []
    pages = 0
    next_page = None
    while pages < max_pages:
        try:
            q = dict(params)
            if next_page:
                q["page"] = next_page
            r = session.get(NEWSDATA_BASE, params=q, timeout=20)
            if r.status_code != 200:
                soft_fail("One API page was skipped (non-200).", f"newsdata {r.status_code} {r.text[:200]}")
                break
            data = r.json()
            results = data.get("results") or data.get("articles") or []
            for a in results:
                items.append(a)
            next_page = data.get("nextPage") or data.get("next_page")
            pages += 1
            if not next_page:
                break
        except Exception as e:
            soft_fail("Temporarily skipped an API page due to connectivity.", f"newsdata EXC {e}")
            break
    return items

def fetch_from_newsdata(
    api_key: str,
    query: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    language: Optional[str] = None,
    country: Optional[str] = None,
    category: Optional[str] = None,
    max_pages: int = 2,
) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    params = {"apikey": api_key, "q": query or ""}
    if language: params["language"] = language
    if country: params["country"] = country
    if category: params["category"] = category

    items_raw = fetch_from_newsdata_cached(params, max_pages=max_pages)
    items: List[Dict[str, Any]] = []
    for a in items_raw:
        try:
            title = _normalize(a.get("title", ""))
            link = a.get("link") or a.get("url") or ""
            source = a.get("source_id") or a.get("source") or "newsdata.io"
            pub = a.get("pubDate") or a.get("published_at") or ""
            desc = _normalize(a.get("description", "")) or _normalize(a.get("content", ""))

            ok_date = True
            published_str = "Date unknown"
            if pub:
                d = parse_date(pub)
                if d:
                    d = d.astimezone(dt.timezone.utc) if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
                    ok_date = start_date <= d <= end_date
                    published_str = d.strftime("%Y-%m-%d %H:%M UTC")
            if not ok_date:
                continue
            items.append({
                "source": f"{source} (newsdata.io)",
                "title": title or "(untitled)",
                "link": link,
                "published": published_str,
                "summary": desc,
            })
        except Exception as e:
            soft_fail("Skipped one API item due to missing fields.", f"newsdata item EXC {e}")
            continue
    return items

# ========================= UI Helpers =========================
st.set_page_config(page_title=APP_NAME, page_icon="🌍", layout="wide", initial_sidebar_state="expanded")
st.markdown(CARD_CSS, unsafe_allow_html=True)

def make_digest(df: pd.DataFrame, top_k: int = 12) -> str:
    header = f"# {APP_NAME} — Daily Digest\n\n*{TAGLINE}*\n\n> {QUOTE}\n\n"
    if df.empty:
        return header + "_No relevant items found for the selected period._"
    parts = [header, f"**Date:** {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"]
    for _, row in df.head(top_k).iterrows():
        impact = ", ".join(row["impact"])
        parts.append(
            f"### {row['title']}\n"
            f"- **Source:** {row['source']}  \n"
            f"- **Published:** {row['published']}  \n"
            f"- **Relevance:** {row['relevance']:.3f}  \n"
            f"- **Impact:** {impact}  \n"
            f"- **Summary:** {row['auto_summary']}\n"
            f"[Read more]({row['link']})\n\n---\n"
        )
    return "\n".join(parts)

with st.container():
    st.markdown(f"""
    <div class="hero">
        <div class="pill">🌍 OneAfrica Market Pulse</div>
        <h1>{TAGLINE}</h1>
        <p>{QUOTE}</p>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")

    with st.expander("📰 RSS/Atom Sources", expanded=True):
        chosen_sources: List[str] = []
        for name, url in DEFAULT_SOURCES.items():
            if st.checkbox(name, value=True):
                chosen_sources.append(url)
        if st.button("🔄 Check Feeds"):
            for name, url in DEFAULT_SOURCES.items():
                ok, status = validate_feed(url, ignore_recency_check=True)
                st.write(f"{'✅' if ok else '❌'} {name}: {status}")

    with st.expander("🧩 Newsdata.io (optional)", expanded=False):
        st.caption("Merge API headlines with the same scoring & summaries.")
        use_newsdata = st.checkbox("Use Newsdata.io", value=True)
        newsdata_key = st.text_input("API Key", value="pub_72ee7f1de10849be8847f7ad4e1b8810", type="password")
        newsdata_query = st.text_input("Query", value="tree crop commodities")
        c1, c2, c3 = st.columns(3)
        with c1: nd_language = st.text_input("Language (e.g., en, fr)", value="")
        with c2: nd_country = st.text_input("Country (e.g., gh, ng, ci)", value="")
        with c3: nd_category = st.text_input("Category (e.g., business)", value="")
        nd_pages = st.slider("Newsdata pages", 1, 5, 2)

    with st.expander("📅 Date Range", expanded=True):
        mode = st.radio("Mode", ["Quick Select", "Custom"], horizontal=True)
        if mode == "Quick Select":
            quick = {"Last 24 Hours": 1, "Last 3 Days": 3, "Last Week": 7, "Last 2 Weeks": 14, "Last Month": 30}
            sel = st.selectbox("Window", list(quick.keys()), index=2)
            end_date = dt.datetime.now(dt.timezone.utc)
            start_date = end_date - dt.timedelta(days=quick[sel])
        else:
            c1, c2 = st.columns(2)
            with c1: sd = st.date_input("Start", value=dt.date.today() - dt.timedelta(days=7))
            with c2: ed = st.date_input("End", value=dt.date.today())
            start_date = dt.datetime.combine(sd, dt.time.min, tzinfo=dt.timezone.utc)
            end_date = dt.datetime.combine(ed, dt.time.max, tzinfo=dt.timezone.utc)

    with st.expander("🔍 Keywords & Filters", expanded=True):
        custom_kw = st.text_area("Keywords (comma-separated)", ", ".join(DEFAULT_KEYWORDS), height=100)
        keywords = [k.strip() for k in custom_kw.split(",") if k.strip()]
        min_relevance = st.slider("Min relevance", 0.0, 1.0, 0.05, 0.01)
        per_source_cap = st.slider("Max articles per source", 5, 50, 20)

    with st.expander("📝 Content Settings", expanded=True):
        n_sent = st.slider("Sentences per summary", 2, 6, 3)
        top_k = st.slider("Digest: top items", 5, 30, 12)

    st.markdown("---")
    st.subheader("🛡️ Resilience")
    force_fetch = st.checkbox("⚡ Force RSS fetch if validation fails", value=True)
    ignore_recency = st.checkbox("🕒 Ignore RSS recency check", value=True)
    dedupe_across_sources = st.checkbox("🧹 Deduplicate across sources", value=True)

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        run_btn = st.button("🚀 Scan Now", use_container_width=True)
    with colB:
        if st.button("♻️ Reset", use_container_width=True):
            st.rerun()

# ========================= Processing =========================
def hash_key(*parts) -> str:
    return hashlib.md5(("||".join([p or "" for p in parts])).encode("utf-8")).hexdigest()

def process_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["source","published","title","relevance","impact","auto_summary","link","image"])
    # Deduplicate by (title, link)
    seen = set()
    cleaned = []
    for r in rows:
        key = hash_key(r.get("title",""), r.get("link",""))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(r)
    df = pd.DataFrame(cleaned)
    if df.empty:
        return pd.DataFrame(columns=["source","published","title","relevance","impact","auto_summary","link","image"])
    return df.sort_values("relevance", ascending=False).reset_index(drop=True)

def enrich(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        article_text, image_url = fetch_article_text_and_image(entry.get("link",""))
        base = entry.get("summary") or ""
        body = article_text if len(article_text) > len(base) else base

        rel = keyword_relevance(" ".join([entry.get("title",""), body]), keywords)
        if rel < min_relevance:
            return None
        summary = simple_extractive_summary(body, n_sentences=n_sent, keywords=keywords)
        impacts = classify_impact(" ".join([entry.get("title",""), body])) or ["General"]

        return {
            "source": entry.get("source",""),
            "title": entry.get("title","(untitled)"),
            "link": entry.get("link",""),
            "published": entry.get("published","Date unknown"),
            "relevance": float(rel),
            "impact": impacts,
            "auto_summary": summary,
            "image": image_url or FALLBACK_IMG,
        }
    except Exception as e:
        soft_fail("Skipped one article that couldn’t be processed.", f"enrich EXC {e}")
        return None

def fetch_all(chosen_sources: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total_tasks = len(chosen_sources) + (1 if (use_newsdata and newsdata_key) else 0)
    total_tasks = max(total_tasks, 1)
    progress = st.progress(0.0)
    info = st.empty()

    # 1) RSS/Atom
    for i, src in enumerate(chosen_sources, start=1):
        info.info(f"Fetching RSS {i}/{total_tasks}: {urlparse(src).netloc}")
        try:
            raw_items = fetch_from_feed(src, start_date, end_date, force_fetch, ignore_recency)
        except Exception as e:
            soft_fail("Skipped a source due to a transient issue.", f"fetch_from_feed EXC {src}: {e}")
            raw_items = []
        if per_source_cap and raw_items:
            raw_items = raw_items[:per_source_cap]

        if raw_items:
            with ThreadPoolExecutor(max_workers=6) as ex:
                futures = [ex.submit(enrich, {**e, "source": urlparse(src).netloc}) for e in raw_items]
                for fut in as_completed(futures):
                    try:
                        r = fut.result()
                        if r: rows.append(r)
                    except Exception as e:
                        soft_fail("One article was skipped during processing.", f"future enrich EXC {e}")
        progress.progress(min(1.0, i / total_tasks))

    # 2) Newsdata.io
    if use_newsdata and newsdata_key:
        info.info(f"Fetching Newsdata.io {len(chosen_sources)+1}/{total_tasks}")
        try:
            nd_items = fetch_from_newsdata(
                api_key=newsdata_key,
                query=newsdata_query,
                start_date=start_date,
                end_date=end_date,
                language=nd_language or None,
                country=nd_country or None,
                category=nd_category or None,
                max_pages=nd_pages,
            )
            if per_source_cap and nd_items:
                nd_items = nd_items[:per_source_cap]
            if nd_items:
                with ThreadPoolExecutor(max_workers=6) as ex:
                    futures = [ex.submit(enrich, it) for it in nd_items]
                    for fut in as_completed(futures):
                        try:
                            r = fut.result()
                            if r: rows.append(r)
                        except Exception as e:
                            soft_fail("One API article was skipped during processing.", f"future API enrich EXC {e}")
        except Exception as e:
            soft_fail("The API was briefly unavailable; results shown are from RSS.", f"fetch_from_newsdata EXC {e}")
        progress.progress(1.0)

    info.empty()
    progress.empty()
    return rows

def render_card(row: pd.Series):
    pub = row["published"]
    src = row["source"]
    rel = f"{row['relevance']:.0%}"
    title = row["title"]
    link = row["link"]
    img = row["image"] or FALLBACK_IMG
    summary = row["auto_summary"] or ""
    st.markdown(f"""
    <div class="card">
      <img class="thumb" src="{img}" alt="thumbnail">
      <div class="card-body">
        <div class="meta">{src} · {pub} · Relevance {rel}</div>
        <div class="title">{title}</div>
        <div class="badges">
            {"".join([f'<span class="badge">{t}</span>' for t in row["impact"]])}
        </div>
        <div class="summary">{summary}</div>
        <div style="margin-top:10px;">
          <a class="link" href="{link}" target="_blank">Read full article →</a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def ui_results(df: pd.DataFrame, top_k: int):
    st.subheader("📊 Results")
    if df.empty:
        st.warning("No relevant articles found. Try widening the date range or lowering the relevance threshold.")
        return

    c1, c2 = st.columns(2)
    with c1:
        all_impacts = sorted({t for tags in df["impact"] for t in tags})
        impact_filter = st.multiselect("Filter by impact", options=all_impacts, default=[])
    with c2:
        source_filter = st.multiselect("Filter by source", options=sorted(df["source"].unique()), default=[])

    filtered = df.copy()
    if impact_filter:
        filtered = filtered[filtered["impact"].apply(lambda x: any(t in x for t in impact_filter))]
    if source_filter:
        filtered = filtered[filtered["source"].isin(source_filter)]

    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for _, row in filtered.iterrows():
        render_card(row)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📝 Daily Digest")
    digest_md = make_digest(filtered if (impact_filter or source_filter) else df, top_k=top_k)
    st.markdown(digest_md)

    st.subheader("⬇️ Downloads")
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    export_df = filtered if (impact_filter or source_filter) else df
    csv_name = f"oneafrica_pulse_{ts}.csv"
    md_name = f"oneafrica_pulse_digest_{ts}.md"
    st.download_button("📥 Download CSV", data=export_df.to_csv(index=False).encode("utf-8"),
                       file_name=csv_name, mime="text/csv")
    st.download_button("📥 Download Digest (Markdown)", data=digest_md.encode("utf-8"),
                       file_name=md_name, mime="text/markdown")
    st.info("💡 Tip: Paste the Markdown into an email, WhatsApp (as a code block), or your wiki for quick sharing.")

def friendly_error_summary():
    """Show a compact, friendly banner summarizing soft issues (no stack traces)."""
    if not SOFT_ERRORS:
        return
    # Count distinct messages and occurrences
    counts: Dict[str,int] = {}
    for m in SOFT_ERRORS:
        counts[m] = counts.get(m, 0) + 1
    bullets = "".join([f"- {msg} _(x{n})_\n" for msg, n in counts.items()])
    st.info(f"""
**Heads up:** Some sources were temporarily skipped or partially loaded.  
This doesn’t affect your ability to scan and summarize current items.

{bullets}
    """)

# ========================= Main =========================
st.markdown(CARD_CSS, unsafe_allow_html=True)

if not run_btn:
    st.info("""
**What this demo does:**
- 📰 Scans curated RSS/Atom feeds (+ optional Newsdata.io API) for the last *N* days  
- 📑 Fetches full article text where possible + **thumbnails** (Open Graph)  
- 🎯 Scores relevance against **your commodity & policy keywords**  
- 📝 Auto-summarizes into 2–6 sentences  
- 🏷️ Tags each item (Supply Risk, FX & Policy, Logistics, etc.)  
- 💾 Outputs a **downloadable CSV** and **Daily Digest (Markdown)**
    """)
else:
    try:
        if not chosen_sources and not (use_newsdata and newsdata_key):
            st.error("Pick at least one RSS source or enable Newsdata.io.")
        else:
            with st.spinner("Scanning sources, extracting content, and generating summaries..."):
                rows = fetch_all(chosen_sources)
                df = process_rows(rows)
                ui_results(df, top_k)
    except Exception as e:
        # Final safety net—never leak stack traces to UI
        soft_fail("Something went wrong while assembling the results.", f"MAIN EXC {e}")
        st.error("We ran into a hiccup assembling the results. Please try again or adjust your filters.")
    finally:
        # Show friendly banner summarizing non-critical issues encountered
        friendly_error_summary()
