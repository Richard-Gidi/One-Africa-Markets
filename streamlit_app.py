# OneAfrica Market Pulse — Automated Market Intelligence (Streamlit Demo)
# Author: Richard Gidi
# Focus: News-only, RSS/Atom parsing without 'feedparser', optional Newsdata.io API, robust validation, and friendly fallbacks.
# Run: streamlit run streamlit_app.py

import os
import re
import html
import json
import datetime as dt
import logging
from typing import List, Dict, Tuple, Optional, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin

import pandas as pd
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import xml.etree.ElementTree as ET

# --------------------- Optional sklearn (graceful fallback) ---------------------
HAS_SK = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    HAS_SK = False

# --------------------- Logging ---------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("oneafrica.pulse")

# --------------------- App strings ---------------------
APP_NAME = "One Africa Market Pulse"
TAGLINE = "Automated intelligence for cashew, shea, cocoa & allied markets."
QUOTE = "“Ask your data why, until it has nothing else to say.” — Richard Gidi"

DEFAULT_KEYWORDS = [
    "cashew", "shea", "shea nut", "cocoa", "palm kernel", "agri", "export", "harvest",
    "shipment", "freight", "logistics", "port", "tariff", "ban", "fx", "currency",
    "cedi", "naira", "inflation", "subsidy", "cooperative", "value-addition", "processing",
    "ghana", "nigeria", "cote d’ivoire", "ivory coast", "benin", "togo", "burkina",
    "west africa", "sahel", "trade policy", "commodity", "price", "market"
]

# --------------------- Curated working feeds (no feedparser needed) ---------------------
DEFAULT_SOURCES = {
    # AllAfrica official categories
    "AllAfrica » Agriculture": "https://allafrica.com/tools/headlines/rdf/agriculture/headlines.rdf",
    "AllAfrica » Business": "https://allafrica.com/tools/headlines/rdf/business/headlines.rdf",

    # Kenya — The Standard (section RSS)
    "The Standard » Business": "https://www.standardmedia.co.ke/rss/business.php",
    "The Standard » Agriculture": "https://www.standardmedia.co.ke/rss/agriculture.php",

    # Ghana — Citi Newsroom
    "CitiNewsroom": "https://citinewsroom.com/feed/",

    # FAO — official feeds
    "FAO News (All topics)": "https://www.fao.org/news/rss/en",
    "FAO GIEWS": "https://www.fao.org/giews/rss/en/",

    # Fresh produce / agri trade (Africa channel)
    "FreshPlaza Africa": "https://www.freshplaza.com/africa/rss.xml",

    # African Arguments
    "African Arguments": "https://africanarguments.org/feed/",

    # How We Made It In Africa (business stories)
    "How We Made It In Africa": "https://www.howwemadeitinafrica.com/feed/",

    # Bizcommunity – generated: agriculture (63) + logistics (76), region Africa
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
        r"\bproduction (?:issue|problem|concern)\b"
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
        r"\bquality premium\b"
    ],
    "Price Downside": [
        r"\b(?:oversupply|surplus|glut)\b",
        r"\b(?:decline|fall|drop|decrease|slump)\b",
        r"\b(?:weak|soft|bearish) (?:price|market|demand)\b",
        r"\bcut (?:price|rate|cost)\b",
        r"\blow(?:er)? (?:price|demand|consumption)\b",
        r"\bmarket (?:weakness|downturn)\b",
        r"\bcompetitive pressure\b"
    ],
    "FX & Policy": [
        r"\b(?:depreciation|devaluation)\b",
        r"\bweak(?:ening)? (?:currency|exchange)\b",
        r"\b(?:fx|forex|dollar|euro|yuan)\b",
        r"\b(?:monetary|fiscal|trade) policy\b",
        r"\b(?:interest|exchange) rate\b",
        r"\b(?:tariff|duty|levy|tax)\b",
        r"\bregulatory (?:change|update|requirement)\b",
        r"\bpolicy (?:change|update|reform)\b"
    ],
    "Logistics & Trade": [
        r"\b(?:freight|shipping|transport)\b",
        r"\b(?:port|container|vessel|cargo)\b",
        r"\b(?:congestion|delay|bottleneck)\b",
        r"\b(?:reroute|divert|alternative route)\b",
        r"\b(?:cost|rate) (?:increase|surge|rise)\b",
        r"\btrade (?:flow|route|pattern)\b",
        r"\b(?:export|import) (?:volume|data|figure)\b"
    ],
    "Market Structure": [
        r"\b(?:merger|acquisition|takeover)\b",
        r"\b(?:investment|expansion|capacity)\b",
        r"\b(?:processing|factory|facility)\b",
        r"\b(?:certification|standard|quality)\b",
        r"\b(?:cooperative|association|group)\b",
        r"\b(?:contract|agreement|deal)\b",
        r"\b(?:partnership|collaboration)\b",
        r"\bmarket (?:structure|reform|development)\b"
    ],
    "Tech & Innovation": [
        r"\b(?:technology|innovation|digital)\b",
        r"\b(?:blockchain|traceability|tracking)\b",
        r"\b(?:sustainability|sustainable)\b",
        r"\b(?:efficiency|optimization)\b",
        r"\b(?:automation|mechanization)\b",
        r"\b(?:research|development|r&d)\b",
        r"\b(?:startup|fintech|agtech)\b"
    ]
}

# --------------------- HTTP utils ---------------------
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

@lru_cache(maxsize=100)
def fetch_article_text(url: str, timeout: int = 12) -> str:
    if not url:
        return ""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; OneAfricaPulse/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        session = get_session()
        r = session.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "iframe", "form"]):
            tag.decompose()
        candidates = []
        selectors = [
            "article", "[role='main']", ".article-body", ".story-body",
            ".post-content", "main", ".content", ".entry-content",
            "#article-body", ".article-content", ".story-content",
            ".news-content", ".page-content", "body"
        ]
        for sel in selectors:
            for node in soup.select(sel):
                text = node.get_text(separator=" ", strip=True)
                if len(text) > 100:
                    candidates.append(text)
        text = max(candidates, key=len) if candidates else soup.get_text(separator=" ", strip=True)
        text = _normalize(text)
        return text if len(text) >= 50 else ""
    except Exception as e:
        logger.warning(f"Article fetch failed: {url} ({e})")
        return ""

# --------------------- Relevance & Summary ---------------------
def keyword_relevance(text: str, keywords: List[str]) -> float:
    if not text:
        return 0.0
    if HAS_SK:
        try:
            corpus = [text]
            query = " ".join(keywords)
            vec = TfidfVectorizer(stop_words="english", max_features=5000)
            X = vec.fit_transform(corpus + [query])
            sim = cosine_similarity(X[0:1], X[-1])[0][0]
            return float(sim)
        except Exception:
            pass
    # fallback: simple ratio of keyword hits
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
            sims = cosine_similarity(X, centroid)
            sims = np.array(sims).ravel()
            if keywords:
                kw = [k.lower() for k in keywords]
                boost = np.array([sum(1 for w in re.findall(r"[a-z']+", s.lower()) if w in kw) for s in sents], dtype=float)
                sims = sims + 0.05 * boost
            idx = sims.argsort()[-n_sentences:][::-1]
            chosen = [sents[i] for i in idx]
            return " ".join(chosen)
        except Exception:
            pass
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
    return list(dict.fromkeys(tags))

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
        return None
    except Exception:
        return None

# --------------------- RSS/Atom parser (no feedparser) ---------------------
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

def parse_feed_xml(content: bytes, base_url: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(content)

        # RSS: <channel><item>...
        channel = root.find("channel")
        if channel is not None:
            for it in channel.findall("item"):
                title = _text(_find(it, "title"))
                link = _text(_find(it, "link"))
                if not link:
                    guid = _find(it, "guid")
                    link = _text(guid)
                if link and link.startswith("/"):
                    link = urljoin(base_url, link)
                summary = _text(_find(it, "description"))
                pub = _text(_find(it, "pubDate"))
                if not title and not link:
                    continue
                items.append({
                    "title": title or "(untitled)",
                    "link": link,
                    "summary": summary,
                    "published_raw": pub
                })
            return items

        # Atom: <entry>...
        for entry in _findall(root, "entry"):
            title = _text(_find(entry, "title"))
            link_el = _find(entry, "link")
            link = ""
            if link_el is not None:
                link = link_el.attrib.get("href", "") or _text(link_el)
            if link and link.startswith("/"):
                link = urljoin(base_url, link)
            summary = _text(_find(entry, "summary")) or _text(_find(entry, "content"))
            pub = _text(_find(entry, "updated")) or _text(_find(entry, "published"))
            if not title and not link:
                continue
            items.append({
                "title": title or "(untitled)",
                "link": link,
                "summary": summary,
                "published_raw": pub
            })
        return items
    except Exception as e:
        logger.error(f"XML parse error: {e}")
        return items

def validate_feed(url: str, ignore_recency_check: bool = False) -> Tuple[bool, str]:
    """Lenient validation using requests + our XML parser."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 OneAfricaPulse/1.0",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"
        }
        session = get_session()
        r = session.get(url, headers=headers, timeout=20, allow_redirects=True)
        content = r.content if r.status_code == 200 else (r.content or b"")
        items = parse_feed_xml(content, base_url=url)
        if not items:
            return False, "No entries found"

        if not ignore_recency_check:
            now = dt.datetime.now(dt.timezone.utc)
            recent = False
            for it in items[:10]:
                d = parse_date(it.get("published_raw", "") or "")
                if d:
                    if d.tzinfo:
                        d = d.astimezone(dt.timezone.utc)
                    else:
                        d = d.replace(tzinfo=dt.timezone.utc)
                    if (now - d).days <= 60:  # relaxed window
                        recent = True
                        break
            if not recent:
                return False, "No recent entries (≤60 days)"
        return True, "OK"
    except Exception as ex:
        return False, f"Validation error: {ex}"

# --------------------- Newsdata.io integration ---------------------
NEWSDATA_BASE = "https://newsdata.io/api/1/latest"

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
    """
    Fetch items from Newsdata.io and map to our common schema.
    Uses incremental pagination with 'nextPage'.
    """
    if not api_key:
        return []
    params = {
        "apikey": api_key,
        "q": query or "",
    }
    # Newsdata latest endpoint supports filters like language, country, category (depending on plan).
    if language: params["language"] = language
    if country: params["country"] = country
    if category: params["category"] = category

    items: List[Dict[str, Any]] = []
    session = get_session()
    next_page = None
    pages = 0

    while pages < max_pages:
        try:
            local_params = dict(params)
            if next_page:
                local_params["page"] = next_page
            r = session.get(NEWSDATA_BASE, params=local_params, timeout=20)
            r.raise_for_status()
            data = r.json()

            results = data.get("results") or data.get("articles") or []
            for a in results:
                title = _normalize(a.get("title", ""))
                link = a.get("link") or a.get("url") or ""
                source = a.get("source_id") or a.get("source") or "newsdata.io"
                pub = a.get("pubDate") or a.get("published_at") or ""
                desc = _normalize(a.get("description", "")) or _normalize(a.get("content", ""))

                # date filter
                published_str = "Date unknown"
                ok_date = True
                if pub:
                    d = parse_date(pub)
                    if d:
                        if d.tzinfo:
                            d = d.astimezone(dt.timezone.utc)
                        else:
                            d = d.replace(tzinfo=dt.timezone.utc)
                        ok_date = start_date <= d <= end_date
                        published_str = d.strftime("%Y-%m-%d %H:%M UTC")
                if not ok_date:
                    continue

                items.append({
                    "source": f"{source} (newsdata.io)",
                    "title": title or "(untitled)",
                    "link": link,
                    "published": published_str,
                    "summary": desc
                })

            next_page = data.get("nextPage") or data.get("next_page")
            pages += 1
            if not next_page:
                break
        except Exception as e:
            logger.warning(f"Newsdata fetch warning: {e}")
            break
    return items

# --------------------- Fetch from RSS/Atom ---------------------
def fetch_from_feed(url: str, start_date: dt.datetime, end_date: dt.datetime,
                    force_fetch: bool, ignore_recency: bool) -> List[Dict[str, Any]]:
    logger.info(f"Fetching feed from: {url}")

    ok, status = validate_feed(url, ignore_recency_check=ignore_recency)
    if not ok and not force_fetch:
        st.warning(f"⚠️ Failed to validate feed {urlparse(url).netloc}: {status}")
        return []
    elif not ok and force_fetch:
        st.warning(f"⚠️ Validation failed for {urlparse(url).netloc} ({status}). Forcing fetch...")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; OneAfricaPulse/1.0)",
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml"
    }
    session = get_session()
    r = session.get(url, headers=headers, timeout=20)
    content = r.content if r.status_code == 200 else b""
    raw_items = parse_feed_xml(content, base_url=url)

    items: List[Dict[str, Any]] = []
    for e in raw_items:
        title = _normalize(e.get("title", ""))
        link = e.get("link", "")
        summary = _normalize(e.get("summary", ""))
        published = None

        if e.get("published_raw"):
            published = parse_date(e["published_raw"])

        if published:
            if published.tzinfo:
                published = published.astimezone(dt.timezone.utc)
            else:
                published = published.replace(tzinfo=dt.timezone.utc)
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
            "summary": summary
        })
    logger.info(f"Found {len(items)} date-filtered entries in feed {url}")
    return items

def make_digest(df: pd.DataFrame, top_k: int = 12) -> str:
    header = f"# {APP_NAME} — Daily Digest\n\n*{TAGLINE}*\n\n> {QUOTE}\n\n"
    if df.empty:
        return header + "_No relevant items found for the selected period._"
    parts = [header, f"**Date:** {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"]
    for _, row in df.head(top_k).iterrows():
        parts.append(
            f"### {row['title']}\n"
            f"- **Source:** {row['source']}  \n"
            f"- **Published:** {row['published']}  \n"
            f"- **Relevance:** {row['relevance']:.3f}  \n"
            f"- **Impact:** {', '.join(row['impact'])}  \n"
            f"- **Summary:** {row['auto_summary']}\n"
            f"[Read more]({row['link']})\n\n---\n"
        )
    return "\n".join(parts)

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title=APP_NAME, page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

st.title("🌍 One Africa Market Pulse")
st.caption(TAGLINE)
st.markdown(f"<blockquote style='font-style: italic; margin: 1em 0;'>{QUOTE}</blockquote>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")

    with st.expander("📰 News Sources (RSS/Atom)", expanded=True):
        st.markdown("Select the sources to scan for market intelligence:")

        chosen_sources: List[str] = []
        categories = {
            "AllAfrica (official)": [("AllAfrica » Agriculture", DEFAULT_SOURCES["AllAfrica » Agriculture"]),
                                     ("AllAfrica » Business", DEFAULT_SOURCES["AllAfrica » Business"])],
            "National & Regional": [("The Standard » Business", DEFAULT_SOURCES["The Standard » Business"]),
                                    ("The Standard » Agriculture", DEFAULT_SOURCES["The Standard » Agriculture"]),
                                    ("CitiNewsroom", DEFAULT_SOURCES["CitiNewsroom"])],
            "Multilateral & Trade": [("FAO News (All topics)", DEFAULT_SOURCES["FAO News (All topics)"]),
                                     ("FAO GIEWS", DEFAULT_SOURCES["FAO GIEWS"])],
            "Agri/Logistics Trade Press": [("FreshPlaza Africa", DEFAULT_SOURCES["FreshPlaza Africa"]),
                                           ("Bizcommunity (Africa • Agri+Logistics)", DEFAULT_SOURCES["Bizcommunity (Africa • Agri+Logistics)"])],
            "Analysis & Features": [("African Arguments", DEFAULT_SOURCES["African Arguments"]),
                                    ("How We Made It In Africa", DEFAULT_SOURCES["How We Made It In Africa"])],
        }

        for category, sources in categories.items():
            st.markdown(f"#### {category}")
            for name, url in sources:
                if st.checkbox(name, value=True):
                    chosen_sources.append(url)
            st.markdown("---")

        if st.button("🔄 Check Feed Status"):
            st.markdown("### Feed Status Check")
            for name, url in DEFAULT_SOURCES.items():
                ok, status = validate_feed(url, ignore_recency_check=True)
                if ok:
                    st.success(f"✅ {name}: {status}")
                else:
                    st.error(f"❌ {name}: {status}")
            st.markdown("---")

    with st.expander("🧩 Newsdata.io (optional API)", expanded=False):
        st.caption("Adds API-based headlines; merged with the same scoring, tags & digest.")
        use_newsdata = st.checkbox("Use Newsdata.io", value=True)
        # Pre-fill with the public key you shared. You can override with your private key.
        newsdata_key = st.text_input("API Key", value="pub_72ee7f1de10849be8847f7ad4e1b8810", type="password")
        newsdata_query = st.text_input("Query", value="tree crop commodities")
        col_nd1, col_nd2, col_nd3 = st.columns(3)
        with col_nd1:
            nd_language = st.text_input("Language (e.g. en, fr) — optional", value="")
        with col_nd2:
            nd_country = st.text_input("Country code (e.g. gh, ng, ci) — optional", value="")
        with col_nd3:
            nd_category = st.text_input("Category (e.g. business, world) — optional", value="")
        max_pages = st.slider("Newsdata pages (pagination)", 1, 5, 2, help="Each page can include multiple results (subject to plan).")

    with st.expander("📅 Date Range", expanded=True):
        date_option = st.radio("Date Selection Mode", ["Quick Select", "Custom Range"], horizontal=True)
        if date_option == "Quick Select":
            quick_options = {"Last 24 Hours": 1, "Last 3 Days": 3, "Last Week": 7, "Last 2 Weeks": 14, "Last Month": 30}
            selected_quick = st.selectbox("Select time period", list(quick_options.keys()), index=2)
            days_back = quick_options[selected_quick]
            end_date = dt.datetime.now(dt.timezone.utc)
            start_date = end_date - dt.timedelta(days=days_back)
        else:
            c1, c2 = st.columns(2)
            with c1:
                sd = st.date_input("Start Date", value=dt.date.today() - dt.timedelta(days=7))
            with c2:
                ed = st.date_input("End Date", value=dt.date.today())
            start_date = dt.datetime.combine(sd, dt.time.min, tzinfo=dt.timezone.utc)
            end_date = dt.datetime.combine(ed, dt.time.max, tzinfo=dt.timezone.utc)

    with st.expander("🔍 Keywords & Filters", expanded=True):
        custom_kw = st.text_area("Add or edit keywords (comma-separated)", value=", ".join(DEFAULT_KEYWORDS), height=120)
        keywords = [k.strip() for k in custom_kw.split(",") if k.strip()] or DEFAULT_KEYWORDS
        min_relevance = st.slider("Minimum relevance score", 0.0, 1.0, 0.05, 0.01)

    with st.expander("📝 Content Settings", expanded=True):
        n_sent = st.slider("Sentences per summary", 2, 6, 3)
        top_k = st.slider("Top items in digest", 5, 30, 12)

    st.markdown("---")
    st.subheader("Resilience Options")
    force_fetch = st.checkbox("⚡ Force fetch RSS even if validation fails", value=True)
    ignore_recency = st.checkbox("🕒 Ignore RSS recency check (accept older feeds)", value=True)

    st.markdown("---")
    col_run, col_reset = st.columns(2)
    with col_run:
        run_btn = st.button("🚀 Scan Now", use_container_width=True)
    with col_reset:
        if st.button("♻️ Reset", use_container_width=True):
            st.rerun()

# --------------------- Processing ---------------------
def process_sources() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)

    def enrich_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            text = fetch_article_text(entry["link"]) if entry.get("link") else ""
            base = entry.get("summary", "") or ""
            body = text if len(text) > len(base) else base

            relevance = keyword_relevance(" ".join([entry["title"], body]), keywords)
            if relevance < min_relevance:
                return None

            summary = simple_extractive_summary(body, n_sentences=n_sent, keywords=keywords)
            impact_tags = classify_impact(" ".join([entry["title"], body]))
            return {
                "source": entry["source"],
                "title": entry["title"],
                "link": entry["link"],
                "published": entry["published"],
                "relevance": relevance,
                "impact": impact_tags,
                "auto_summary": summary
            }
        except Exception as e:
            logger.error(f"Failed to process entry {entry.get('link','')}: {e}")
            return None

    # 1) Gather from RSS sources
    total_sources = len(chosen_sources) + (1 if use_newsdata else 0)
    completed_sources = 0

    if chosen_sources:
        for idx, src in enumerate(chosen_sources, 1):
            try:
                progress_placeholder.text(f"Processing source {idx}/{total_sources}: {urlparse(src).netloc}")
                status_placeholder.info(f"Fetching RSS from {urlparse(src).netloc}...")
                entries_raw = fetch_from_feed(src, start_date, end_date, force_fetch, ignore_recency)
                completed_sources += 1
                if not entries_raw:
                    status_placeholder.warning(f"No entries found in feed: {src}")
                    progress_bar.progress(completed_sources / total_sources)
                    continue

                status_placeholder.info(f"Processing {len(entries_raw)} RSS articles from {urlparse(src).netloc}...")
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(enrich_entry, {**e, "source": urlparse(src).netloc}): e for e in entries_raw}
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res: rows.append(res)
                progress_bar.progress(completed_sources / total_sources)

            except Exception as ex:
                logger.error(f"Failed to process RSS {src}: {ex}")
                st.warning(f"Failed to process {urlparse(src).netloc}: {ex}")
                progress_bar.progress(completed_sources / total_sources)

    # 2) Gather from Newsdata.io (optional)
    if use_newsdata and newsdata_key:
        try:
            progress_placeholder.text(f"Processing source {completed_sources+1}/{total_sources}: newsdata.io")
            status_placeholder.info("Fetching Newsdata.io API results...")
            nd_items = fetch_from_newsdata(
                api_key=newsdata_key,
                query=newsdata_query,
                start_date=start_date,
                end_date=end_date,
                language=nd_language or None,
                country=nd_country or None,
                category=nd_category or None,
                max_pages=max_pages,
            )
            completed_sources += 1
            if nd_items:
                status_placeholder.info(f"Processing {len(nd_items)} Newsdata.io items...")
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(enrich_entry, it): it for it in nd_items}
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res: rows.append(res)
            else:
                status_placeholder.warning("No Newsdata.io results for this query/time window.")
            progress_bar.progress(completed_sources / total_sources)
        except Exception as ex:
            logger.warning(f"Newsdata.io processing error: {ex}")
            progress_bar.progress(completed_sources / total_sources)

    progress_placeholder.empty()
    status_placeholder.empty()
    progress_bar.empty()
    return rows

def make_digest(df: pd.DataFrame, top_k: int = 12) -> str:
    header = f"# {APP_NAME} — Daily Digest\n\n*{TAGLINE}*\n\n> {QUOTE}\n\n"
    if df.empty:
        return header + "_No relevant items found for the selected period._"
    parts = [header, f"**Date:** {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"]
    for _, row in df.head(top_k).iterrows():
        parts.append(
            f"### {row['title']}\n"
            f"- **Source:** {row['source']}  \n"
            f"- **Published:** {row['published']}  \n"
            f"- **Relevance:** {row['relevance']:.3f}  \n"
            f"- **Impact:** {', '.join(row['impact'])}  \n"
            f"- **Summary:** {row['auto_summary']}\n"
            f"[Read more]({row['link']})\n\n---\n"
        )
    return "\n".join(parts)

def main_app():
    if not run_btn:
        st.info("""
**What this demo does:**  
- 📰 Scans curated RSS/Atom feeds (and optional Newsdata.io API) for the last *N* days  
- 📑 Fetches full article text where possible  
- 🎯 Scores relevance against your **commodity & policy keywords**  
- 📝 Auto-summarizes into 2–6 sentences  
- 🏷️ Tags each item with impact labels (Supply Risk, FX & Policy, Logistics, etc.)  
- 💾 Produces a **downloadable CSV** and **Daily Digest (Markdown)**
        """)
        st.markdown("This is a lightweight, API-friendly prototype — perfect for a live demo with One Africa Markets.")
        return

    # guard
    if not chosen_sources and not (use_newsdata and newsdata_key):
        st.error("Please select at least one RSS source or enable Newsdata.io in the sidebar.")
        return

    with st.spinner("Scanning feeds/APIs, extracting content, and generating summaries..."):
        rows = process_sources()
        if not rows:
            st.warning("No relevant articles found matching your criteria. Tip: widen your date window or lower the relevance threshold.")
            return

        df = pd.DataFrame(rows).sort_values(by=["relevance"], ascending=False).reset_index(drop=True)

        tab1, tab2, tab3 = st.tabs(["📊 Results", "📝 Daily Digest", "⬇️ Downloads"])

        with tab1:
            st.success(f"Found {len(df)} relevant items.")
            col1, col2 = st.columns(2)
            with col1:
                impact_filter = st.multiselect(
                    "Filter by impact",
                    options=sorted(set(tag for tags in df['impact'] for tag in tags)),
                    default=[]
                )
            with col2:
                source_filter = st.multiselect(
                    "Filter by source",
                    options=sorted(df['source'].unique()),
                    default=[]
                )

            filtered_df = df.copy()
            if impact_filter:
                filtered_df = filtered_df[filtered_df['impact'].apply(lambda x: any(tag in x for tag in impact_filter))]
            if source_filter:
                filtered_df = filtered_df[filtered_df['source'].isin(source_filter)]

            st.dataframe(
                filtered_df[["source", "published", "title", "relevance", "impact", "auto_summary", "link"]],
                height=450,
                use_container_width=True
            )

        with tab2:
            digest_md = make_digest(filtered_df if (impact_filter or source_filter) else df, top_k=top_k)
            st.markdown(digest_md)

        with tab3:
            ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            export_df = filtered_df if (impact_filter or source_filter) else df
            csv_name = f"oneafrica_pulse_{ts}.csv"
            md_name = f"oneafrica_pulse_digest_{ts}.md"
            export_df.to_csv(csv_name, index=False)
            with open(md_name, "w", encoding="utf-8") as f:
                f.write(digest_md)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📥 Download CSV", data=open(csv_name, "rb"), file_name=csv_name, mime="text/csv")
            with c2:
                st.download_button("📥 Download Digest (Markdown)", data=open(md_name, "rb"), file_name=md_name, mime="text/markdown")
            st.info("💡 Tip: Paste the Markdown into an email, WhatsApp (as a code block), or your wiki for quick sharing.")

if __name__ == "__main__":
    main_app()
