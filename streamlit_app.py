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
import logging
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin

import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import xml.etree.ElementTree as ET

# --------------------- Optional imports (graceful fallback) ---------------------
# If scikit-learn isn't installed, we still run (fallback scorers/summarizer kick in).
HAS_SK = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    HAS_SK = False

# --------------------- Logging ---------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

DEFAULT_SOURCES = {
    # Major African Business News
    "AllAfrica Business": "https://allafrica.com/tools/headlines/rdf/business/headlines.rdf",
    "AllAfrica Agriculture": "https://allafrica.com/tools/headlines/rdf/agriculture/headlines.rdf",
    "Africa News Agency": "https://www.africannewsagency.com/feed",
    "African Arguments": "https://africanarguments.org/feed/",
    "Africa Feeds": "https://africafeeds.com/feed/",
    # Agriculture & Food Business
    "FoodBusiness Africa": "https://www.foodbusinessafrica.com/feed/",
    "AgFunderNews": "https://agfundernews.com/feed",
    "Fresh Plaza Africa": "https://www.freshplaza.com/africa/rss.xml",
    # Trade & Markets
    "Global Trade Review": "https://www.gtreview.com/feed/",
    # Additional Sources
    "Ghana Business News": "https://www.ghanabusinessnews.com/feed/",
    "Business Daily Africa": "https://www.businessdailyafrica.com/feed",
    "Agri Orbit": "https://www.agriorbit.com/feed/",
    "Farmers Review Africa": "https://www.farmersreviewafrica.com/feed/",
    "Ventures Africa": "https://venturesafrica.com/feed/",
    "Africa Business Communities": "https://africabusinesscommunities.com/feed/",
    "The Exchange Africa": "https://theexchange.africa/feed/",
    "ESI Africa": "https://www.esi-africa.com/feed/"  # Energy & commodities coverage
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

# --------------------- Utils ---------------------
def _normalize(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_session() -> requests.Session:
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

@lru_cache(maxsize=100)
def fetch_article_text(url: str, timeout: int = 10) -> str:
    """Fetch and lightly clean article text with caching and retry mechanism."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; OneAfricaPulse/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
        }
        session = get_session()
        logger.info(f"Fetching article from: {url}")
        r = session.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "iframe", "form"]):
            tag.decompose()

        # Try to get article body with extended selectors
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
        if len(text) < 50:
            logger.warning(f"Short content from {url}: {len(text)} chars")
            return ""
        logger.info(f"Successfully fetched article from {url}: {len(text)} chars")
        return text

    except requests.RequestException as e:
        logger.error(f"Failed to fetch article from {url}: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"Error processing article from {url}: {str(e)}")
        return ""

def keyword_relevance(text: str, keywords: List[str]) -> float:
    """Compute a simple relevance score via TF-IDF; fallback to keyword hit ratio if sklearn absent."""
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
    # Fallback: simple hits ratio
    tokens = re.findall(r"[a-zA-Z']{3,}", text.lower())
    kwset = set(k.lower().strip() for k in keywords)
    hits = sum(1 for t in tokens if t in kwset)
    return hits / max(1, len(tokens))

def simple_extractive_summary(text: str, n_sentences: int = 3, keywords: Optional[List[str]] = None) -> str:
    """Extract top sentences by TF-IDF centroid (if sklearn), else fallback to the first N sentences."""
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
    """Parse date strings in various formats"""
    try:
        fmts = [
            "%Y-%m-%dT%H:%M:%S%z",  # ISO with tz
            "%Y-%m-%dT%H:%M:%SZ",   # ISO Z
            "%Y-%m-%d %H:%M:%S",
            "%a, %d %b %Y %H:%M:%S %z",  # RFC822
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%d",
            "%d %b %Y",
            "%B %d, %Y",
        ]
        for fmt in fmts:
            try:
                return dt.datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    except Exception:
        return None

# --------------------- Minimal RSS/Atom parser (no feedparser) ---------------------
ATOM_NS = "{http://www.w3.org/2005/Atom}"

def _text(elem: Optional[ET.Element]) -> str:
    return _normalize(elem.text if elem is not None and elem.text else "")

def _find(elem: ET.Element, tag: str) -> Optional[ET.Element]:
    # Try both RSS (no ns) and Atom namespace when relevant
    e = elem.find(tag)
    if e is not None:
        return e
    # Atom try
    if not tag.startswith("{"):
        e = elem.find(ATOM_NS + tag)
    return e

def _findall(elem: ET.Element, tag: str) -> List[ET.Element]:
    out = list(elem.findall(tag))
    out += list(elem.findall(ATOM_NS + tag))
    return out

def parse_feed_xml(content: bytes, base_url: str) -> List[Dict]:
    """
    Parse RSS/Atom bytes and return list of dicts: {title, link, summary, published}
    """
    items = []
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
                # ensure absolute links if relative
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

def validate_feed(url: str) -> Tuple[bool, str]:
    """Validate if a feed URL is accessible and returns entries (no feedparser)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"
        }
        session = get_session()
        r = session.get(url, headers=headers, timeout=15, verify=True, allow_redirects=True)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        items = parse_feed_xml(r.content, base_url=url)
        if not items:
            return False, "No entries found in feed"
        # Check recency in first 5 items
        now = dt.datetime.now(dt.timezone.utc)
        recent = False
        for it in items[:5]:
            d = parse_date(it.get("published_raw", "") or "")
            if d:
                if d.tzinfo:
                    d = d.astimezone(dt.timezone.utc)
                else:
                    d = d.replace(tzinfo=dt.timezone.utc)
                if (now - d).days <= 30:
                    recent = True
                    break
        if not recent:
            return False, "No recent entries found"
        return True, "OK"
    except requests.exceptions.SSLError:
        try:
            r = session.get(url, headers=headers, timeout=15, verify=False)
            items = parse_feed_xml(r.content, base_url=url)
            if items:
                return True, "OK (SSL verification disabled)"
        except Exception as e:
            return False, f"SSL Error: {str(e)}"
    except requests.exceptions.ConnectionError:
        return False, "Connection failed"
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except Exception as e:
        return False, str(e)

def fetch_from_feed(url: str, start_date: dt.datetime, end_date: dt.datetime) -> List[Dict]:
    """Fetch articles from feed within the specified date range (no feedparser)."""
    logger.info(f"Fetching feed from: {url}")
    is_valid, status = validate_feed(url)
    if not is_valid:
        logger.error(f"Feed validation failed for {url}: {status}")
        st.warning(f"⚠️ Failed to validate feed {urlparse(url).netloc}: {status}")
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; OneAfricaPulse/1.0)",
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml"
    }
    session = get_session()
    r = session.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    raw_items = parse_feed_xml(r.content, base_url=url)

    items = []
    for e in raw_items:
        title = _normalize(e.get("title", ""))
        link = e.get("link", "")
        summary = _normalize(e.get("summary", ""))
        published = None

        # Attempt to parse the date string
        if e.get("published_raw"):
            published = parse_date(e["published_raw"])

        # Filter by date range if we have a date
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
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .stButton > button { width: 100%; border-radius: 6px; height: 3em; font-weight: 600; }
    div.stMarkdown p { line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.title("🌍 One Africa Market Pulse")
    st.caption(TAGLINE)
with col2:
    theme = st.selectbox("Theme", ["Light", "Dark"], key="theme")
    if theme == "Dark":
        st.markdown("""
            <style>
            .main { background-color: #1e1e1e; color: #ffffff; }
            .sidebar { background-color: #262626; color: #ffffff; }
            </style>
        """, unsafe_allow_html=True)

st.markdown(f"<blockquote style='font-style: italic; margin: 1em 0;'>{QUOTE}</blockquote>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")

    with st.expander("📰 News Sources", expanded=True):
        st.markdown("Select the sources to scan for market intelligence:")

        # Feed status check
        if st.button("🔄 Check Feed Status", help="Test the availability of all news sources"):
            st.markdown("### Feed Status Check")
            for name, url in DEFAULT_SOURCES.items():
                ok, status = validate_feed(url)
                if ok:
                    st.success(f"✅ {name}: Available ({status})")
                else:
                    st.error(f"❌ {name}: {status}")
            st.markdown("---")

        chosen_sources = []
        categories = {"Major African Business News": [], "Agriculture & Food Business": [], "Trade & Markets": [], "Additional Sources": []}
        for name, url in DEFAULT_SOURCES.items():
            if name in ["AllAfrica Business", "AllAfrica Agriculture", "Africa News Agency", "African Arguments", "Africa Feeds"]:
                categories["Major African Business News"].append((name, url))
            elif name in ["FoodBusiness Africa", "AgFunderNews", "Fresh Plaza Africa", "Agri Orbit", "Farmers Review Africa"]:
                categories["Agriculture & Food Business"].append((name, url))
            elif name in ["Global Trade Review", "Business Daily Africa", "The Exchange Africa"]:
                categories["Trade & Markets"].append((name, url))
            else:
                categories["Additional Sources"].append((name, url))

        for category, sources in categories.items():
            st.markdown(f"#### {category}")
            for name, url in sources:
                if st.checkbox(name, value=True, help=f"Include articles from {name}"):
                    chosen_sources.append(url)
            st.markdown("---")

        if not chosen_sources:
            st.warning("⚠️ Please select at least one source")

    with st.expander("📅 Date Range", expanded=True):
        st.markdown("#### Select Time Period")
        date_option = st.radio("Date Selection Mode", options=["Quick Select", "Custom Range"], horizontal=True)
        if date_option == "Quick Select":
            quick_options = {
                "Last 24 Hours": 1, "Last 3 Days": 3, "Last Week": 7,
                "Last 2 Weeks": 14, "Last Month": 30, "Last 3 Months": 90
            }
            selected_quick = st.selectbox("Select time period", options=list(quick_options.keys()), index=2)
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
        st.markdown("#### Keywords")
        custom_kw = st.text_area("Add or edit keywords (comma-separated)", value=", ".join(DEFAULT_KEYWORDS), height=120)
        keywords = [k.strip() for k in custom_kw.split(",") if k.strip()] or DEFAULT_KEYWORDS

        st.markdown("#### Relevance Filter")
        min_relevance = st.slider("Minimum relevance score", 0.0, 1.0, 0.10, 0.01)

    with st.expander("📝 Content Settings", expanded=True):
        st.markdown("#### Summarization")
        n_sent = st.slider("Sentences per summary", 2, 6, 3)
        st.markdown("#### Digest Length")
        top_k = st.slider("Top items in digest", 5, 30, 12)

    st.markdown("---")
    run_col1, run_col2 = st.columns(2)
    with run_col1:
        run_btn = st.button("🚀 Scan Now", use_container_width=True)
    with run_col2:
        if st.button("♻️ Reset", use_container_width=True):
            st.rerun()

# --------------------- Processing ---------------------
def process_sources() -> List[Dict]:
    rows = []
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)

    def process_entry(entry: Dict, source_url: str) -> Optional[Dict]:
        try:
            text = fetch_article_text(entry["link"]) if entry["link"] else ""
            base = entry["summary"] or ""
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
            logger.error(f"Failed to process entry {entry.get('link','')}: {str(e)}")
            return None

    total_sources = len(chosen_sources)
    if total_sources == 0:
        return rows

    for idx, src in enumerate(chosen_sources, 1):
        try:
            progress_placeholder.text(f"Processing source {idx}/{total_sources}: {urlparse(src).netloc}")
            status_placeholder.info(f"Fetching feed from {urlparse(src).netloc}...")

            entries = fetch_from_feed(src, start_date=start_date, end_date=end_date)
            if not entries:
                status_placeholder.warning(f"No entries found in feed: {src}")
                progress_bar.progress(idx / total_sources)
                continue

            status_placeholder.info(f"Processing {len(entries)} articles from {urlparse(src).netloc}...")
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(process_entry, e, src): e for e in entries}
                completed = 0
                for fut in as_completed(futures):
                    completed += 1
                    progress = (idx - 1 + completed / max(1, len(entries))) / total_sources
                    progress_bar.progress(min(1.0, progress))
                    result = fut.result()
                    if result:
                        rows.append(result)

        except Exception as ex:
            logger.error(f"Failed to process source {src}: {str(ex)}")
            st.warning(f"Failed to process {urlparse(src).netloc}: {str(ex)}")

        progress_bar.progress(min(1.0, idx / total_sources))

    progress_placeholder.empty()
    status_placeholder.empty()
    progress_bar.empty()
    return rows

def main_app():
    if not run_btn:
        st.info("""
**What this demo does:**  
- 📰 Scans curated RSS/Atom feeds for the last *N* days  
- 📑 Fetches full article text where possible  
- 🎯 Scores relevance against your **commodity & policy keywords**  
- 📝 Auto-summarizes into 2–6 sentences  
- 🏷️ Tags each item with impact labels (Supply Risk, Price Upside, FX & Policy, Logistics, etc.)  
- 💾 Produces a **downloadable CSV** and **Daily Digest (Markdown)**
        """)
        st.markdown("This is a lightweight, API-free prototype — perfect for a live demo with One Africa Markets.")
        return

    if not chosen_sources:
        st.error("Please select at least one news source from the sidebar.")
        return

    with st.spinner("Scanning feeds, extracting content, and generating summaries..."):
        rows = process_sources()
        if not rows:
            st.warning("No relevant articles found matching your criteria.")
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

            st.info("💡 Tip: Paste the Markdown into an email, WhatsApp (as a code block), or your company wiki for quick sharing.")

if __name__ == "__main__":
    main_app()
