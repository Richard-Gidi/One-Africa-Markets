# OneAfrica Market Pulse — Automated Market Intelligence (Streamlit Demo)
# Run: streamlit run streamlit_app.py

import os
import re
import html
import hashlib
import datetime as dt
from typing import List, Dict, Tuple, Optional, Any
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

# ========================= Page / logging =========================
st.set_page_config(page_title="One Africa Market Pulse", page_icon="🌍",
                   layout="wide", initial_sidebar_state="expanded")
st.set_option('client.showErrorDetails', False)

logger = logging.getLogger("oneafrica.pulse")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ========================= .env & OpenAI =========================
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_OK = True
try:
    from openai import OpenAI  # pip install openai==1.*
except Exception:
    OPENAI_OK = False

# ========================= Optional sklearn =========================
HAS_SK = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    HAS_SK = False
    logger.info("sklearn not available; fallback keyword scoring enabled.")

# ========================= Constants / theme =========================
APP_NAME = "One Africa Market Pulse"
TAGLINE = "Automated intelligence for cashew, shea, cocoa & allied markets."
QUOTE = "“Ask your data why, until it has nothing else to say.” — Richard Gidi"
FALLBACK_IMG = "https://images.unsplash.com/photo-1519681393784-d120267933ba?q=80&w=1200&auto=format&fit=crop"

DEFAULT_KEYWORDS = [
    "cashew","shea","shea nut","cocoa","palm kernel","agri","export","harvest",
    "shipment","freight","logistics","port","tariff","ban","fx","currency",
    "cedi","naira","inflation","subsidy","cooperative","value-addition","processing",
    "ghana","nigeria","cote d’ivoire","ivory coast","benin","togo","burkina",
    "west africa","sahel","trade policy","commodity","price","market"
]

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
    "Supply Risk":[r"\bexport (?:ban|restriction|control)\b",r"\b(?:import|trade) (?:ban|restriction|control)\b",
                   r"\bembargo\b",r"\b(?:drought|flood|rainfall|weather)\b",r"\b(?:pest|disease|infestation)\b",
                   r"\b(?:shortage|scarcity)\b",r"\b(?:strike|protest|unrest)\b",r"\bport (?:closure|congestion|delay)\b",
                   r"\bharvest (?:delay|loss|damage)\b",r"\bproduction (?:issue|problem|concern)\b"],
    "Price Upside":[r"\bstrong (?:demand|buying|interest)\b",r"\b(?:surge|spike|jump|rise|increase)\b",
                    r"\b(?:grant|stimulus|support)\b",r"\b(?:incentive|subsidy|funding)\b",
                    r"\bvalue[- ](?:addition|chain|processing)\b",r"\bhigh(?:er)? (?:price|demand|consumption)\b",
                    r"\bmarket (?:rally|strength|upturn)\b",r"\bsupply (?:squeeze|shortage|tightness)\b",r"\bquality premium\b"],
    "Price Downside":[r"\b(?:oversupply|surplus|glut)\b",r"\b(?:decline|fall|drop|decrease|slump)\b",
                      r"\b(?:weak|soft|bearish) (?:price|market|demand)\b",r"\bcut (?:price|rate|cost)\b",
                      r"\blow(?:er)? (?:price|demand|consumption)\b",r"\bmarket (?:weakness|downturn)\b",r"\bcompetitive pressure\b"],
    "FX & Policy":[r"\b(?:depreciation|devaluation)\b",r"\bweak(?:ening)? (?:currency|exchange)\b",
                   r"\b(?:fx|forex|dollar|euro|yuan)\b",r"\b(?:monetary|fiscal|trade) policy\b",
                   r"\b(?:interest|exchange) rate\b",r"\b(?:tariff|duty|levy|tax)\b",r"\bregulatory (?:change|update|requirement)\b",
                   r"\bpolicy (?:change|update|reform)\b"],
    "Logistics & Trade":[r"\b(?:freight|shipping|transport)\b",r"\b(?:port|container|vessel|cargo)\b",
                         r"\b(?:congestion|delay|bottleneck)\b",r"\b(?:reroute|divert|alternative route)\b",
                         r"\b(?:cost|rate) (?:increase|surge|rise)\b",r"\btrade (?:flow|route|pattern)\b",
                         r"\b(?:export|import) (?:volume|data|figure)\b"],
    "Market Structure":[r"\b(?:merger|acquisition|takeover)\b",r"\b(?:investment|expansion|capacity)\b",
                        r"\b(?:processing|factory|facility)\b",r"\b(?:certification|standard|quality)\b",
                        r"\b(?:cooperative|association|group)\b",r"\b(?:contract|agreement|deal)\b",
                        r"\b(?:partnership|collaboration)\b",r"\bmarket (?:structure|reform|development)\b"],
    "Tech & Innovation":[r"\b(?:technology|innovation|digital)\b",r"\b(?:blockchain|traceability|tracking)\b",
                         r"\b(?:sustainability|sustainable)\b",r"\b(?:efficiency|optimization)\b",
                         r"\b(?:automation|mechanization)\b",r"\b(?:research|development|r&d)\b",
                         r"\b(?:startup|fintech|agtech)\b"],
}

CARD_CSS = """
<style>
.hero{position:relative;border-radius:16px;padding:28px;background:linear-gradient(135deg,#0ea5e9,#7c3aed 60%);color:#fff;box-shadow:0 14px 40px rgba(0,0,0,.18)}
.hero h1{margin:0 0 6px 0;font-size:28px;font-weight:800}
.hero p{margin:0;opacity:.95}
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;background:rgba(255,255,255,.15);color:#fff;font-weight:600;font-size:13px}
.card{background:#fff;border:1px solid rgba(0,0,0,.06);border-radius:14px;overflow:hidden;transition:transform .15s ease,box-shadow .15s ease}
.card:hover{transform:translateY(-3px);box-shadow:0 10px 24px rgba(0,0,0,.08)}
.thumb{width:100%;height:180px;object-fit:cover;background:#f6f7f9}
.card-body{padding:14px}
.card .title{color:#111827 !important;font-weight:800;font-size:18px;margin:6px 0 8px 0;line-height:1.25}
.card .meta{color:#6b7280 !important;font-size:12px;display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px}
.card .summary{color:#374151 !important;font-size:13px;line-height:1.55;margin-top:6px}
.badges{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.badge{font-size:11px;font-weight:700;padding:4px 8px;border-radius:999px;background:#eef2ff;color:#3730a3;border:1px solid #c7d2fe}
.link{text-decoration:none;font-weight:700;color:#2563eb !important}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

# ========================= Stable session model =========================
def boot_session():
    if "cfg" not in st.session_state:
        st.session_state.cfg = {
            "chosen_sources": list(DEFAULT_SOURCES.values()),
            "use_newsdata": True,
            "newsdata_key_override": "",
            "newsdata_query": "tree crop commodities",
            "nd_language": "", "nd_country": "", "nd_category": "", "nd_pages": 2,
            "date_mode": "Quick Select", "date_window": "Last Week",
            "start_date": dt.date.today() - dt.timedelta(days=7),
            "end_date": dt.date.today(),
            "keywords": DEFAULT_KEYWORDS[:],
            "min_relevance": 0.05,
            "per_source_cap": 20,
            "n_sent": 3,
            "top_k": 12,
            "force_fetch": True,
            "ignore_recency": True,
            "dedupe": True,
            "quick_url": "",
        }
    if "ai_analyses" not in st.session_state:
        st.session_state.ai_analyses = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system",
             "content": ("You are a crisp market-intelligence assistant for West African tree crops "
                         "(cashew, shea, cocoa, palm kernel). Be concise, cite assumptions, and "
                         "suggest actionable next steps.")}
        ]
boot_session()

cfg = st.session_state.cfg  # alias

# ========================= Secrets / HTTP helpers =========================
def _get_secret_safely(name: str) -> str:
    v = os.environ.get(name, "")
    if v: return v.strip().strip('"').strip("'")
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets.get(name, "")).strip().strip('"').strip("'")
    except Exception:
        pass
    return ""

def get_newsdata_api_key() -> str: return _get_secret_safely("NEWSDATA_API_KEY")
def get_openai_api_key() -> str:   return _get_secret_safely("OPENAI_API_KEY")

def get_session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=3, backoff_factor=0.6, status_forcelist=[429,500,502,503,504])
    s.mount("http://", HTTPAdapter(max_retries=r)); s.mount("https://", HTTPAdapter(max_retries=r))
    return s

def _normalize(t: str) -> str:
    t = html.unescape(t or ""); return re.sub(r"\s+", " ", t).strip()

SOFT_ERRORS: List[str] = []
def soft_fail(msg: str, detail: Optional[str] = None):
    if msg: SOFT_ERRORS.append(msg)
    if detail: logger.warning(detail)

# ========================= Content extraction =========================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_page(url: str, timeout: int = 12) -> str:
    try:
        h = {"User-Agent":"Mozilla/5.0 (compatible; OneAfricaPulse/1.0)",
             "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
             "Accept-Language":"en-US,en;q=0.5"}
        r = get_session().get(url, headers=h, timeout=timeout)
        if r.status_code != 200:
            soft_fail("Skipped a page that didn’t load cleanly.", f"fetch_page {url} -> {r.status_code}")
            return ""
        return r.text
    except Exception as e:
        soft_fail("Skipped one page due to connectivity.", f"fetch_page EXC {url}: {e}")
        return ""

def get_og_image(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    for tag, attrs in [("meta", {"property":"og:image"}), ("meta", {"name":"twitter:image"}),
                       ("meta", {"property":"twitter:image"}), ("link", {"rel":"image_src"})]:
        el = soup.find(tag, attrs=attrs)
        if el:
            src = el.get("content") or el.get("href")
            if src:
                if src.startswith("//"): return "https:" + src
                if src.startswith("/"):  return urljoin(base_url, src)
                return src
    return None

def get_favicon_url(domain_url: str) -> str:
    p = urlparse(domain_url); return f"{p.scheme}://{p.netloc}/favicon.ico"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_article_text_and_image(url: str) -> Tuple[str, str]:
    if not url: return "", FALLBACK_IMG
    html_text = fetch_page(url)
    if not html_text: return "", FALLBACK_IMG
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        for t in soup(["script","style","noscript","nav","footer","iframe","form"]): t.decompose()
        candidates, sels = [], [
            "article","[role='main']",".article-body",".story-body",".post-content","main",".content",".entry-content",
            "#article-body",".article-content",".story-content",".news-content",".page-content","body"
        ]
        for sel in sels:
            for node in soup.select(sel):
                txt = node.get_text(separator=" ", strip=True)
                if len(txt) > 100: candidates.append(txt)
        text = max(candidates, key=len) if candidates else soup.get_text(separator=" ", strip=True)
        text = _normalize(text); 
        if len(text) < 50: text = ""
        img = get_og_image(soup, url) or get_favicon_url(url) or FALLBACK_IMG
        return text, img
    except Exception as e:
        soft_fail("Used a fallback image for one article.", f"fetch_article_text_and_image EXC {url}: {e}")
        return "", FALLBACK_IMG

# ========================= Relevance / summary =========================
def keyword_relevance(text: str, keywords: List[str]) -> float:
    if not text: return 0.0
    if HAS_SK:
        try:
            v = TfidfVectorizer(stop_words="english", max_features=5000)
            X = v.fit_transform([text, " ".join(keywords)])
            return float(cosine_similarity(X[0:1], X[1:2])[0][0])
        except Exception as e:
            logger.info(f"tfidf fallback: {e}")
    tokens = re.findall(r"[a-zA-Z']{3,}", text.lower())
    kwset = {k.lower() for k in keywords}
    hits = sum(1 for t in tokens if t in kwset)
    return hits / max(1, len(tokens))

def simple_extractive_summary(text: str, n_sentences: int = 3, keywords: Optional[List[str]] = None) -> str:
    if not text: return ""
    sents = re.split(r"(?<=[\.\?\!])\s+", text)
    sents = [s for s in sents if 30 <= len(s) <= 400][:60]
    if len(sents) <= n_sentences: return " ".join(sents)
    if HAS_SK:
        try:
            v = TfidfVectorizer(stop_words="english", max_features=8000)
            X = v.fit_transform(sents); centroid = X.mean(axis=0)
            sims = cosine_similarity(X, centroid).ravel()
            if keywords:
                kw = [k.lower() for k in keywords]
                boost = np.array([sum(1 for w in re.findall(r"[a-z']+", s.lower()) if w in kw) for s in sents], float)
                sims = sims + 0.05 * boost
            idx = sims.argsort()[-n_sentences:][::-1]
            return " ".join([sents[i] for i in idx])
        except Exception as e:
            logger.info(f"summary fallback: {e}")
    return " ".join(sents[:n_sentences])

def classify_impact(text: str) -> List[str]:
    tags, lower = [], text.lower()
    for label, patterns in IMPACT_RULES.items():
        try:
            if any(re.search(p, lower) for p in patterns): tags.append(label)
        except Exception: pass
    return list(dict.fromkeys(tags)) or ["General"]

def parse_date(date_str: str) -> Optional[dt.datetime]:
    for fmt in ["%Y-%m-%dT%H:%M:%S%z","%Y-%m-%dT%H:%M:%SZ","%Y-%m-%d %H:%M:%S",
                "%a, %d %b %Y %H:%M:%S %z","%a, %d %b %Y %H:%M:%S %Z","%Y-%m-%d",
                "%d %b %Y","%B %d, %Y"]:
        try: return dt.datetime.strptime(date_str, fmt)
        except ValueError: continue
    return None

# ========================= RSS / Atom =========================
ATOM_NS = "{http://www.w3.org/2005/Atom}"
def _text(e: Optional[ET.Element]) -> str: return _normalize(e.text if e is not None and e.text else "")
def _find(e: ET.Element, tag: str) -> Optional[ET.Element]:
    x = e.find(tag); 
    if x is not None: return x
    if not tag.startswith("{"): x = e.find(ATOM_NS + tag)
    return x
def _findall(e: ET.Element, tag: str) -> List[ET.Element]:
    return list(e.findall(tag)) + list(e.findall(ATOM_NS + tag))

@st.cache_data(ttl=600, show_spinner=False)
def fetch_feed_raw(url: str, timeout: int = 20) -> bytes:
    try:
        h = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 OneAfricaPulse/1.0",
             "Accept":"application/rss+xml, application/atom+xml, application/xml, text/xml, */*"}
        r = get_session().get(url, headers=h, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            soft_fail("Skipped a source that returned a non-200 response.", f"fetch_feed_raw {url} -> {r.status_code}")
        return r.content if r.status_code == 200 else (r.content or b"")
    except Exception as e:
        soft_fail("Temporarily skipped one source due to connectivity.", f"fetch_feed_raw EXC {e}")
        return b""

def parse_feed_xml(content: bytes, base_url: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not content: return items
    try:
        root = ET.fromstring(content)
        channel = root.find("channel")
        if channel is not None:  # RSS
            for it in channel.findall("item"):
                title = _text(_find(it,"title")) or "(untitled)"
                link = _text(_find(it,"link"))
                if not link:
                    link = _text(_find(it,"guid"))
                if link and link.startswith("/"): link = urljoin(base_url, link)
                summary = _text(_find(it,"description"))
                pub = _text(_find(it,"pubDate"))
                if title or link: items.append({"title":title,"link":link,"summary":summary,"published_raw":pub})
            return items
        for entry in _findall(root,"entry"):  # Atom
            title = _text(_find(entry,"title")) or "(untitled)"
            link_el = _find(entry,"link"); link = ""
            if link_el is not None:
                link = link_el.attrib.get("href","") or _text(link_el)
            if link and link.startswith("/"): link = urljoin(base_url, link)
            summary = _text(_find(entry,"summary")) or _text(_find(entry,"content"))
            pub = _text(_find(entry,"updated")) or _text(_find(entry,"published"))
            if title or link: items.append({"title":title,"link":link,"summary":summary,"published_raw":pub})
        return items
    except Exception as e:
        soft_fail("Skipped one feed that had invalid XML.", f"parse_feed_xml EXC {e}")
        return items

def validate_feed(url: str, ignore_recency_check: bool = False) -> Tuple[bool, str]:
    try:
        content = fetch_feed_raw(url); items = parse_feed_xml(content, base_url=url)
        if not items: return False, "No entries found"
        if ignore_recency_check: return True, "OK"
        now = dt.datetime.now(dt.timezone.utc)
        for it in items[:10]:
            d = parse_date(it.get("published_raw","") or "")
            if d:
                d = d.astimezone(dt.timezone.utc) if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
                if (now - d).days <= 60: return True, "OK"
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
    raw = fetch_feed_raw(url); raw_items = parse_feed_xml(raw, base_url=url)
    items: List[Dict[str, Any]] = []
    for e in raw_items:
        title = _normalize(e.get("title","")); link = e.get("link","")
        summary = _normalize(e.get("summary",""))
        published = None
        if e.get("published_raw"):
            published = parse_date(e["published_raw"])
        if published:
            published = published.astimezone(dt.timezone.utc) if published.tzinfo else published.replace(tzinfo=dt.timezone.utc)
            if not (start_date <= published <= end_date): continue
            published_str = published.strftime("%Y-%m-%d %H:%M UTC")
        else:
            published_str = "Date unknown"
        items.append({"source":urlparse(url).netloc,"title":title,"link":link,
                      "published":published_str,"summary":summary})
    return items

# ========================= Newsdata.io =========================
NEWSDATA_BASE = "https://newsdata.io/api/1/latest"

@st.cache_data(ttl=600, show_spinner=False)
def _newsdata_cached(redacted_params: Dict[str, Any], max_pages: int) -> List[Dict[str, Any]]:
    return []

def _newsdata_runtime(api_key: str, base_params: Dict[str, Any], max_pages: int) -> List[Dict[str, Any]]:
    s = get_session(); items: List[Dict[str, Any]] = []; pages = 0; next_page = None
    params = dict(base_params); params["apikey"] = api_key
    while pages < max_pages:
        try:
            q = dict(params); 
            if next_page: q["page"] = next_page
            r = s.get(NEWSDATA_BASE, params=q, timeout=20)
            if r.status_code != 200:
                soft_fail("One API page was skipped (non-200).", f"newsdata {r.status_code} {r.text[:200]}"); break
            data = r.json()
            results = data.get("results") or data.get("articles") or []
            for a in results: items.append(a)
            next_page = data.get("nextPage") or data.get("next_page")
            pages += 1
            if not next_page: break
        except Exception as e:
            soft_fail("Temporarily skipped an API page due to connectivity.", f"newsdata EXC {e}")
            break
    return items

def fetch_from_newsdata(api_key: str, query: str, start_date: dt.datetime, end_date: dt.datetime,
                        language: Optional[str]=None, country: Optional[str]=None,
                        category: Optional[str]=None, max_pages: int=2) -> List[Dict[str, Any]]:
    if not api_key: return []
    redacted = {"q": query or ""}
    if language: redacted["language"] = language
    if country: redacted["country"] = country
    if category: redacted["category"] = category
    _ = _newsdata_cached(redacted, max_pages=max_pages)
    raw = _newsdata_runtime(api_key=api_key, base_params=redacted, max_pages=max_pages)
    items: List[Dict[str, Any]] = []
    for a in raw:
        try:
            title = _normalize(a.get("title",""))
            link = a.get("link") or a.get("url") or ""
            source = a.get("source_id") or a.get("source") or "newsdata.io"
            pub = a.get("pubDate") or a.get("published_at") or ""
            desc = _normalize(a.get("description","")) or _normalize(a.get("content",""))
            ok_date = True; published_str = "Date unknown"
            if pub:
                d = parse_date(pub)
                if d:
                    d = d.astimezone(dt.timezone.utc) if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
                    ok_date = start_date <= d <= end_date
                    published_str = d.strftime("%Y-%m-%d %H:%M UTC")
            if not ok_date: continue
            items.append({"source":f"{source} (newsdata.io)","title":title or "(untitled)",
                          "link":link,"published":published_str,"summary":desc})
        except Exception as e:
            soft_fail("Skipped one API item due to missing fields.", f"newsdata item EXC {e}")
            continue
    return items

# ========================= OpenAI helpers =========================
def have_openai() -> bool:
    return OPENAI_OK and bool(get_openai_api_key())

def get_openai_client():
    try:
        api_key = get_openai_api_key()
        if not api_key: return None
        base_url = os.environ.get("OPENAI_BASE_URL","").strip()
        return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    except Exception as e:
        logger.warning(f"OpenAI client init failed: {e}")
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def _llm_analyze_article_cached(model: str, title: str, body: str, tags: List[str]) -> str:
    client = get_openai_client()
    if client is None: return ""
    prompt = f"""
You are a market-intelligence analyst for West African agri value chains.

Analyze the ARTICLE and return these sections:
1) WHAT THE ARTICLE MEANS (2–3 sentences)
2) KEY INSIGHTS (3–6 bullets)
3) MARKET IMPACT
4) BUSINESS OPPORTUNITIES (3–6 bullets)
5) RISK FACTORS (3–5 bullets)
6) ACTIONABLE RECOMMENDATIONS (3–5 steps)
7) TIME HORIZON (0–3m / 3–12m / 12m+)
8) CONFIDENCE (High/Medium/Low + why)

TITLE:
{title[:400]}

BODY (may be partial):
{body[:7000]}

HEURISTIC TAGS: {", ".join(tags) if tags else "General"}

Constraints: Be pragmatic, specific to West Africa, note assumptions/uncertainty explicitly.
"""
    resp = client.chat.completions.create(
        model=model, temperature=0.3,
        messages=[{"role":"system","content":"Be precise and action oriented."},
                  {"role":"user","content":prompt}]
    )
    return (resp.choices[0].message.content or "").strip()

def analyze_with_llm(title: str, body: str, tags: List[str]) -> str:
    if not have_openai(): return ""
    for m in ["gpt-4o-mini","gpt-4o","gpt-4.1-mini","gpt-3.5-turbo-0125"]:
        try:
            out = _llm_analyze_article_cached(m, title, body, tags)
            if out: return out
        except Exception as e:
            logger.warning(f"LLM analyze failed on {m}: {e}")
    return ""

# ========================= Helpers =========================
def hash_key(*parts) -> str:
    return hashlib.md5(("||".join([p or "" for p in parts])).encode("utf-8")).hexdigest()

def process_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["source","published","title","relevance","impact","auto_summary","link","image"])
    seen = set(); cleaned = []
    for r in rows:
        key = hash_key(r.get("title",""), r.get("link",""))
        if key in seen: continue
        seen.add(key); cleaned.append(r)
    df = pd.DataFrame(cleaned)
    if df.empty:
        return pd.DataFrame(columns=["source","published","title","relevance","impact","auto_summary","link","image"])
    return df.sort_values("relevance", ascending=False).reset_index(drop=True)

def enrich(entry: Dict[str, Any], keywords: List[str], n_sent: int, min_relevance: float) -> Optional[Dict[str, Any]]:
    try:
        article_text, image_url = fetch_article_text_and_image(entry.get("link",""))
        base = entry.get("summary") or ""
        body = article_text if len(article_text) > len(base) else base
        rel = keyword_relevance(" ".join([entry.get("title",""), body]), keywords)
        if rel < min_relevance: return None
        summary = simple_extractive_summary(body, n_sentences=n_sent, keywords=keywords)
        impacts = classify_impact(" ".join([entry.get("title",""), body])) or ["General"]
        return {"source": entry.get("source",""),
                "title": entry.get("title","(untitled)"),
                "link": entry.get("link",""),
                "published": entry.get("published","Date unknown"),
                "relevance": float(rel),
                "impact": impacts,
                "auto_summary": summary,
                "image": image_url or FALLBACK_IMG}
    except Exception as e:
        soft_fail("Skipped one article that couldn’t be processed.", f"enrich EXC {e}")
        return None

def fetch_all(chosen_sources: List[str], start_date: dt.datetime, end_date: dt.datetime,
              force_fetch: bool, ignore_recency: bool, per_source_cap: int,
              use_newsdata: bool, newsdata_key: str, newsdata_query: str,
              nd_language: str, nd_country: str, nd_category: str, nd_pages: int,
              keywords: List[str], n_sent: int, min_relevance: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total_tasks = len(chosen_sources) + (1 if (use_newsdata and newsdata_key) else 0)
    total_tasks = max(total_tasks, 1)
    progress = st.progress(0.0); info = st.empty()

    # RSS
    for i, src in enumerate(chosen_sources, start=1):
        info.info(f"Fetching RSS {i}/{total_tasks}: {urlparse(src).netloc}")
        try:
            raw_items = fetch_from_feed(src, start_date, end_date, force_fetch, ignore_recency)
        except Exception as e:
            soft_fail("Skipped a source due to a transient issue.", f"fetch_from_feed EXC {src}: {e}")
            raw_items = []
        if per_source_cap and raw_items: raw_items = raw_items[:per_source_cap]
        if raw_items:
            with ThreadPoolExecutor(max_workers=6) as ex:
                futures = [ex.submit(enrich, {**e, "source": urlparse(src).netloc}, keywords, n_sent, min_relevance)
                           for e in raw_items]
                for fut in as_completed(futures):
                    try:
                        r = fut.result()
                        if r: rows.append(r)
                    except Exception as e:
                        soft_fail("One article was skipped during processing.", f"future enrich EXC {e}")
        progress.progress(min(1.0, i/total_tasks))

    # Newsdata
    if use_newsdata and newsdata_key:
        info.info(f"Fetching Newsdata.io {len(chosen_sources)+1}/{total_tasks}")
        try:
            nd_items = fetch_from_newsdata(
                api_key=newsdata_key, query=newsdata_query,
                start_date=start_date, end_date=end_date,
                language=nd_language or None, country=nd_country or None,
                category=nd_category or None, max_pages=int(nd_pages),
            )
            if per_source_cap and nd_items: nd_items = nd_items[:per_source_cap]
            if nd_items:
                with ThreadPoolExecutor(max_workers=6) as ex:
                    futures = [ex.submit(enrich, it, keywords, n_sent, min_relevance) for it in nd_items]
                    for fut in as_completed(futures):
                        try:
                            r = fut.result()
                            if r: rows.append(r)
                        except Exception as e:
                            soft_fail("One API article was skipped during processing.", f"future API enrich EXC {e}")
        except Exception as e:
            soft_fail("The API was briefly unavailable; results shown are from RSS.", f"fetch_from_newsdata EXC {e}")
        progress.progress(1.0)

    info.empty(); progress.empty()
    return rows

def make_digest(df: pd.DataFrame, top_k: int = 12) -> str:
    header = f"# {APP_NAME} — Daily Digest\n\n*{TAGLINE}*\n\n> {QUOTE}\n\n"
    if df.empty: return header + "_No relevant items found for the selected period._"
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

# ========================= HERO & top bar =========================
st.markdown(f"""
<div class="hero">
  <div class="pill">🌍 One Africa Market Pulse</div>
  <h1>{TAGLINE}</h1>
  <p>{QUOTE}</p>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

c_run, c_reset = st.columns([1,1])
with c_run:
    run_btn = st.button("🚀 Scan Now", use_container_width=True, key="run_main")
with c_reset:
    if st.button("♻️ Reset", use_container_width=True, key="reset_main"):
        st.session_state.clear()
        st.rerun()

# ========================= Quick Analyze by URL =========================
st.markdown("### 🔗 Quick Analyze by URL (LLM)")
qa1, qa2 = st.columns([4,1])
with qa1:
    cfg["quick_url"] = st.text_input("Paste any article URL", value=cfg["quick_url"],
                                     placeholder="https://example.com/article", key="quick_url_input")
with qa2:
    run_quick = st.button("Analyze", use_container_width=True, key="an_quick")

if run_quick:
    if not have_openai():
        st.warning("Add an `OPENAI_API_KEY` to use AI analysis.")
    elif not cfg["quick_url"]:
        st.info("Please paste a valid URL.")
    else:
        with st.spinner("Fetching and analyzing..."):
            text, img = fetch_article_text_and_image(cfg["quick_url"])
            if not text:
                st.error("Could not extract article text from this URL.")
            else:
                title_guess = text.split(".")[0][:140] if text else cfg["quick_url"]
                tags = classify_impact(text)
                md = analyze_with_llm(title_guess, text, tags)
                if not md:
                    st.error("AI analysis failed. Please try again.")
                else:
                    st.image(img, use_column_width=True)
                    st.markdown(f"**Source:** {urlparse(cfg['quick_url']).netloc}")
                    st.markdown(md)

# ========================= Chat Assistant =========================
st.markdown("### 🤖 Chat Assistant")
for m in st.session_state.chat_history:
    if m["role"] == "system": continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type your message...")
if prompt:
    st.session_state.chat_history.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)
    if not have_openai():
        with st.chat_message("assistant"):
            st.warning("No `OPENAI_API_KEY` found. Add it and click **Reset**.")
    else:
        client = get_openai_client()
        if client:
            try:
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.chat_history,
                    stream=True, temperature=0.4,
                )
                with st.chat_message("assistant"):
                    ph = st.empty(); buf = ""
                    for ch in stream:
                        delta = ch.choices[0].delta.content or ""
                        if delta: buf += delta; ph.markdown(buf)
                if buf:
                    st.session_state.chat_history.append({"role":"assistant","content":buf})
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"Assistant error: {e}")

# ========================= Sidebar: Config (persisted) =========================
with st.sidebar:
    with st.expander("⚙️ Configurations", expanded=False):
        st.header("Settings")

        # Sources
        st.subheader("📰 RSS/Atom Sources")
        chosen = []
        for name, url in DEFAULT_SOURCES.items():
            default_on = url in cfg["chosen_sources"]
            if st.checkbox(name, value=default_on, key=f"src_{name}"):
                chosen.append(url)
        cfg["chosen_sources"] = chosen

        if st.button("🔄 Check Feeds", key="check_feeds"):
            for name, url in DEFAULT_SOURCES.items():
                ok, status = validate_feed(url, ignore_recency_check=True)
                st.write(f"{'✅' if ok else '❌'} {name}: {status}")

        st.markdown("---")

        # Newsdata
        st.subheader("🧩 Newsdata.io (optional)")
        cfg["use_newsdata"] = st.checkbox("Use Newsdata.io", value=cfg["use_newsdata"], key="use_nd")
        auto_key = get_newsdata_api_key()
        override = st.checkbox("Temporarily override API key (not saved)", value=bool(cfg["newsdata_key_override"]), key="nd_override")
        cfg["newsdata_key_override"] = st.text_input("Enter API key", type="password",
                                                     value=cfg["newsdata_key_override"], key="nd_key_input") if override else ""
        newsdata_key = (cfg["newsdata_key_override"] or auto_key).strip()
        if cfg["use_newsdata"]:
            if newsdata_key: st.success("Using secured API key.")
            else: st.warning("No API key found. Add NEWSDATA_API_KEY or use a temporary override.")

        cfg["newsdata_query"] = st.text_input("Query", value=cfg["newsdata_query"], key="nd_query")
        c1, c2, c3 = st.columns(3)
        with c1: cfg["nd_language"] = st.text_input("Language (e.g., en, fr)", value=cfg["nd_language"], key="nd_lang")
        with c2: cfg["nd_country"] = st.text_input("Country (e.g., gh, ng, ci)", value=cfg["nd_country"], key="nd_cty")
        with c3: cfg["nd_category"] = st.text_input("Category (e.g., business)", value=cfg["nd_category"], key="nd_cat")
        cfg["nd_pages"] = st.number_input("Newsdata pages", min_value=1, max_value=10, value=int(cfg["nd_pages"]), step=1, key="nd_pages")

        st.markdown("---")

        # Date range
        st.subheader("📅 Date Range")
        cfg["date_mode"] = st.radio("Mode", ["Quick Select","Custom"], horizontal=True,
                                    index=0 if cfg["date_mode"]=="Quick Select" else 1, key="date_mode")
        if cfg["date_mode"] == "Quick Select":
            quick = {"Last 24 Hours":1,"Last 3 Days":3,"Last Week":7,"Last 2 Weeks":14,"Last Month":30}
            idx = list(quick.keys()).index(cfg.get("date_window","Last Week")) if cfg.get("date_window") in quick else 2
            cfg["date_window"] = st.selectbox("Window", list(quick.keys()), index=idx, key="date_win")
            end_date = dt.datetime.now(dt.timezone.utc)
            start_date = end_date - dt.timedelta(days=quick[cfg["date_window"]])
        else:
            d1, d2 = st.columns(2)
            with d1: cfg["start_date"] = st.date_input("Start", value=cfg["start_date"], key="start_date")
            with d2: cfg["end_date"] = st.date_input("End", value=cfg["end_date"], key="end_date")
            start_date = dt.datetime.combine(cfg["start_date"], dt.time.min, tzinfo=dt.timezone.utc)
            end_date = dt.datetime.combine(cfg["end_date"], dt.time.max, tzinfo=dt.timezone.utc)

        st.markdown("---")

        # Keywords & filters
        st.subheader("🔍 Keywords & Filters")
        kw_text = st.text_area("Keywords (comma-separated)", ", ".join(cfg["keywords"]), height=100, key="kw_text")
        cfg["keywords"] = [k.strip() for k in kw_text.split(",") if k.strip()]
        cfg["min_relevance"] = st.number_input("Min relevance (0.00–1.00)", min_value=0.0, max_value=1.0,
                                               value=float(cfg["min_relevance"]), step=0.01, format="%.2f", key="min_rel")
        cfg["per_source_cap"] = st.number_input("Max articles per source", min_value=1, max_value=200,
                                                value=int(cfg["per_source_cap"]), step=1, key="cap")

        st.markdown("---")

        # Content settings
        st.subheader("📝 Content Settings")
        cfg["n_sent"] = st.number_input("Sentences per summary", min_value=2, max_value=10,
                                        value=int(cfg["n_sent"]), step=1, key="n_sent")
        cfg["top_k"] = st.number_input("Digest: top items", min_value=5, max_value=100,
                                       value=int(cfg["top_k"]), step=1, key="top_k")

        st.markdown("---")

        # Resilience
        st.subheader("🛡️ Resilience")
        cfg["force_fetch"] = st.checkbox("⚡ Force RSS fetch if validation fails", value=cfg["force_fetch"], key="force")
        cfg["ignore_recency"] = st.checkbox("🕒 Ignore RSS recency check", value=cfg["ignore_recency"], key="ignore_recent")
        cfg["dedupe"] = st.checkbox("🧹 Deduplicate across sources", value=cfg["dedupe"], key="dedupe")

# ========================= Cards / analysis =========================
def render_card(row: pd.Series):
    key = f"card_{hash_key(row['title'], row['link'])}"
    pub, src = row["published"], row["source"]
    rel = f"{row['relevance']:.0%}"
    title, link = row["title"], row["link"]
    img = row["image"] or FALLBACK_IMG
    summary = row["auto_summary"] or ""
    tags = row["impact"] or ["General"]

    with st.container():
        st.markdown(f"""
        <div class="card">
          <img class="thumb" src="{img}" alt="thumbnail">
          <div class="card-body">
            <div class="meta">{src} · {pub} · Relevance {rel}</div>
            <div class="title">{title}</div>
            <div class="badges">
                {"".join([f'<span class="badge">{t}</span>' for t in tags])}
            </div>
            <div class="summary">{summary}</div>
            <div style="margin-top:10px;">
              <a class="link" href="{link}" target="_blank">Read full article →</a>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔎 Analyze with AI", expanded=False):
            if not have_openai():
                st.warning("Add an `OPENAI_API_KEY` to run AI analysis.")
                st.stop()

            prev = st.session_state.ai_analyses.get(key)
            if prev: st.markdown(prev)

            if st.button("Run LLM Analysis", key=f"btn_{key}"):
                with st.spinner("Analyzing article with AI..."):
                    full_text, _ = fetch_article_text_and_image(link)
                    body = full_text if len(full_text) > len(summary) else summary
                    if not body:
                        st.error("Could not extract article text to analyze.")
                    else:
                        md = analyze_with_llm(title, body, tags)
                        if not md:
                            st.error("AI analysis failed. Try again.")
                        else:
                            st.session_state.ai_analyses[key] = md
                            st.markdown(md)

def ui_results(df: pd.DataFrame, top_k: int):
    st.subheader("📊 Results")
    if df.empty:
        st.warning("No relevant articles found.")
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

    records = list(filtered.to_dict("records"))
    n = 3
    for i in range(0, len(records), n):
        cols = st.columns(n)
        for j, col in enumerate(cols):
            if i + j < len(records):
                with col:
                    render_card(pd.Series(records[i + j]))
        st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("📝 Daily Digest")
    digest_md = make_digest(filtered if (impact_filter or source_filter) else df, top_k=top_k)
    st.markdown(digest_md)

    st.subheader("⬇️ Downloads")
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    export_df = filtered if (impact_filter or source_filter) else df
    st.download_button("📥 Download CSV", data=export_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"oneafrica_pulse_{ts}.csv", mime="text/csv")
    st.download_button("📥 Download Digest (Markdown)", data=digest_md.encode("utf-8"),
                       file_name=f"oneafrica_pulse_digest_{ts}.md", mime="text/markdown")

def friendly_error_summary():
    if not SOFT_ERRORS: return
    counts: Dict[str,int] = {}
    for m in SOFT_ERRORS: counts[m] = counts.get(m, 0) + 1
    bullets = "".join([f"- {msg} _(x{n})_\n" for msg, n in counts.items()])
    st.info(f"**Heads up:** Some sources were skipped.\n\n{bullets}")

# ========================= Main =========================
if not run_btn:
    st.info("""**What this demo does:**
- Scans curated RSS/Atom feeds (+ optional Newsdata.io)
- Fetches full article text & thumbnails
- Scores relevance vs your keywords
- Auto-summarizes + impact tags
- Downloadable CSV & Markdown digest
""")
else:
    try:
        newsdata_key = (cfg["newsdata_key_override"] or get_newsdata_api_key()).strip()
        if cfg["date_mode"] == "Quick Select":
            quick_days = {"Last 24 Hours":1,"Last 3 Days":3,"Last Week":7,"Last 2 Weeks":14,"Last Month":30}[cfg["date_window"]]
            end_date = dt.datetime.now(dt.timezone.utc)
            start_date = end_date - dt.timedelta(days=quick_days)
        else:
            start_date = dt.datetime.combine(cfg["start_date"], dt.time.min, tzinfo=dt.timezone.utc)
            end_date = dt.datetime.combine(cfg["end_date"], dt.time.max, tzinfo=dt.timezone.utc)

        if not cfg["chosen_sources"] and not (cfg["use_newsdata"] and newsdata_key):
            st.error("Pick at least one RSS source or enable Newsdata.io in Configurations.")
        else:
            with st.spinner("Scanning sources, extracting content, and generating summaries..."):
                rows = fetch_all(
                    chosen_sources=cfg["chosen_sources"],
                    start_date=start_date, end_date=end_date,
                    force_fetch=cfg["force_fetch"], ignore_recency=cfg["ignore_recency"],
                    per_source_cap=int(cfg["per_source_cap"]),
                    use_newsdata=cfg["use_newsdata"], newsdata_key=newsdata_key,
                    newsdata_query=cfg["newsdata_query"],
                    nd_language=cfg["nd_language"], nd_country=cfg["nd_country"],
                    nd_category=cfg["nd_category"], nd_pages=int(cfg["nd_pages"]),
                    keywords=cfg["keywords"], n_sent=int(cfg["n_sent"]),
                    min_relevance=float(cfg["min_relevance"]),
                )
                df = process_rows(rows)
                ui_results(df, top_k=int(cfg["top_k"]))
    except Exception as e:
        soft_fail("Something went wrong assembling results.", f"MAIN EXC {e}")
        st.error("We hit a hiccup. Please try again.")
    finally:
        friendly_error_summary()
