# Africa Commodity Pulse — Cashew • Shea • Cocoa • Gold • Oil
# Run: streamlit run streamlit_app.py

import os, re, html, hashlib, datetime as dt
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

# ---------- .env (optional) ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Optional sklearn ----------
HAS_SK = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    HAS_SK = False

# ---------- Streamlit safety ----------
st.set_option("client.showErrorDetails", False)

# ---------- Logging (server console only) ----------
logger = logging.getLogger("africa.pulse")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ---------- App strings ----------
APP_NAME = "Africa Commodity Pulse"
TAGLINE = "Africa-focused market intelligence for cashew, shea, cocoa, gold & oil."
QUOTE = "“Ask your data why, until it has nothing else to say.” — Richard Gidi"
FALLBACK_IMG = "https://images.unsplash.com/photo-1519681393784-d120267933ba?q=80&w=1200&auto=format&fit=crop"

# ---------- Hard interest filters ----------
TARGET_TERMS = [
    "cashew", "shea", "cocoa", "gold", "oil", "crude", "brent", "wti",
    "palm", "palm oil"  # (keep palm if you want broader edible oils)
]
# Africa region signals (countries + 'Africa')
AF_COUNTRIES = [
    "ghana","nigeria","côte d’ivoire","cote d’ivoire","ivory coast","benin","togo",
    "burkina","burkina faso","senegal","gambia","sierra leone","liberia","guinea",
    "mali","niger","cameroon","tanzania","kenya","uganda","rwanda","burundi",
    "ethiopia","sudan","south sudan","angola","mozambique","zambia","zimbabwe",
    "namibia","botswana","south africa","lesotho","eswatini","swaziland","somalia",
    "eritrea","djibouti","comoros","seychelles","mauritius","madagascar","algeria",
    "tunisia","morocco","libya","egypt","mauritania","cape verde","cabo verde","africa"
]
AFRICAN_HINTS = set(AF_COUNTRIES)

# ---------- Default keywords for scoring (soft boost) ----------
DEFAULT_KEYWORDS = list(dict.fromkeys(TARGET_TERMS + [
    "export","import","tariff","shipment","freight","port","logistics",
    "price","premium","discount","supply","demand","harvest","processing",
    "cooperative","traceability","fx","currency","naira","cedi","rand",
    "policy","subsidy","ban","embargo","strike","congestion"
]))

# ---------- Feeds (Africa + commodity heavy) ----------
# Keep to feeds that typically work without paywalls/403.
DEFAULT_SOURCES = {
    # Pan-Africa & regional business/agri
    "AllAfrica • Agriculture": "https://allafrica.com/tools/headlines/rdf/agriculture/headlines.rdf",
    "AllAfrica • Business":    "https://allafrica.com/tools/headlines/rdf/business/headlines.rdf",
    "FreshPlaza Africa":       "https://www.freshplaza.com/africa/rss.xml",
    "CitiNewsroom (Ghana)":    "https://citinewsroom.com/feed/",
    "Ghana Business News":     "https://www.ghanabusinessnews.com/feed/",
    "Business Day (Nigeria)":  "https://businessday.ng/feed/",
    "Daily Monitor (Uganda)":  "https://www.monitor.co.ug/rss.xml",
    "The EastAfrican":         "https://www.theeastafrican.co.ke/tea/rss.xml",
    "How We Made It In Africa":"https://www.howwemadeitinafrica.com/feed/",
    "Bizcommunity (Africa Agri)":"https://www.bizcommunity.com/GenerateRss.aspx?i=63&c=81",

    # Cocoa & West Africa supply chain often covered here:
    "Reuters Commodities":     "https://feeds.reuters.com/news/commodities",
    "FAO News":                "https://www.fao.org/news/rss/en",
    "WTO News":                "https://www.wto.org/english/news_e/news_e.rss",

    # Energy / Oil (Africa & global context)
    "Maritime Executive":      "https://www.maritime-executive.com/rss",
    "gCaptain":                "https://gcaptain.com/feed/",
    "The Loadstar":            "https://theloadstar.com/feed/",
}

# ---------- CSS ----------
st.markdown("""
<style>
.hero {
  border-radius: 16px; padding: 24px;
  background: linear-gradient(135deg, #047857, #0ea5e9 70%);
  color: white; box-shadow: 0 14px 40px rgba(0,0,0,0.18);
}
.hero h1 { margin: 0 0 6px 0; font-size: 26px; font-weight: 800; }
.hero p { margin: 0; opacity: .95; }

.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 16px; }
.card { background:#fff; border:1px solid rgba(0,0,0,.06); border-radius:14px; overflow:hidden; transition:transform .15s, box-shadow .15s; }
.card:hover { transform: translateY(-3px); box-shadow:0 10px 24px rgba(0,0,0,.08); }
.thumb { width:100%; height:180px; object-fit:cover; background:#f6f7f9; }
.card-body { padding:14px; }
.title { color:#111827 !important; font-weight:800; font-size:18px; margin:6px 0 8px; line-height:1.25; }
.meta { color:#6b7280 !important; font-size:12px; display:flex; gap:10px; flex-wrap:wrap; margin-bottom:8px; }
.summary { color:#374151 !important; font-size:13px; line-height:1.55; margin-top:6px; }
.badges { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }
.badge { font-size:11px; font-weight:700; padding:4px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe; }
.link { text-decoration:none; font-weight:700; color:#2563eb !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Secrets helpers ----------
def _secret(name: str) -> str:
    v = os.environ.get(name, "")
    if v: return v.strip().strip('"').strip("'")
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets.get(name, "")).strip().strip('"').strip("'")
    except Exception:
        pass
    return ""

def get_newsdata_api_key() -> str:
    return _secret("NEWSDATA_API_KEY")

# ---------- HTTP utils ----------
def get_session() -> requests.Session:
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])))
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])))
    return s

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()

SOFT_ERRORS: List[str] = []
def soft_fail(msg: str, detail: Optional[str] = None):
    if msg: SOFT_ERRORS.append(msg)
    if detail: logger.warning(detail)

# ---------- Minimal impact tags ----------
IMPACT_RULES = {
    "Supply Risk": [r"\bban\b", r"\bembargo\b", r"\bstrike\b", r"\bshortage\b", r"\bcongestion\b", r"\bharvest\b"],
    "Price Upside": [r"\brise\b", r"\bspike\b", r"\btightness\b", r"\bpremium\b"],
    "Price Downside": [r"\bdrop\b", r"\bdecline\b", r"\bglut\b", r"\boversupply\b"],
    "FX & Policy": [r"\bdevaluation\b", r"\bfx\b", r"\bpolicy\b", r"\btariff\b"],
    "Logistics & Trade": [r"\bfreight\b", r"\bshipping\b", r"\bport\b", r"\bcontainer\b"],
}

def classify_impact(text: str) -> List[str]:
    tags = []
    lower = text.lower()
    for label, pats in IMPACT_RULES.items():
        if any(re.search(p, lower) for p in pats):
            tags.append(label)
    return tags or ["General"]

# ---------- Fast relevance ----------
def hard_hit(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(re.search(rf"\b{re.escape(k.lower())}\b", t) for k in terms)

def africa_hit(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in AFRICAN_HINTS)

def keyword_relevance(text: str, keywords: List[str]) -> float:
    if not text: return 0.0
    if HAS_SK:
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=3000)
            X = vec.fit_transform([text, " ".join(keywords)])
            return float(cosine_similarity(X[0:1], X[1:2])[0][0])
        except Exception:
            pass
    tokens = re.findall(r"[a-zA-Z']{3,}", text.lower())
    kw = {k.lower() for k in keywords}
    hits = sum(1 for t in tokens if t in kw)
    return hits / max(1, len(tokens))

# ---------- Dates ----------
def parse_date(date_str: str) -> Optional[dt.datetime]:
    fmts = [
        "%Y-%m-%dT%H:%M:%S%z","%Y-%m-%dT%H:%M:%SZ","%Y-%m-%d %H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %z","%a, %d %b %Y %H:%M:%S %Z","%Y-%m-%d",
        "%d %b %Y","%B %d, %Y"
    ]
    for f in fmts:
        try:
            return dt.datetime.strptime(date_str, f)
        except Exception:
            continue
    return None

# ---------- RSS (no feedparser) ----------
ATOM = "{http://www.w3.org/2005/Atom}"

def _text(elem: Optional[ET.Element]) -> str:
    return _normalize(elem.text if elem is not None and elem.text else "")

def _find(e: ET.Element, tag: str) -> Optional[ET.Element]:
    x = e.find(tag)
    if x is not None: return x
    if not tag.startswith("{"): return e.find(ATOM + tag)
    return None

def _findall(e: ET.Element, tag: str) -> List[ET.Element]:
    return list(e.findall(tag)) + list(e.findall(ATOM + tag))

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_feed_raw(url: str, timeout: int = 12) -> bytes:
    try:
        r = get_session().get(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; AfricaCommodityPulse/1.0)",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        }, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            soft_fail("Skipped a source (non-200).", f"{url} -> {r.status_code}")
        return r.content if r.status_code == 200 else (r.content or b"")
    except Exception as e:
        soft_fail("Skipped one source due to connectivity.", f"fetch_feed_raw EXC {url}: {e}")
        return b""

def parse_feed(content: bytes, base: str) -> List[Dict[str,str]]:
    items = []
    if not content: return items
    try:
        root = ET.fromstring(content)
        ch = root.find("channel")
        if ch is not None:
            for it in ch.findall("item"):
                title = _text(_find(it,"title")) or "(untitled)"
                link = _text(_find(it,"link"))
                if not link:
                    link = _text(_find(it,"guid"))
                if link.startswith("/"):
                    link = urljoin(base, link)
                summary = _text(_find(it,"description"))
                pub = _text(_find(it,"pubDate"))
                items.append({"title":title,"link":link,"summary":summary,"published_raw":pub})
            return items
        # Atom
        for en in _findall(root,"entry"):
            title = _text(_find(en,"title")) or "(untitled)"
            link_el = _find(en,"link")
            link = link_el.attrib.get("href","") if link_el is not None else ""
            if link.startswith("/"):
                link = urljoin(base, link)
            summary = _text(_find(en,"summary")) or _text(_find(en,"content"))
            pub = _text(_find(en,"updated")) or _text(_find(en,"published"))
            items.append({"title":title,"link":link,"summary":summary,"published_raw":pub})
        return items
    except Exception as e:
        soft_fail("Skipped one feed with invalid XML.", f"parse_feed EXC {e}")
        return items

def fetch_from_feed(url: str, start_dt: dt.datetime, end_dt: dt.datetime) -> List[Dict[str,Any]]:
    raw = fetch_feed_raw(url)
    base = url
    out = []
    for e in parse_feed(raw, base):
        title = _normalize(e.get("title",""))
        link = e.get("link","")
        summary = _normalize(e.get("summary",""))
        pub = e.get("published_raw","")
        published_str = "Date unknown"
        if pub:
            d = parse_date(pub)
            if d:
                d = d.astimezone(dt.timezone.utc) if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
                if not (start_dt <= d <= end_dt):  # date window
                    continue
                published_str = d.strftime("%Y-%m-%d %H:%M UTC")
        out.append({"source": urlparse(url).netloc, "title": title, "link": link, "summary": summary, "published": published_str})
    return out

# ---------- Light page scraping (optional slow step) ----------
def get_og_image(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    for sel in [("meta",{"property":"og:image"}),("meta",{"name":"twitter:image"}),
                ("meta",{"property":"twitter:image"}),("link",{"rel":"image_src"})]:
        el = soup.find(*sel)
        if el:
            src = el.get("content") or el.get("href")
            if src:
                if src.startswith("//"): return "https:" + src
                if src.startswith("/"):  return urljoin(base_url, src)
                return src
    return None

def get_favicon_url(domain_url: str) -> str:
    p = urlparse(domain_url)
    return f"{p.scheme}://{p.netloc}/favicon.ico"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_article_text_and_image(url: str, timeout: int = 8) -> Tuple[str,str]:
    if not url: return "", FALLBACK_IMG
    try:
        r = get_session().get(url, headers={"User-Agent":"Mozilla/5.0 (compatible; AfricaCommodityPulse/1.0)"}, timeout=timeout)
        if r.status_code != 200:
            return "", FALLBACK_IMG
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","noscript","nav","footer","iframe","form"]): tag.decompose()

        text = ""
        for sel in ["article","[role='main']",".article-body",".post-content","main",".content",".entry-content",".story-content","body"]:
            for n in soup.select(sel):
                t = _normalize(n.get_text(" ", strip=True))
                if len(t) > len(text): text = t
        if len(text) < 60: text = ""

        img = get_og_image(soup, url) or get_favicon_url(url) or FALLBACK_IMG
        return text, img
    except Exception:
        return "", FALLBACK_IMG

# ---------- Summaries ----------
def summarize(text: str, n: int, keywords: List[str]) -> str:
    if not text: return ""
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s for s in sents if 30 <= len(s) <= 400][:60]
    if len(sents) <= n: return " ".join(sents)
    if HAS_SK:
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=4000)
            X = vec.fit_transform(sents)
            centroid = X.mean(axis=0)
            sims = cosine_similarity(X, centroid).ravel()
            if keywords:
                kw = [k.lower() for k in keywords]
                sims += 0.05 * np.array([sum(1 for w in re.findall(r"[a-z']+", s.lower()) if w in kw) for s in sents])
            idx = sims.argsort()[-n:][::-1]
            return " ".join([sents[i] for i in idx])
        except Exception:
            pass
    return " ".join(sents[:n])

# ---------- UI: hero ----------
st.set_page_config(page_title=APP_NAME, page_icon="🌍", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"""
<div class="hero">
  <h1>{APP_NAME}</h1>
  <p>{TAGLINE}</p>
  <p style="opacity:.9;">{QUOTE}</p>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---------- Action bar ----------
c1, c2 = st.columns([1,1])
with c1:
    run_btn = st.button("🚀 Scan Now", use_container_width=True)
with c2:
    if st.button("♻️ Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ---------- Config sidebar ----------
with st.sidebar:
    with st.expander("⚙️ Configurations", expanded=False):
        st.subheader("Feeds (Africa & commodities)")
        chosen_sources: List[str] = []
        for name, url in DEFAULT_SOURCES.items():
            if st.checkbox(name, value=True, key=f"src_{name}"):
                chosen_sources.append(url)
        st.caption("You can paste more RSS/Atom URLs below (one per line):")
        extra = st.text_area("Extra feeds", value="", height=80, key="extra_feeds")
        for line in (extra or "").splitlines():
            u = line.strip()
            if u.startswith("http"):
                chosen_sources.append(u)

        st.markdown("---")
        st.subheader("Window & Limits")
        mode = st.radio("Date mode", ["Quick", "Custom"], horizontal=True)
        if mode == "Quick":
            quick = {"24h":1,"3d":3,"7d":7,"14d":14,"30d":30}
            sel = st.selectbox("Range", list(quick.keys()), index=2)
            end_date = dt.datetime.now(dt.timezone.utc)
            start_date = end_date - dt.timedelta(days=quick[sel])
        else:
            d1, d2 = st.columns(2)
            with d1: sd = st.date_input("Start", value=dt.date.today()-dt.timedelta(days=7))
            with d2: ed = st.date_input("End", value=dt.date.today())
            start_date = dt.datetime.combine(sd, dt.time.min, tzinfo=dt.timezone.utc)
            end_date   = dt.datetime.combine(ed, dt.time.max, tzinfo=dt.timezone.utc)

        per_source_cap = st.number_input("Per-source cap", 1, 100, 15, 1)
        overall_cap    = st.number_input("Overall cap (after filters)", 10, 500, 120, 10)
        top_k_fulltext = st.number_input("Fetch full pages for top K / source", 0, 30, 5, 1)
        timeout_s      = st.number_input("Network timeout (sec)", 4, 30, 8, 1)

        st.markdown("---")
        st.subheader("Relevance Filters")
        africa_only = st.checkbox("Africa-only filter", value=True)
        strict_titles = st.checkbox("Require commodity term in title/summary", value=True)
        hard_terms_edit = st.text_input("Commodity terms (comma-separated)", ", ".join(TARGET_TERMS))
        hard_terms = [t.strip() for t in hard_terms_edit.split(",") if t.strip()]
        kw_text = st.text_area("Soft keywords (scoring)", ", ".join(DEFAULT_KEYWORDS), height=100)
        soft_keywords = [k.strip() for k in kw_text.split(",") if k.strip()]
        min_rel = st.slider("Min soft relevance", 0.0, 1.0, 0.10, 0.01)

        st.markdown("---")
        st.subheader("Performance")
        fast_mode = st.checkbox("Fast mode (no full-text scraping)", value=True)
        workers = st.number_input("Concurrency (threads)", 2, 16, 6, 1)

        st.markdown("---")
        st.subheader("Newsdata.io (optional)")
        use_nd = st.checkbox("Use Newsdata.io", value=False)
        nd_key = get_newsdata_api_key()
        if use_nd and not nd_key:
            st.warning("Add NEWSDATA_API_KEY to .env or Streamlit Secrets to enable Newsdata.")
        nd_query = st.text_input("Query", "(cashew OR shea OR cocoa OR gold OR oil) AND (Africa OR Ghana OR Nigeria OR Côte d’Ivoire)")
        nd_pages = st.number_input("API pages", 1, 10, 2, 1)

# ---------- Fetchers ----------
def prefilter_item(t: str, s: str) -> bool:
    # 1) commodity term check
    if strict_titles and not (hard_hit(t, hard_terms) or hard_hit(s, hard_terms)):
        return False
    # 2) Africa-only
    if africa_only and not (africa_hit(t) or africa_hit(s)):
        return False
    return True

def enrich_fast(entry: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    try:
        title = entry["title"] or "(untitled)"
        summary = entry.get("summary","")
        if not prefilter_item(title, summary):
            return None
        combo = " ".join([title, summary])
        rel = keyword_relevance(combo, soft_keywords)
        if rel < min_rel:
            return None
        impacts = classify_impact(combo)
        return {
            "source": entry["source"],
            "title": title,
            "link": entry["link"],
            "published": entry["published"],
            "relevance": float(rel),
            "impact": impacts,
            "auto_summary": summarize(summary or title, 3, soft_keywords),
            "image": FALLBACK_IMG,
        }
    except Exception as e:
        soft_fail("Skipped one item (fast enrich).", f"enrich_fast EXC {e}")
        return None

def enrich_full(entry: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    try:
        title = entry["title"] or "(untitled)"
        summary = entry.get("summary","")
        if not prefilter_item(title, summary):
            return None

        # fetch full page ONLY now
        body, img = fetch_article_text_and_image(entry["link"], timeout=timeout_s)
        text_for_score = " ".join([title, body or summary])
        rel = keyword_relevance(text_for_score, soft_keywords)
        if rel < min_rel:
            return None

        impacts = classify_impact(text_for_score)
        return {
            "source": entry["source"],
            "title": title,
            "link": entry["link"],
            "published": entry["published"],
            "relevance": float(rel),
            "impact": impacts,
            "auto_summary": summarize(body or summary or title, 3, soft_keywords),
            "image": img or FALLBACK_IMG,
        }
    except Exception as e:
        soft_fail("Skipped one item (full enrich).", f"enrich_full EXC {e}")
        return None

def fetch_all(sources: List[str]) -> List[Dict[str,Any]]:
    rows: List[Dict[str,Any]] = []
    if not sources and not use_nd:
        return rows

    progress = st.progress(0.0)
    info = st.empty()

    # 1) RSS scan (FAST)
    raw_items: List[Dict[str,Any]] = []
    for i, src in enumerate(sources, start=1):
        info.info(f"Fetching {urlparse(src).netloc} ({i}/{max(1,len(sources))}) …")
        try:
            items = fetch_from_feed(src, start_date, end_date)
            if per_source_cap: items = items[:per_source_cap]
            # quick prefilter on title/summary before any heavy work
            items = [it for it in items if prefilter_item(it["title"], it.get("summary",""))]
            raw_items.extend(items)
        except Exception as e:
            soft_fail("Skipped one source due to a transient issue.", f"{src} EXC {e}")
        progress.progress(min(0.6, i/max(1,len(sources))*0.6))  # first 60%

    # 2) Optional Newsdata
    if use_nd and nd_key:
        info.info("Fetching Newsdata.io …")
        try:
            items = []
            next_page = None
            for _ in range(int(nd_pages)):
                q = {"apikey": nd_key, "q": nd_query}
                if next_page: q["page"] = next_page
                r = get_session().get("https://newsdata.io/api/1/latest", params=q, timeout=timeout_s)
                if r.status_code != 200:
                    soft_fail("One API page was skipped.", f"newsdata {r.status_code} {r.text[:200]}")
                    break
                data = r.json()
                results = data.get("results") or []
                for a in results:
                    title = _normalize(a.get("title",""))
                    link = a.get("link") or a.get("url") or ""
                    desc = _normalize(a.get("description","")) or _normalize(a.get("content",""))
                    pub  = a.get("pubDate") or a.get("published_at") or ""
                    published_str = "Date unknown"
                    ok_date = True
                    if pub:
                        d = parse_date(pub)
                        if d:
                            d = d.astimezone(dt.timezone.utc) if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
                            ok_date = start_date <= d <= end_date
                            published_str = d.strftime("%Y-%m-%d %H:%M UTC")
                    if not ok_date: continue
                    items.append({"source":"newsdata.io", "title":title, "link":link, "summary":desc, "published":published_str})
                next_page = data.get("nextPage") or data.get("next_page")
                if not next_page: break
            if per_source_cap: items = items[:per_source_cap]
            items = [it for it in items if prefilter_item(it["title"], it.get("summary",""))]
            raw_items.extend(items)
        except Exception as e:
            soft_fail("API temporarily unavailable.", f"newsdata EXC {e}")

    if not raw_items:
        info.empty(); progress.empty()
        return rows

    # 3) Enrich: either fast or 2-stage (topK fulltext)
    info.info("Scoring & summarizing …")
    rows_fast: List[Dict[str,Any]] = []
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futures = [ex.submit(enrich_fast, it) for it in raw_items]
        for i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            if r: rows_fast.append(r)
            if i % 10 == 0:
                progress.progress(0.6 + min(0.2, i/len(raw_items)*0.2))  # to 80%

    rows_fast = [r for r in rows_fast if r]
    rows_fast.sort(key=lambda x: x["relevance"], reverse=True)

    # If fast mode, we're done
    if fast_mode or top_k_fulltext == 0:
        final_rows = rows_fast[:int(overall_cap)]
        info.empty(); progress.progress(1.0); progress.empty()
        return final_rows

    # Else fetch full text for top K per source (speed + quality)
    info.info("Fetching full articles for top items …")
    # bucket by source
    by_src: Dict[str, List[Dict[str,Any]]] = {}
    for r in rows_fast:
        by_src.setdefault(r["source"], []).append(r)

    candidates: List[Dict[str,Any]] = []
    for src, bucket in by_src.items():
        candidates.extend(bucket[:int(top_k_fulltext)])

    final_rows: List[Dict[str,Any]] = []
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futures = [ex.submit(enrich_full, it) for it in candidates]
        for i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            if r: final_rows.append(r)
            if i % 10 == 0:
                progress.progress(0.8 + min(0.2, i/max(1,len(candidates))*0.2))  # to 100%

    # merge: prefer full rows, then fill with fast rows if needed
    def keyer(x): return hashlib.md5((x["title"]+"||"+x["link"]).encode()).hexdigest()
    full_keys = {keyer(r) for r in final_rows}
    spill = [r for r in rows_fast if keyer(r) not in full_keys]
    final = (final_rows + spill)[:int(overall_cap)]

    info.empty(); progress.progress(1.0); progress.empty()
    return final

# ---------- Rendering ----------
def render_card(row: pd.Series):
    st.markdown(f"""
    <div class="card">
      <img class="thumb" src="{row.get('image',FALLBACK_IMG)}" alt="thumbnail">
      <div class="card-body">
        <div class="meta">{row['source']} · {row['published']} · Relevance {row['relevance']:.0%}</div>
        <div class="title">{row['title']}</div>
        <div class="badges">{"".join([f'<span class="badge">{t}</span>' for t in row["impact"]])}</div>
        <div class="summary">{row['auto_summary']}</div>
        <div style="margin-top:10px;"><a class="link" href="{row['link']}" target="_blank">Read full article →</a></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

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

def friendly_error_summary():
    if not SOFT_ERRORS: return
    counts: Dict[str,int] = {}
    for m in SOFT_ERRORS: counts[m] = counts.get(m,0)+1
    bullets = "".join([f"- {msg} _(x{n})_\n" for msg,n in counts.items()])
    st.info(f"**Heads up:** Some sources were skipped/slow, but results are complete.\n\n{bullets}")

def process_rows(rows: List[Dict[str,Any]]) -> pd.DataFrame:
    if not rows: return pd.DataFrame(columns=["source","published","title","relevance","impact","auto_summary","link","image"])
    seen=set(); cleaned=[]
    for r in rows:
        k = hashlib.md5((r["title"]+"||"+r["link"]).encode()).hexdigest()
        if k in seen: continue
        seen.add(k); cleaned.append(r)
    df = pd.DataFrame(cleaned)
    if df.empty: return df
    return df.sort_values("relevance", ascending=False).reset_index(drop=True)

def ui_results(df: pd.DataFrame):
    st.subheader("📊 Results")
    if df.empty:
        st.warning("No relevant articles found. Try widening date range or relaxing filters.")
        return
    c1, c2 = st.columns(2)
    with c1:
        impacts = sorted({t for tags in df["impact"] for t in tags})
        f_imp = st.multiselect("Filter by impact", impacts, [])
    with c2:
        f_src = st.multiselect("Filter by source", sorted(df["source"].unique()), [])
    filt = df.copy()
    if f_imp: filt = filt[filt["impact"].apply(lambda x: any(t in x for t in f_imp))]
    if f_src: filt = filt[filt["source"].isin(f_src)]

    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for _, row in filt.iterrows():
        render_card(row)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📝 Daily Digest")
    digest_md = make_digest(filt if (f_imp or f_src) else df, top_k=12)
    st.markdown(digest_md)

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    st.download_button("📥 Download CSV", data=filt.to_csv(index=False).encode("utf-8"),
        file_name=f"africa_commodity_pulse_{ts}.csv", mime="text/csv")
    st.download_button("📥 Download Digest (Markdown)", data=digest_md.encode("utf-8"),
        file_name=f"africa_commodity_pulse_digest_{ts}.md", mime="text/markdown")

# ---------- Main ----------
if not run_btn:
    st.info("""**What this does**
- Prioritizes **Africa** + **cashew/shea/cocoa/gold/oil**
- Fast scan of RSS/Atom, then (optionally) fetches full pages for the **top K** only
- Strict title/summary filter keeps noise out; soft relevance rescoring ranks results
- Friendly errors (no tracebacks), CSV + Markdown digest""")
else:
    try:
        with st.spinner("Scanning feeds and ranking relevant items…"):
            rows = fetch_all(chosen_sources)
            df = process_rows(rows)[:200]
            ui_results(df)
    except Exception as e:
        soft_fail("Something went wrong building results.", f"MAIN EXC {e}")
        st.error("We hit a hiccup assembling the results. Please tweak filters and try again.")
    finally:
        friendly_error_summary()
