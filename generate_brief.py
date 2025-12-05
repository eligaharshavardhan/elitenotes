#!/usr/bin/env python3#!/usr/bin/env python3
"""
generate_brief.py - Elitenotes generator (drop-in)

Features:
- Fetch RSS/JSON feeds in parallel
- Summarize via Bytez (preferred) with fallback
- Cache article text & summaries in SQLite
- Optional embeddings (sentence-transformers + faiss) if installed; disable with NO_EMBED=1
- Generate insight via OpenAI if OPENAI_API_KEY present, otherwise heuristic
- Render HTML using brief_template.html, injects SEO meta + JSON-LD
- Push .md + .html to GitHub and open PR (requires GITHUB_TOKEN); checks for existing open PRs with similar title and skips duplicates
"""

import os, json, base64, hashlib, traceback, re, requests, feedparser, sqlite3, time, pathlib
from datetime import datetime, timezone
from readability import Document
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from html import escape
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter, Retry
from urllib.parse import urljoin

# ---------- OPTIONAL HEAVY LIBS ----------
NO_EMBED = os.getenv("NO_EMBED", "") == "1"
try:
    import numpy as np
except Exception:
    np = None

try:
    if not NO_EMBED:
        from sentence_transformers import SentenceTransformer
        HAS_ST = True
    else:
        HAS_ST = False
except Exception:
    HAS_ST = False

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# ---------- CONFIG ----------
BYTEZ_TOKEN = os.getenv("BYTEZ_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.getenv("GITHUB_REPO", "eligaharshavardhan/elitenotes").strip()
GITHUB_DEFAULT_BRANCH = os.getenv("GITHUB_DEFAULT_BRANCH", "main").strip()

SITE_NAME = os.getenv("SITE_NAME", "elitenotes")
SITE_URL = os.getenv("SITE_URL", "https://yourdomain.com").rstrip("/")
TWITTER_HANDLE = os.getenv("TWITTER_HANDLE", "elitenotes")
DEFAULT_OG = os.getenv("DEFAULT_OG", SITE_URL + "/assets/og-elitenotes-2025.jpg")

SOURCES_FILE = "sources.json"
CONTENT_DIR = "content"

# IMPORTANT: separate local and repo preview paths to avoid "public\previews" bug
PREVIEW_DIR_LOCAL = os.path.join("public", "previews")  # local filesystem
PREVIEW_DIR_REPO = "public/previews"                    # GitHub repo path (must use /)

TEMPLATE_PATH = "brief_template.html"
DB_PATH = "elitenotes.db"
CACHE_DIR = pathlib.Path(".cache/articles")
GITHUB_API = "https://api.github.com"
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = 384

MIN_WORDS, MAX_WORDS = 350, 450
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(CONTENT_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR_LOCAL, exist_ok=True)

# ---------- HELPERS ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_hash(s: str) -> str:
    return hashlib.sha1((s or "").encode()).hexdigest()[:8]


def load_json(p, d):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return d


def save_json(p, d):
    json.dump(d, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def text_to_paragraphs(text: str) -> str:
    if not text:
        return ""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "\n\n".join("<p>" + escape(p).replace("\n", "<br/>") + "</p>" for p in paras)


def make_session():
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "elitenotes-bot"})
    return s

# ---------- DB ----------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS articles (
        id TEXT PRIMARY KEY, title TEXT, url TEXT, source TEXT,
        published TEXT, summary TEXT, content TEXT, embedding BLOB
    )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS seen (
        url_hash TEXT PRIMARY KEY,
        url TEXT,
        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS cache (
        cache_key TEXT PRIMARY KEY,
        value BLOB,
        content_type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )"""
    )
    con.commit()
    con.close()


def save_article(article):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    emb = None
    try:
        if article.get("embedding") is not None and np is not None:
            emb = article["embedding"].astype("float32").tobytes()
    except Exception:
        emb = None
    cur.execute(
        """INSERT OR REPLACE INTO articles
        (id,title,url,source,published,summary,content,embedding)
        VALUES (?,?,?,?,?,?,?,?)""",
        (
            article["id"],
            article["title"],
            article["url"],
            article["source"],
            article["published"],
            article["summary"],
            article["content"],
            emb,
        ),
    )
    con.commit()
    con.close()


def mark_seen(url_hash: str, url: str = None):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT OR IGNORE INTO seen (url_hash, url) VALUES (?, ?)", (url_hash, url))
        con.commit()


def is_seen(url_hash: str) -> bool:
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT 1 FROM seen WHERE url_hash=? LIMIT 1", (url_hash,))
        return cur.fetchone() is not None


def cache_set(key: str, value: bytes, content_type: str = "text"):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO cache (cache_key, value, content_type, created_at) 
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (key, value, content_type),
        )
        con.commit()


def cache_get(key: str):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT value FROM cache WHERE cache_key=? LIMIT 1", (key,))
        row = cur.fetchone()
        return row[0] if row else None

# ---------- FEEDS ----------
def fetch_reddit_feed(url, session):
    try:
        r = session.get(url, timeout=8)
        posts = []
        for p in r.json().get("data", {}).get("children", []):
            d = p.get("data", {})
            posts.append(
                {
                    "title": d.get("title", ""),
                    "link": f"https://reddit.com{d.get('permalink', '')}",
                    "summary": d.get("selftext", "")[:500],
                    "published": datetime.utcfromtimestamp(d.get("created_utc", 0)).isoformat(),
                    "source": "Reddit",
                }
            )
        return posts
    except Exception:
        return []


def fetch_feed_items_parallel(sources, limit=3):
    session = make_session()

    def fetch_one(url):
        try:
            if "reddit.com" in url and url.endswith(".json"):
                return fetch_reddit_feed(url, session)
            r = session.get(url, timeout=6)
            feed = feedparser.parse(r.text)
            items = []
            for e in feed.entries[:limit]:
                items.append(
                    {
                        "title": e.get("title", "").strip(),
                        "link": e.get("link", "").strip(),
                        "summary": e.get("summary", "")[:1000],
                        "published": e.get("published", ""),
                        "source": feed.feed.get("title", url),
                    }
                )
            return items
        except Exception:
            return []

    items = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sources)))) as ex:
        futures = [ex.submit(fetch_one, u) for u in sources]
        for fut in as_completed(futures):
            items += fut.result() or []

    def dparse(x):
        try:
            return dateparser.parse(x) if x else datetime.min
        except Exception:
            return datetime.min

    items.sort(key=lambda it: dparse(it.get("published", "1970")), reverse=True)
    return items


def load_sources():
    defaults = [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://www.ft.com/rss/home",
        "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en",
        "https://rsshub.app/twitter/trending",
        "https://www.reddit.com/r/news/top/.json?limit=5&t=day",
        "https://www.reddit.com/r/technology/top/.json?limit=5&t=day",
        "https://rsshub.app/instagram/tag/ai",
        "https://trends.google.com/trends/trendingsearches/daily/rss?geo=US",
    ]
    file_sources = load_json(SOURCES_FILE, [])
    out = []
    seen_urls = set()
    for u in defaults + file_sources:
        if u not in seen_urls:
            seen_urls.add(u)
            out.append(u)
    return out

# ---------- ARTICLE SCRAPER & CACHE ----------
def key_for_url(url: str) -> str:
    return "url:" + hashlib.sha1((url or "").encode("utf-8")).hexdigest()[:16]


def key_for_summary(content_hash: str) -> str:
    return "summary:" + (content_hash or "")[:32]


def get_cached_article_text(url: str):
    b = cache_get(key_for_url(url))
    return b.decode("utf-8") if b else None


def set_cached_article_text(url: str, text: str):
    cache_set(key_for_url(url), text.encode("utf-8"), content_type="article_text")


def get_cached_summary(content_hash: str):
    b = cache_get(key_for_summary(content_hash))
    return b.decode("utf-8") if b else None


def set_cached_summary(content_hash: str, summary_text: str):
    cache_set(key_for_summary(content_hash), summary_text.encode("utf-8"), content_type="summary")


def cached_fetch_article(url, session, max_chars=6000):
    if not url:
        return ""
    try:
        cached = get_cached_article_text(url)
        if cached:
            return cached[:max_chars]
    except Exception:
        pass

    try:
        r = session.get(url, timeout=10)
        if r.status_code != 200:
            return ""
        doc = Document(r.text)
        text = BeautifulSoup(doc.summary(), "html.parser").get_text("\n").strip()
        try:
            set_cached_article_text(url, text)
        except Exception:
            pass
        return text[:max_chars]
    except Exception:
        return ""

# ---------- SUMMARIZERS ----------
def summarize_with_bytez(text):
    if not BYTEZ_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {BYTEZ_TOKEN}", "Content-Type": "application/json"}
    prompt = (
        "Summarize and analyze this content in ~400 words. "
        "Include TL;DR, 3 bullet points, and Why it matters.\n\n"
        f"{text[:2500]}"
    )
    payload = {
        "model": "google/gemma-3-1b-it",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert journalist writing concise briefs for elitenotes.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 800,
    }
    endpoints = [
        "https://api.bytez.ai/v1/chat/completions",
        "https://bytez-gateway-production.up.railway.app/v1/chat/completions",
    ]
    for url in endpoints:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=25)
            if r.status_code != 200:
                continue
            j = r.json()
            if "choices" in j and j["choices"]:
                c = j["choices"][0]
                msg = c.get("message", {}).get("content") or c.get("text")
                if msg:
                    return msg.strip()
        except Exception:
            continue
    return None


def split_sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (t or "").strip()) if s.strip()]


def fallback_summarize(text):
    if not text:
        return None
    s = split_sentences(text)
    tldr = " ".join(s[:2]) if len(s) >= 2 else (s[0] if s else "")
    bullets = [sent for sent in s[2:8] if 8 <= len(sent.split()) <= 40][:3]
    body = " ".join(s)[: MAX_WORDS * 8]
    why = bullets[0] if bullets else tldr
    out = f"{tldr}\n\nHighlights:\n" + "\n".join(f"- {b}" for b in bullets)
    out += f"\n\nSummary:\n{body}\n\nWhy it matters: {why}"
    return out

# ---------- EMBEDDING PIPELINE (optional) ----------
_emb_model = None
_index = None
_index_ids = []


def init_embedding_pipeline():
    global _emb_model, _index, _index_ids
    if not HAS_ST or NO_EMBED or np is None:
        _emb_model = None
        _index = None
        _index_ids = []
        return
    try:
        _emb_model = SentenceTransformer(EMBED_MODEL)
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT id,embedding FROM articles WHERE embedding IS NOT NULL")
        rows = cur.fetchall()
        con.close()
        _index_ids = [r[0] for r in rows]
        if rows and HAS_FAISS:
            arr = np.vstack([np.frombuffer(r[1], dtype="float32") for r in rows])
            faiss.normalize_L2(arr)
            _index = faiss.IndexFlatIP(EMBED_DIM)
            _index.add(arr)
    except Exception:
        _emb_model = None
        _index = None
        _index_ids = []


def embed_texts(texts):
    if _emb_model is None:
        raise RuntimeError("Embedding model not initialized")
    return _emb_model.encode(texts, convert_to_numpy=True)


def add_article_embedding_to_db(article):
    if _emb_model is None or np is None:
        return
    try:
        txt = (article.get("summary", "") or "") + "\n\n" + (article.get("content", "") or "")
        emb = embed_texts([txt])[0].astype("float32")
        article["embedding"] = emb
        save_article(article)
        init_embedding_pipeline()
    except Exception:
        pass


def query_similar_articles(text, k=6):
    if _emb_model is None or _index is None:
        return []
    emb = embed_texts([text])[0].astype("float32")
    faiss.normalize_L2(emb.reshape(1, -1))
    D, I = _index.search(emb.reshape(1, -1), k)
    results = []
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(_index_ids):
            continue
        aid = _index_ids[idx]
        cur.execute(
            "SELECT id,title,url,source,published,summary,content FROM articles WHERE id=?",
            (aid,),
        )
        row = cur.fetchone()
        if row:
            results.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "url": row[2],
                    "source": row[3],
                    "published": row[4],
                    "summary": row[5],
                    "content": row[6],
                    "score": float(dist),
                }
            )
    con.close()
    return results

# ---------- INSIGHT (OpenAI optional) ----------
def call_openai_insight(system_prompt, user_prompt):
    if not OPENAI_API_KEY:
        return None
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 700,
        }
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=40)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def generate_insight(article, k=6):
    text_for_search = (article.get("summary", "") or "") + "\n\n" + (article.get("content", "") or "")
    try:
        similar = query_similar_articles(text_for_search, k=k)
    except Exception:
        similar = []

    evidence = ""
    for s in similar:
        evidence += (
            f"- {s.get('published', '')}: {s.get('title', '')} ({s.get('source', '')})\n"
            f"  {(s.get('summary') or '')[:220]}\n\n"
        )

    words = [w for w in (article.get("title") or "").split() if len(w) > 4][:4]
    counts = {}
    try:
        import pandas as pd

        con = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql_query(
                "SELECT id,title,url,source,published,summary,content FROM articles",
                con,
                parse_dates=["published"],
            )
        except Exception:
            df = None
        con.close()
        if df is not None and not df.empty:
            df["text"] = (
                df["title"].fillna("")
                + " "
                + df["summary"].fillna("")
                + " "
                + df["content"].fillna("")
            ).str.lower()
            for w in words:
                raw = int(df["text"].str.contains(w.lower(), na=False).sum())
                counts[w] = raw
    except Exception:
        counts = {}

    counts = {str(k): int(v) for k, v in (counts or {}).items()}
    system = (
        "You are a journalist-analyst. Use evidence and counts to produce a short insight "
        "connecting past patterns to near-future implications."
    )
    user = (
        f"Article:\n{article.get('title')}\n\n"
        f"Summary:\n{article.get('summary')}\n\n"
        f"Similar past items:\n{evidence}\n\n"
        f"Keyword counts:\n{json.dumps(counts)}\n\n"
        "Write: 1) TL;DR (one line), 2) 3 bullet points of pattern, 3) One 1-sentence prediction (what to watch next)."
    )
    if OPENAI_API_KEY:
        res = call_openai_insight(system, user)
        if res:
            return res

    # Fallback insight
    tl = (article.get("summary") or "")[:200]
    bullets = [
        "This mirrors patterns in our archive.",
        f"Keywords: {', '.join(words) or 'n/a'}. Counts: {counts}.",
        "If mentions rise in the next week, trend is accelerating.",
    ]
    pred = "If signals continue, expect follow-ups in the coming weeks."
    out = (
        f"{tl}\n\nHighlights:\n"
        + "\n".join("- " + b for b in bullets)
        + f"\n\nPrediction:\n{pred}"
    )
    return out

# ---------- SEO helper ----------
def make_seo_metadata(title, summary, date_iso, url_path, author="elitenotes", tags=None, og_image=None):
    if not title:
        title = SITE_NAME
    seo_title = title.strip()
    desc = (summary or "").strip().replace("\n", " ").replace('"', "'")
    if len(desc) > 155:
        desc = desc[:152].rsplit(" ", 1)[0] + "..."
    canonical = urljoin(SITE_URL + "/", url_path.lstrip("/"))
    og = og_image or DEFAULT_OG
    seo = {
        "SEO_TITLE": seo_title,
        "SEO_DESCRIPTION": desc,
        "CANONICAL_URL": canonical,
        "OG_IMAGE": og,
        "OG_IMAGE_WIDTH": 1200,
        "OG_IMAGE_HEIGHT": 630,
        "SITE_NAME": SITE_NAME,
        "TWITTER_HANDLE": TWITTER_HANDLE,
        "ROBOTS_META": "index,follow",
    }
    published_iso = date_iso or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "mainEntityOfPage": {"@type": "WebPage", "@id": canonical},
        "headline": seo_title,
        "description": desc,
        "image": [og],
        "datePublished": published_iso,
        "dateModified": published_iso,
        "author": {"@type": "Person", "name": author},
        "publisher": {
            "@type": "Organization",
            "name": SITE_NAME,
            "logo": {
                "@type": "ImageObject",
                "url": urljoin(SITE_URL, "/assets/og-elitenotes-2025.jpg"),
            },
        },
    }
    if tags:
        ld["keywords"] = ",".join(tags if isinstance(tags, (list, tuple)) else [tags])
    json_ld_str = '<script type="application/ld+json">' + json.dumps(ld, ensure_ascii=False) + "</script>"
    return seo, json_ld_str

# ---------- TEMPLATE RENDER ----------
def render_template_from_file(template_path, meta, content_html, insight_html):
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    tpl = open(template_path, encoding="utf-8").read()
    for k, v in meta.items():
        tpl = tpl.replace(f"{{{{{k}}}}}", str(v))
    if "{{INSIGHT}}" in tpl:
        tpl = tpl.replace("{{INSIGHT}}", insight_html or "")
    else:
        insight_block = (
            '<div class="card" style="margin-bottom:14px;">'
            '<h3 style="margin:0 0 8px">Insight</h3>'
            f'<div class="muted">{insight_html or ""}</div></div>\n'
        )
        if "{{CONTENT_HTML}}" in tpl:
            tpl = tpl.replace("{{CONTENT_HTML}}", insight_block + (content_html or ""))
            return tpl
        else:
            if "</article>" in tpl:
                tpl = tpl.replace(
                    "</article>",
                    insight_block + (content_html or "") + "</article>",
                    1,
                )
            else:
                tpl = tpl + insight_block + (content_html or "")
            return tpl
    tpl = tpl.replace("{{CONTENT_HTML}}", content_html or "")
    return tpl

# ---------- GITHUB helpers ----------
def gh_headers():
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN env var not set.")
    return {"Authorization": f"token {GITHUB_TOKEN.strip()}", "Accept": "application/vnd.github+json"}


def create_branch(branch):
    r = requests.get(
        f"{GITHUB_API}/repos/{GITHUB_REPO}/git/ref/heads/{GITHUB_DEFAULT_BRANCH}",
        headers=gh_headers(),
        timeout=10,
    )
    r.raise_for_status()
    sha = r.json()["object"]["sha"]
    resp = requests.post(
        f"{GITHUB_API}/repos/{GITHUB_REPO}/git/refs",
        headers=gh_headers(),
        json={"ref": f"refs/heads/{branch}", "sha": sha},
        timeout=10,
    )
    if resp.status_code not in (201, 422):
        resp.raise_for_status()


def create_file(path, content, branch, msg):
    b64 = base64.b64encode(content.encode()).decode()
    requests.put(
        f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}",
        headers=gh_headers(),
        json={"message": msg, "content": b64, "branch": branch},
        timeout=10,
    )


def open_pr(branch, title):
    r = requests.post(
        f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls",
        headers=gh_headers(),
        json={
            "title": title,
            "head": branch,
            "base": GITHUB_DEFAULT_BRANCH,
            "body": "Automated brief — please review",
        },
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("html_url")


def find_open_pr_for_title(title_substr):
    if not GITHUB_TOKEN:
        return None
    try:
        headers = gh_headers()
        page = 1
        while True:
            r = requests.get(
                f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls?state=open&per_page=100&page={page}",
                headers=headers,
                timeout=10,
            )
            r.raise_for_status()
            prs = r.json()
            if not prs:
                break
            for pr in prs:
                pr_title = pr.get("title", "") or ""
                if title_substr.lower() in pr_title.lower():
                    return pr
            page += 1
            if page > 5:
                break
    except Exception:
        return None
    return None

# ---------- MAIN ----------
def run_once():
    start_all = time.time()
    try:
        init_db()
        init_embedding_pipeline()

        sources = load_sources()
        if not sources:
            print("❌ No sources.json found.")
            return

        print(f"🌎 Fetching from {len(sources)} feeds...")
        t0 = time.time()
        items = fetch_feed_items_parallel(sources)
        print("⏱ Feeds fetched:", round(time.time() - t0, 2), "s")

        candidate = None
        for it in items:
            h = short_hash(it.get("link") or it.get("title"))
            if not is_seen(h):
                candidate = it
                candidate["_hash"] = h
                break

        if not candidate:
            print("ℹ️ No new articles.")
            return

        print("📰 Candidate:", candidate.get("title"))
        session = make_session()

        content_hash = short_hash(candidate.get("link") or candidate.get("title"))

        try:
            cached_summary = get_cached_summary(content_hash)
        except Exception:
            cached_summary = None

        if cached_summary:
            content = cached_summary
            article_text = get_cached_article_text(candidate.get("link")) or candidate.get(
                "summary", ""
            )
            bytez_used = False
            print("♻️ Used cached summary.")
        else:
            if candidate.get("summary") and len(candidate.get("summary")) > 200:
                article_text = candidate.get("summary")
            else:
                article_text = cached_fetch_article(candidate.get("link"), session) or candidate.get(
                    "summary", ""
                )
            print("🧾 Article text length:", len(article_text or ""))

            t1 = time.time()
            content = None
            if BYTEZ_TOKEN:
                content = summarize_with_bytez(article_text)
                bytez_used = bool(content)
            else:
                bytez_used = False

            if not content:
                print("⚠️ Bytez failed or not configured — fallback summarizer used.")
                content = fallback_summarize(article_text)

            print("✍️ Summarize time:", round(time.time() - t1, 2), "s — Bytez used:", bytez_used)

            try:
                set_cached_summary(content_hash, content)
            except Exception:
                pass
            try:
                if article_text:
                    set_cached_article_text(candidate.get("link"), article_text)
            except Exception:
                pass

        if not content:
            print("❌ Nothing generated.")
            return

        # Save article to DB
        aid = content_hash
        article_record = {
            "id": aid,
            "title": candidate.get("title"),
            "url": candidate.get("link"),
            "source": candidate.get("source"),
            "published": candidate.get("published") or now_iso(),
            "summary": (candidate.get("summary") or "")[:800],
            "content": article_text or "",
        }
        save_article(article_record)

        if HAS_ST and not NO_EMBED and np is not None:
            try:
                add_article_embedding_to_db(article_record)
            except Exception as e:
                print("Embedding error:", e)

        t2 = time.time()
        insight_text = generate_insight(article_record) or ""
        print("🔍 Insight generated in", round(time.time() - t2, 2), "s")

        # Render outputs (Markdown + HTML)
        md_meta = {
            "title": candidate.get("title"),
            "date": now_iso(),
            "sources": [candidate.get("link")],
            "summary": candidate.get("summary", "")[:200],
        }
        md_frontmatter = "---\n" + "\n".join(
            f"{k}: {json.dumps(v)}" for k, v in md_meta.items()
        ) + "\n---\n\n"
        md = md_frontmatter + content

        content_html = text_to_paragraphs(content)
        insight_html = text_to_paragraphs(insight_text)

        template_meta = {
            "TITLE": f"elitenotes — {candidate.get('title')}",
            "TITLE_H1": candidate.get("title"),
            "DESCRIPTION": candidate.get("summary", "")[:160],
            "AUTHOR": candidate.get("source", "elitenotes"),
            "DATE": datetime.utcnow().strftime("%B %d, %Y"),
            "LEDE": candidate.get("summary", ""),
            "OG_IMAGE": "/assets/og-elitenotes-2025.jpg",
            "HERO_IMAGE_SRC": "/assets/hero.jpg",
            "HERO_IMAGE_SRCSET": "",
            "HERO_IMAGE_SIZES": "(max-width:980px)100vw,900px",
            "HERO_IMAGE_ALT": "",
            "FEATURED_SHORT_IMAGE": "/assets/short-thumb.jpg",
            "FEATURED_SHORT_ALT": "",
            "FEATURED_SHORT_DURATION": "49s",
            "FEATURED_SHORT_TITLE": "Featured Short",
            "KEY_TAKEAWAYS_LIST": "<li>Predictive intelligence is the core trend.</li><li>Privacy-by-edge compute is now mainstream.</li><li>Sustainability apps solve physical problems.</li>",
            "BRIEFS_LIST": "",
        }

        # SEO + JSON-LD
        date_slug = datetime.utcnow().strftime("%Y-%m-%d")
        urlpath = f"previews/{date_slug}-{aid}.html"

        seo_meta, json_ld = make_seo_metadata(
            title=candidate.get("title"),
            summary=(candidate.get("summary") or "")[:300],
            date_iso=article_record.get("published"),
            url_path=urlpath,
            author=template_meta.get("AUTHOR", "elitenotes"),
            tags=[w for w in (candidate.get("title") or "").split() if len(w) > 3][:6],
            og_image=template_meta.get("OG_IMAGE"),
        )
        template_meta.update(seo_meta)

        html = render_template_from_file(TEMPLATE_PATH, template_meta, content_html, insight_html)

        if "</head>" in html:
            html = html.replace("</head>", json_ld + "\n</head>", 1)
        else:
            html = json_ld + "\n" + html

        # --------- PATHS (LOCAL vs REPO) ----------
        branch = f"auto/{datetime.utcnow().strftime('%Y%m%d')}-{aid}"

        md_filename = f"{date_slug}-{aid}.md"
        html_filename = f"{date_slug}-{aid}.html"

        # repo paths (used with GitHub API) – MUST use forward slashes
        md_path_repo = f"{CONTENT_DIR}/{md_filename}"
        html_path_repo = f"{PREVIEW_DIR_REPO}/{html_filename}"

        # local disk paths (used when GITHUB_TOKEN is not set)
        md_path_local = os.path.join(CONTENT_DIR, md_filename)
        html_path_local = os.path.join(PREVIEW_DIR_LOCAL, html_filename)

        pr_url = None
        try:
            if GITHUB_TOKEN:
                title_snip = (candidate.get("title") or "")[:60]
                existing = find_open_pr_for_title(title_snip)
                if existing:
                    pr_url = existing.get("html_url")
                    print("ℹ️ Existing open PR found — skipping new PR. PR URL:", pr_url)
                else:
                    print("📦 Creating branch:", branch)
                    create_branch(branch)
                    create_file(md_path_repo, md, branch, "Auto brief upload")
                    create_file(html_path_repo, html, branch, "Add HTML preview")
                    pr_url = open_pr(branch, f"Auto: {candidate.get('title')[:80]}")
                    print("✅ PR opened:", pr_url)
            else:
                print("⚠️ GITHUB_TOKEN not set — skipping GitHub upload.")
                # still write files locally for review
                open(md_path_local, "w", encoding="utf-8").write(md)
                open(html_path_local, "w", encoding="utf-8").write(html)
        except Exception as e:
            print("❌ GitHub error:", e)
            try:
                print("GitHub response detail:", e.response.text)
            except Exception:
                pass

        # mark this URL as seen
        try:
            mark_seen(candidate.get("_hash"), candidate.get("link"))
        except Exception:
            pass

        total = round(time.time() - start_all, 2)
        print("\n--- Diagnostics ---")
        print("Bytez used:", bytez_used if "bytez_used" in locals() else False, "| OpenAI available:", bool(OPENAI_API_KEY))
        print("Insight chars:", len(insight_text or ""))
        print("PR URL:", pr_url)
        print("Total runtime:", total, "s")
        print("--- Done ---\n")

    except Exception as e:
        try:
            print("❌ ERROR:", e)
        except Exception:
            print("ERROR:", e)
        traceback.print_exc()


if __name__ == "__main__":
    run_once()


