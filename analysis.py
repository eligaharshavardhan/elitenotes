# analysis.py
import os, sqlite3, json, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from dateutil import parser as dateparser
from html import escape
import requests

DB_PATH = "elitenotes.db"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = 384   # for all-MiniLM-L6-v2

# ---------- DB helpers ----------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id TEXT PRIMARY KEY,
        title TEXT,
        url TEXT,
        source TEXT,
        published TEXT,
        summary TEXT,
        content TEXT,
        embedding BLOB
    )
    """)
    con.commit()
    con.close()

def save_article(article):
    """article: dict with id,title,url,source,published,summary,content,embedding(np.array)"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    emb_blob = None
    if "embedding" in article and article["embedding"] is not None:
        emb_blob = article["embedding"].astype('float32').tobytes()
    cur.execute("""
    INSERT OR REPLACE INTO articles (id,title,url,source,published,summary,content,embedding)
    VALUES (?,?,?,?,?,?,?,?)
    """, (article["id"], article.get("title"), article.get("url"), article.get("source"),
          article.get("published"), article.get("summary"), article.get("content"), emb_blob))
    con.commit()
    con.close()

def load_all_articles():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id,title,url,source,published,summary,content FROM articles", con, parse_dates=["published"])
    con.close()
    return df

def load_embeddings():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, embedding FROM articles WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    con.close()
    ids = []
    embs = []
    for r in rows:
        ids.append(r[0])
        embs.append(np.frombuffer(r[1], dtype='float32'))
    if embs:
        arr = np.vstack(embs)
    else:
        arr = np.empty((0, EMBED_DIM), dtype='float32')
    return ids, arr

# ---------- Embedding & Index ----------
_model = None
_index = None
_ids = []

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model

def build_index():
    global _index, _ids
    ids, arr = load_embeddings()
    _ids = ids
    if arr.shape[0] == 0:
        _index = faiss.IndexFlatIP(EMBED_DIM)  # inner product
        return
    # normalize for cosine
    faiss.normalize_L2(arr)
    _index = faiss.IndexFlatIP(EMBED_DIM)
    _index.add(arr)

def embed_texts(texts):
    model = get_model()
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # normalize
    faiss.normalize_L2(embs)
    return embs

def add_and_index_article(article):
    # article must have id, content (or title+summary)
    text = (article.get("summary") or "") + "\n\n" + (article.get("content") or "")
    emb = embed_texts([text])[0]
    article["embedding"] = emb
    save_article(article)
    # rebuild index (simple, keeps code small)
    build_index()

# ---------- Similarity search ----------
def query_similar(text, k=5):
    emb = embed_texts([text])[0]
    if _index is None:
        build_index()
    if _index.ntotal == 0:
        return []
    D, I = _index.search(np.expand_dims(emb, 0), k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(_ids): continue
        aid = _ids[idx]
        # load metadata
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT id,title,url,source,published,summary,content FROM articles WHERE id=?", (aid,))
        row = cur.fetchone()
        con.close()
        if row:
            results.append({"id":row[0],"title":row[1],"url":row[2],"source":row[3],"published":row[4],"summary":row[5],"content":row[6],"score":float(dist)})
    return results

# ---------- Time-series & topic simple analytics ----------
def count_mentions(keyword, window_days=90, freq='7D'):
    df = load_all_articles()
    if df.empty:
        return pd.Series([], dtype=int)
    df['published'] = pd.to_datetime(df['published'], errors='coerce').fillna(pd.Timestamp.utcnow())
    df['text'] = (df['title'].fillna('') + ' ' + df['summary'].fillna('') + ' ' + df['content'].fillna('')).str.lower()
    df['match'] = df['text'].str.contains(keyword.lower(), na=False)
    series = df.set_index('published').resample(freq)['match'].sum()
    return series

# ---------- LLM helper (OpenAI) ----------
def call_openai_insight(system_prompt, user_prompt):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model":"gpt-4o-mini", "messages":[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], "temperature":0.3, "max_tokens":700}
    r = requests.post(url, headers=headers, json=payload, timeout=40)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()

# ---------- High-level "generate insights" ----------
def generate_insight_for_article(article, k=6):
    """
    Steps:
     - find similar past articles
     - compute simple stats (counts)
     - ask LLM to synthesize patterns and future-looking insight (if API available)
    """
    qtext = (article.get("summary") or "") + "\n\n" + (article.get("content") or "")
    similar = query_similar(qtext, k=k)
    # build evidence text
    evidence = ""
    for s in similar:
        evidence += f"- {s['published']}: {s['title']} ({s['source']})\n  {s['summary'][:250]}\n\n"
    # compute trend example: mention counts for keywords from candidate title words
    # pick keywords (very naive)
    words = [w for w in (article.get("title") or "").split() if len(w)>4][:4]
    counts = {}
    for w in words:
        counts[w] = count_mentions(w, window_days=180, freq='30D').tolist()
    system = "You are an analytics journalist. Use evidence to produce a concise insight (3-6 sentences) that explains how the current story connects to past patterns and suggests near-future implications."
    user = f"Article:\n{article.get('title')}\n\nSummary:\n{article.get('summary')}\n\nSimilar past items:\n{evidence}\n\nKeyword time-series (last windows):\n{json.dumps(counts)}\n\nWrite: 1) TL;DR (one line), 2) 3 bullets about pattern, 3) 1 short prediction (what to watch next).\n"
    try:
        res = call_openai_insight(system, user)
        return res or None
    except Exception as e:
        print("OpenAI insight failed:", e)
        # fallback: build heuristic insight
        tl = (article.get('summary') or "")[:200]
        bullets = ["Trend appears consistent with past coverage.", "Mentions of "+(", ".join(words))+ " increased recently.", "Watch for follow-up in next 2 weeks."]
        pred = "This topic may continue to rise if further events occur."
        out = f"{tl}\n\nHighlights:\n" + "\n".join("- "+b for b in bullets) + "\n\nPrediction:\n"+pred
        return out

# ---------- quick init ----------
if __name__ == "__main__":
    init_db()
    build_index()
    print("analysis ready")
