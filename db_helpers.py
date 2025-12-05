# db_helpers.py
import sqlite3
from typing import Optional

DB_PATH = "elitenotes.db"
DEFAULT_TIMEOUT = 30

def init_db():
    """Create DB and required tables if they don't exist."""
    with sqlite3.connect(DB_PATH, timeout=DEFAULT_TIMEOUT) as con:
        cur = con.cursor()
        # articles table if not already created by other parts of your app
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
        );
        """)
        # seen table to replace seen.json
        cur.execute("""
        CREATE TABLE IF NOT EXISTS seen (
            url_hash TEXT PRIMARY KEY,
            url TEXT,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_seen_first ON seen(first_seen)")
        # cache table for article text / summaries / embeddings
        cur.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            cache_key TEXT PRIMARY KEY,
            value BLOB,
            content_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_ct ON cache(content_type)")
        con.commit()

# ---------- seen helpers ----------
def mark_seen(url_hash: str, url: str = None):
    with sqlite3.connect(DB_PATH, timeout=DEFAULT_TIMEOUT) as con:
        con.execute("INSERT OR IGNORE INTO seen (url_hash, url) VALUES (?, ?)", (url_hash, url))
        con.commit()

def is_seen(url_hash: str) -> bool:
    with sqlite3.connect(DB_PATH, timeout=DEFAULT_TIMEOUT) as con:
        cur = con.execute("SELECT 1 FROM seen WHERE url_hash=? LIMIT 1", (url_hash,))
        return cur.fetchone() is not None

def get_all_seen():
    with sqlite3.connect(DB_PATH, timeout=DEFAULT_TIMEOUT) as con:
        cur = con.execute("SELECT url_hash, url, first_seen FROM seen ORDER BY first_seen DESC")
        return cur.fetchall()

# ---------- cache helpers ----------
def cache_set(key: str, value: bytes, content_type: str = "text"):
    """Store binary value (bytes). For text, encode to utf-8 before passing."""
    with sqlite3.connect(DB_PATH, timeout=DEFAULT_TIMEOUT) as con:
        con.execute("""
            INSERT OR REPLACE INTO cache (cache_key, value, content_type, created_at) 
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (key, value, content_type))
        con.commit()

def cache_get(key: str) -> Optional[bytes]:
    with sqlite3.connect(DB_PATH, timeout=DEFAULT_TIMEOUT) as con:
        cur = con.execute("SELECT value FROM cache WHERE cache_key=? LIMIT 1", (key,))
        row = cur.fetchone()
        return row[0] if row else None

def cache_delete(key: str):
    with sqlite3.connect(DB_PATH, timeout=DEFAULT_TIMEOUT) as con:
        con.execute("DELETE FROM cache WHERE cache_key=?", (key,))
        con.commit()
