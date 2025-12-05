# cache_api.py
import hashlib
from typing import Optional
from db_helpers import cache_get, cache_set

def key_for_url(url: str) -> str:
    return "url:" + hashlib.sha1((url or "").encode("utf-8")).hexdigest()[:16]

def key_for_summary(content_hash: str) -> str:
    return "summary:" + (content_hash or "")[:32]

def get_cached_article_text(url: str) -> Optional[str]:
    key = key_for_url(url)
    b = cache_get(key)
    return b.decode("utf-8") if b else None

def set_cached_article_text(url: str, text: str):
    key = key_for_url(url)
    cache_set(key, text.encode("utf-8"), content_type="article_text")

def get_cached_summary(content_hash: str) -> Optional[str]:
    key = key_for_summary(content_hash)
    b = cache_get(key)
    return b.decode("utf-8") if b else None

def set_cached_summary(content_hash: str, summary_text: str):
    key = key_for_summary(content_hash)
    cache_set(key, summary_text.encode("utf-8"), content_type="summary")
