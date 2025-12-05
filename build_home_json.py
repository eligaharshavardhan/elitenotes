#!/usr/bin/env python3
"""
build_home_json.py

Generate public/data/home.json for the Elitenotes landing page.

It reads from elitenotes.db (articles table) and produces:
- top_insights (from latest briefs)
- briefs (latest N articles)
- features (static list for now, can be made dynamic later)
"""

import os
import json
import sqlite3
from datetime import datetime, timezone

DB_PATH = "elitenotes.db"
OUT_PATH = os.path.join("public", "data", "home.json")
MAX_BRIEFS = 9  # how many cards to show on homepage


def load_latest_articles(limit=MAX_BRIEFS):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, title, summary, published, source
        FROM articles
        ORDER BY datetime(published) DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    con.close()
    return rows


def estimate_read_time(text: str) -> int:
    """Very rough word count → minutes."""
    if not text:
        return 3
    words = len(text.split())
    return max(2, min(8, int(words / 200) + 1))


def build_briefs(rows):
    briefs = []
    for row in rows:
        aid, title, summary, published, source = row
        # Parse/format date
        try:
            dt = datetime.fromisoformat(published)
        except Exception:
            dt = datetime.now(timezone.utc)

        date_str = dt.strftime("%b %e, %Y")  # e.g. "Nov  8, 2025"
        read_time = estimate_read_time(summary or "")

        # This URL pattern should match how generate_brief.py writes previews.
        # If your filenames differ, update this pattern accordingly.
        url = f"/previews/{dt.strftime('%Y-%m-%d')}-{aid}.html"

        briefs.append(
            {
                "title": title or "(untitled)",
                "summary": (summary or "").strip(),
                "date": date_str,
                "read_time": read_time,
                "url": url,
                "source": source or "",
            }
        )
    return briefs


def build_features_static():
    """
    Temporary feature list.
    You can later:
    - Read from another table,
    - Scan public/features/*.html, etc.
    """
    return [
        {
            "title": "The Quiet Revolution: How 10 Mobile Apps Are Rewriting Everyday Life in 2025",
            "summary": "The mobile apps that quietly restructured daily behaviour in 2025 — predictive simplicity, ambient intelligence and the disappearance of 'app' as a concept.",
            "date": "Nov 1, 2025",
            "read_time": 10,
            "url": "/page1.html",  # you already have page1.html
        },
        {
            "title": "Startup Strategy for an AI-First World",
            "summary": "A practical playbook for founders: what changes when models can replace entire teams, and where human leverage still compounds.",
            "date": "Oct 20, 2025",
            "read_time": 12,
            "url": "/page1.html",  # second feature reuses same page for now
        },
    ]


def main():
    rows = load_latest_articles()
    briefs = build_briefs(rows)

    # Top insights = top 3 brief titles (simple heuristic)
    top_insights = [b["title"] for b in briefs[:3]]

    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "top_insights": top_insights,
        "briefs": briefs,
        "features": build_features_static(),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {OUT_PATH} with {len(briefs)} briefs.")


if __name__ == "__main__":
    main()
