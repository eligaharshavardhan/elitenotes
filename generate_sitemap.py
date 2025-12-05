# generate_sitemap.py
# Scans public/previews/*.html and writes public/sitemap.xml with lastmod timestamps.

import os, time
from urllib.parse import urljoin

# Replace SITE_URL with your production site domain (no trailing slash)
SITE_URL = os.getenv("SITE_URL", "https://yourdomain.com").rstrip("/")

PREVIEWS_DIR = os.path.join("public", "previews")
OUT_PATH = os.path.join("public", "sitemap.xml")

if not os.path.isdir(PREVIEWS_DIR):
    print("No previews directory found at:", PREVIEWS_DIR)
    raise SystemExit(1)

entries = []
files = sorted([f for f in os.listdir(PREVIEWS_DIR) if f.endswith(".html")])
for fname in files:
    rel_path = f"previews/{fname}".replace("\\", "/")
    url = urljoin(SITE_URL + "/", rel_path)
    filepath = os.path.join(PREVIEWS_DIR, fname)
    mtime = time.gmtime(os.path.getmtime(filepath))
    lastmod = time.strftime("%Y-%m-%dT%H:%M:%SZ", mtime)
    entries.append(f"  <url>\n    <loc>{url}</loc>\n    <lastmod>{lastmod}</lastmod>\n    <changefreq>daily</changefreq>\n    <priority>0.6</priority>\n  </url>")

sitemap = "<?xml version='1.0' encoding='UTF-8'?>\n" \
          "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>\n" \
          + "\n".join(entries) + "\n</urlset>\n"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(sitemap)

print("Generated sitemap:", OUT_PATH, "| URLs:", len(entries))
