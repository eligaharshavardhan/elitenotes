# migrate_seen_to_sqlite.py
import json, os
from db_helpers import init_db, mark_seen

SEEN_FILE = "seen.json"

def migrate():
    init_db()
    if not os.path.exists(SEEN_FILE):
        print("No seen.json found — nothing to migrate.")
        return
    with open(SEEN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    count = 0
    for h in data:
        mark_seen(h, None)
        count += 1
    print(f"Imported {count} hashes into seen table.")

if __name__ == "__main__":
    migrate()
