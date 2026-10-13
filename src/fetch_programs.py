#!/usr/bin/env python3
"""
Fetch Lithuanian study program data from the Lithuanian Open Data Portal.

Data source: Nacionalinė švietimo agentūra (NŠA) — SMPKR Register
API: Spinta REST API at https://get.data.gov.lt
Endpoint: datasets/gov/nsa/smpkr/studiju_mokymu_programu_duomenys/StudijuPrograma

No authentication required. Public, read-only API.

Usage:
    cd /Users/namankumar/Work/Thesis
    source venv/bin/activate
    python src/fetch_programs.py
"""

import requests
import json
import time
import os
import ssl
import certifi
from datetime import datetime

# Fix SSL certificates for macOS Homebrew Python
# (Homebrew Python 3.10 doesn't use macOS system certificates by default)
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# === Configuration ===
API_BASE = "https://get.data.gov.lt/datasets/gov/nsa/smpkr/studiju_mokymu_programu_duomenys/StudijuPrograma"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "study_programs_raw.json")
PAGE_SIZE = 100  # Records per request
REQUEST_DELAY = 0.5  # Seconds between requests (polite rate limiting)


def fetch_all_programs():
    """
    Fetch all study program records using cursor-based pagination.
    
    The Spinta API uses cursor-based pagination:
    - Each response includes a '_page.next' token
    - Pass this token in the next request to get the next page
    - When '_page.next' is absent, all records have been fetched
    """
    all_records = []
    page_cursor = None
    page_num = 0
    max_retries = 3

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting data fetch from:")
    print(f"  {API_BASE}")
    print(f"  Page size: {PAGE_SIZE}")
    print()

    while True:
        page_num += 1

        # Build request URL with proper encoding
        params = {}
        url = API_BASE
        if page_cursor:
            # Use query parameter style for cursor
            url = f"{API_BASE}?limit({PAGE_SIZE})&page('{page_cursor}')"
        else:
            url = f"{API_BASE}?limit({PAGE_SIZE})"

        # Retry logic with exponential backoff
        response = None
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url,
                    headers={"Accept": "application/json"},
                    timeout=30
                )
                if response.status_code == 500 and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  ⚠ Server error on page {page_num}, "
                          f"retry {attempt + 1}/{max_retries} "
                          f"(waiting {wait_time}s)...")
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  ⚠ Error on page {page_num}: {e}")
                    print(f"    Retry {attempt + 1}/{max_retries} "
                          f"(waiting {wait_time}s)...")
                    time.sleep(wait_time)
                else:
                    print(f"  ✗ Failed on page {page_num} after "
                          f"{max_retries} attempts: {e}")
                    if all_records:
                        print(f"  → Saving {len(all_records)} records "
                              f"fetched so far...")
                    return all_records

        data = response.json()
        records = data.get("_data", [])

        if not records:
            print(f"  → No more records. Done!")
            break

        all_records.extend(records)
        print(f"  Page {page_num:3d}: fetched {len(records):3d} records "
              f"(total: {len(all_records):,})")

        # Check for next page cursor
        page_info = data.get("_page", {})
        page_cursor = page_info.get("next")

        if not page_cursor:
            print(f"  → Last page reached. Done!")
            break

        time.sleep(REQUEST_DELAY)

    return all_records


def save_records(records, filepath):
    """Save records to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\n  ✓ Saved {len(records):,} records to: {filepath}")
    print(f"  ✓ File size: {file_size_mb:.1f} MB")


def print_summary(records):
    """Print a quick summary of the fetched data."""
    print(f"\n{'='*60}")
    print(f"  DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Total records:          {len(records):,}")

    # Count by program level (Lithuanian)
    levels = {}
    for r in records:
        level = r.get("prg_lygmuo_lt", "Unknown") or "Unknown"
        levels[level] = levels.get(level, 0) + 1
    
    print(f"\n  Programs by level:")
    for level, count in sorted(levels.items(), key=lambda x: -x[1]):
        print(f"    {level}: {count}")

    # Count by study field
    fields = {}
    for r in records:
        field = r.get("stud_sritis_lt", "Unknown") or "Unknown"
        fields[field] = fields.get(field, 0) + 1

    print(f"\n  Top 10 study fields:")
    for field, count in sorted(fields.items(), key=lambda x: -x[1])[:10]:
        print(f"    {field}: {count}")

    # Count with English names
    has_en = sum(1 for r in records if r.get("prg_pavad_en"))
    has_lt = sum(1 for r in records if r.get("prg_pavad_lt"))
    print(f"\n  Records with Lithuanian title: {has_lt:,}")
    print(f"  Records with English title:    {has_en:,}")
    
    # Count active vs deregistered
    active = sum(1 for r in records if not r.get("prg_isreg_data"))
    print(f"\n  Active programs:      {active:,}")
    print(f"  Deregistered:         {len(records) - active:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Lithuanian Study Program Data Fetcher")
    print("  Source: NŠA / SMPKR via data.gov.lt Spinta API")
    print("=" * 60)
    print()

    records = fetch_all_programs()
    save_records(records, OUTPUT_FILE)
    print_summary(records)

    print(f"\nNext steps:")
    print(f"  1. Review the data: python3 -c \"import json; d=json.load(open('{OUTPUT_FILE}')); print(json.dumps(d[0], indent=2, ensure_ascii=False))\"")
    print(f"  2. Run preprocessing: python src/preprocess.py")
