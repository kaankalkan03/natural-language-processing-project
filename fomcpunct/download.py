"""Download FOMC Chair press conference transcripts (raw PDFs).

Finds every press conference listed on the Federal Reserve Board's FOMC
calendar page (recent years) and historical year pages (older years),
downloads the transcript PDF and writes data/raw/manifest.csv.

Usage:
    python -m fomcpunct.download [--first-year 2011] [--out data/raw]

Files that already exist are not downloaded again; their hash is recomputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://www.federalreserve.gov"
CALENDAR_URL = f"{BASE}/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL = f"{BASE}/monetarypolicy/fomchistorical{{year}}.htm"
PAGE_URL = f"{BASE}/monetarypolicy/fomcpresconf{{date}}.htm"
PDF_URL = f"{BASE}/mediacenter/files/FOMCpresconf{{date}}.pdf"
HEADERS = {"User-Agent": "Mozilla/5.0 (university NLP course project; transcript download)"}
# "fomcpressconf" also occurs on the calendar page (January 2026 link).
DATE_IN_LINK = re.compile(r"fomcpress?conf(\d{8})\.(?:htm|pdf)", re.IGNORECASE)
PDF_IN_PAGE = re.compile(r'href="([^"]*FOMCpresconf\d{8}\.pdf)"', re.IGNORECASE)
DELAY_SECONDS = 1.0


def get(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, headers=HEADERS, timeout=60)
    time.sleep(DELAY_SECONDS)
    return response


def find_dates(session: requests.Session, first_year: int, last_year: int) -> list[str]:
    """Collect press conference dates (YYYYMMDD) from calendar and historical pages."""
    dates: set[str] = set()
    calendar = get(session, CALENDAR_URL)
    calendar.raise_for_status()
    dates.update(DATE_IN_LINK.findall(calendar.text))
    found_years = {d[:4] for d in dates}
    for year in range(first_year, last_year + 1):
        if str(year) in found_years:
            continue
        page = get(session, HISTORICAL_URL.format(year=year))
        if page.status_code == 404:
            print(f"  no historical page for {year}", file=sys.stderr)
            continue
        page.raise_for_status()
        year_dates = {d for d in DATE_IN_LINK.findall(page.text) if d.startswith(str(year))}
        print(f"  {year}: {len(year_dates)} press conferences")
        dates.update(year_dates)
    return sorted(d for d in dates if first_year <= int(d[:4]) <= last_year)


def pdf_url_for(session: requests.Session, date: str) -> str:
    """Return the transcript PDF URL; fall back to the link on the HTML page."""
    url = PDF_URL.format(date=date)
    head = session.head(url, headers=HEADERS, timeout=60, allow_redirects=True)
    time.sleep(DELAY_SECONDS)
    if head.status_code == 200:
        return url
    page = get(session, PAGE_URL.format(date=date))
    page.raise_for_status()
    match = PDF_IN_PAGE.search(page.text)
    if not match:
        raise RuntimeError(f"no transcript PDF link on press conference page {date}")
    link = match.group(1)
    return link if link.startswith("http") else BASE + link


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first-year", type=int, default=2011)
    parser.add_argument("--last-year", type=int, default=datetime.now().year)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    pdf_dir = args.out / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"
    previous = {}
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            previous = {row["date"]: row for row in csv.DictReader(handle)}

    session = requests.Session()
    print("Finding press conferences ...")
    dates = find_dates(session, args.first_year, args.last_year)
    print(f"Found {len(dates)} press conferences from {dates[0]} to {dates[-1]}")

    rows = []
    for date in dates:
        path = pdf_dir / f"FOMCpresconf{date}.pdf"
        if path.exists() and date in previous:
            row = dict(previous[date])
            if row["sha256"] != sha256(path):
                raise RuntimeError(f"{path} changed since it was downloaded")
        else:
            url = pdf_url_for(session, date)
            response = get(session, url)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise RuntimeError(f"{url} did not return a PDF")
            path.write_bytes(response.content)
            row = {
                "date": date,
                "pdf_url": url,
                "file": path.relative_to(args.out).as_posix(),
                "bytes": str(len(response.content)),
                "sha256": sha256(path),
                "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            print(f"  downloaded {path.name} ({len(response.content):,} bytes)")
        rows.append(row)

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    total = sum(int(r["bytes"]) for r in rows)
    print(f"Wrote {manifest_path} ({len(rows)} PDFs, {total / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
