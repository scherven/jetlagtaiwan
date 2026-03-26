"""
Step 1: Scrape Taiwan railway station names from Wikipedia.

Produces data/taiwan_raw/stations.json — a list of station records:
  {
    "name":    "Taipei",
    "network": "THSR" | "TRA",
    "line":    "<line name>",
    "order":   <int, position along line starting from 1>
  }

For THSR the 12 stations are hardcoded (stable).
For TRA the script parses the Wikipedia table for the Western Main Line and
other trunk lines, best-effort.

Run: python3 scripts/build_taiwan_gtfs/scrape_stations.py
"""

import json
import sys
import re
import pathlib

# Allow running from repo root or from this directory
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import requests
from bs4 import BeautifulSoup

from config import STATIONS_JSON, THSR_STATIONS, WIKIPEDIA_URL


def scrape_tra_stations() -> list[dict]:
    """
    Fetch the Wikipedia page and parse the Taiwan Railway (TRA) station table.
    Returns a list of station dicts sorted by line then order.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; TaiwanGTFSBuilder/1.0; "
            "+https://github.com/example/jetlagtaiwan)"
        )
    }
    print(f"Fetching {WIKIPEDIA_URL} ...")
    resp = requests.get(WIKIPEDIA_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Locate the "Taiwan Railway" anchor / section heading
    tra_heading = soup.find(id="Taiwan_Railway") or soup.find(
        lambda tag: tag.name in ("h2", "h3")
        and "Taiwan Railway" in tag.get_text()
    )
    if tra_heading is None:
        print("WARNING: Could not find 'Taiwan Railway' section. Trying full-page table scan.")
        return _fallback_station_list()

    # Walk siblings until we find the first wikitable
    section_tables = []
    for sibling in tra_heading.parent.next_siblings:
        if sibling.name in ("h2", "h3"):
            break
        if sibling.name == "table" and "wikitable" in sibling.get("class", []):
            section_tables.append(sibling)

    if not section_tables:
        print("WARNING: No wikitable found in TRA section. Using fallback list.")
        return _fallback_station_list()

    stations = []
    for table in section_tables:
        rows = table.find_all("tr")
        if not rows:
            continue

        # Detect header row to find "Station" and "Line" columns
        header_row = rows[0]
        headers_text = [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]

        station_col = next(
            (i for i, h in enumerate(headers_text) if "station" in h or "name" in h), 0
        )
        line_col = next(
            (i for i, h in enumerate(headers_text) if "line" in h or "route" in h), None
        )

        line_name = _infer_line_name_from_table(table)
        order = 0
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) <= station_col:
                continue
            name_raw = cells[station_col].get_text(strip=True)
            # Strip footnote markers like [1] and parentheticals
            name = re.sub(r"\[.*?\]", "", name_raw).split("(")[0].strip()
            if not name or name.lower() in ("station", "name", ""):
                continue

            inferred_line = line_name
            if line_col is not None and line_col < len(cells):
                line_text = cells[line_col].get_text(strip=True)
                if line_text:
                    inferred_line = line_text

            order += 1
            stations.append({
                "name": name,
                "network": "TRA",
                "line": inferred_line,
                "order": order,
            })

    if not stations:
        print("WARNING: Parsed 0 TRA stations from Wikipedia. Using fallback list.")
        return _fallback_station_list()

    print(f"  Parsed {len(stations)} TRA station entries from Wikipedia.")
    return stations


def _infer_line_name_from_table(table) -> str:
    """Try to get a line name from a caption or preceding heading."""
    caption = table.find("caption")
    if caption:
        return caption.get_text(strip=True)
    prev = table.find_previous(["h2", "h3", "h4"])
    if prev:
        return prev.get_text(strip=True)
    return "TRA Main Line"


def _fallback_station_list() -> list[dict]:
    """
    Hardcoded TRA Western Main Line stations (N→S) as a fallback when
    Wikipedia parsing fails.  This is representative enough for the game.
    """
    western_main = [
        "Keelung", "Badu", "Nuannuan", "Sidu", "Banqiao", "Taipei",
        "Songshan", "Nangang", "Xizhi", "Sijhih", "Ruifang",
        "Hualien",  # via Yilan line continues but this covers the key stretch
        # Western Main Line proper S from Taipei:
        "Wanhua", "Shulin", "Yingge", "Taoyuan", "Zhongli", "Nei-Li",
        "Yangmei", "Fengzhong", "Beihu", "Hukou", "Xinpu", "Zhubei",
        "Hsinchu", "Qiandong", "Xiangshan", "Zhunan",
        "Toufen", "Miaoli", "Houlong", "Sanyi",
        "Taichung", "Changhua", "Yuanlin", "Tianjhong", "Ershui",
        "Linnei", "Douliu", "Dachang", "Huwei", "Taixi",
        "Xinying", "Chiayi", "Shalun", "Tainan", "Yongkang",
        "Zuoying", "Kaohsiung",
    ]
    return [
        {"name": s, "network": "TRA", "line": "Western Main Line", "order": i + 1}
        for i, s in enumerate(western_main)
    ]


def build_thsr_stations() -> list[dict]:
    """Return the hardcoded THSR station list."""
    from config import THSR_STATIONS
    return [
        {"name": s, "network": "THSR", "line": "THSR Main Line", "order": i + 1}
        for i, s in enumerate(THSR_STATIONS)
    ]


def main():
    tra = scrape_tra_stations()
    thsr = build_thsr_stations()

    all_stations = thsr + tra

    # Deduplicate by (name, network, line) — Wikipedia tables sometimes repeat headers
    seen = set()
    deduped = []
    for s in all_stations:
        key = (s["name"].lower(), s["network"], s["line"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    with open(STATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(deduped)} stations to {STATIONS_JSON}")
    thsr_count = sum(1 for s in deduped if s["network"] == "THSR")
    tra_count  = sum(1 for s in deduped if s["network"] == "TRA")
    print(f"  THSR: {thsr_count}  TRA: {tra_count}")


if __name__ == "__main__":
    main()
