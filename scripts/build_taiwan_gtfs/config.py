"""
Configuration for Taiwan GTFS builder.
Set GOOGLE_MAPS_API_KEY in your environment before running any step that hits the API.
"""

import os

# --- API ---
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# --- Paths ---
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.parent  # repo root

RAW_DIR = ROOT / "data" / "taiwan_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TRA_GTFS_DIR  = ROOT / "data" / "tra_gtfs"
THSR_GTFS_DIR = ROOT / "data" / "thsr_gtfs"

STATIONS_JSON         = RAW_DIR / "stations.json"
STATIONS_GEOCODED_JSON = RAW_DIR / "stations_geocoded.json"
GEOCODE_CACHE_JSON    = RAW_DIR / "geocode_cache.json"
RAW_TRIPS_CACHE_JSON  = RAW_DIR / "raw_trips_cache.json"
RAW_TRIPS_JSON        = RAW_DIR / "raw_trips.json"

# --- Scraping ---
WIKIPEDIA_URL = (
    "https://en.wikipedia.org/wiki/"
    "List_of_railway_and_metro_stations_in_Taiwan"
)

# --- Schedule sampling ---
# Hours (local Taiwan time, UTC+8) to query departures for
SAMPLE_HOURS = list(range(6, 24))       # 06:00–23:00

# Reference date for API queries (a weekday, YYYY-MM-DD, must be in the future)
SAMPLE_DATE = "2026-04-01"             # Wednesday

# Timezone for departure times
TAIWAN_TZ = "Asia/Taipei"             # UTC+8

# Max RPS to the Routes API (Google allows up to 50 QPS but be conservative)
API_RATE_LIMIT_RPS = 2.0

# --- GTFS calendar ---
CALENDAR_START = "20250101"
CALENDAR_END   = "20251231"

# --- THSR stations in N→S order (hardcoded — stable, only 12 stations) ---
THSR_STATIONS = [
    "Nangang",
    "Taipei",
    "Banqiao",
    "Taoyuan",
    "Hsinchu",
    "Miaoli",
    "Taichung",
    "Changhua",
    "Yunlin",
    "Chiayi",
    "Tainan",
    "Zuoying",
]
