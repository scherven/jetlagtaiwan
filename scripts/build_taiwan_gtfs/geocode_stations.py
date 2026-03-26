"""
Step 2: Geocode each station via the Google Maps Geocoding API.

Reads  data/taiwan_raw/stations.json
Writes data/taiwan_raw/stations_geocoded.json

Each output record extends the input with:
  stop_id  : URL-safe slug  (e.g. "taipei_thsr")
  stop_lat : float
  stop_lon : float
  place_id : Google Maps place_id (used in Routes API queries)

Stations that fail geocoding are kept with lat/lon = 0.0 and a warning printed.

Run: python3 scripts/build_taiwan_gtfs/geocode_stations.py
Requires: GOOGLE_MAPS_API_KEY environment variable
"""

import json
import re
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import requests

from config import (
    GOOGLE_MAPS_API_KEY,
    STATIONS_JSON,
    STATIONS_GEOCODED_JSON,
    GEOCODE_CACHE_JSON,
    API_RATE_LIMIT_RPS,
)

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def load_cache() -> dict:
    if GEOCODE_CACHE_JSON.exists():
        with open(GEOCODE_CACHE_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(GEOCODE_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def slugify(name: str, network: str) -> str:
    """Convert station name + network to a stable stop_id slug."""
    s = f"{name}_{network}".lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def geocode_station(name: str, network: str, cache: dict) -> dict:
    """
    Query the Geocoding API for a Taiwan rail station (with cache).
    Returns {"stop_lat": float, "stop_lon": float, "place_id": str} or zeros on failure.
    """
    query_map = {
        "THSR": f"{name} High Speed Rail station Taiwan",
        "TRA":  f"{name} Taiwan Railway station Taiwan",
    }
    query = query_map.get(network, f"{name} railway station Taiwan")
    cache_key = f"{network}__{name}"

    if cache_key in cache:
        return cache[cache_key]

    params = {
        "address": query,
        "region":  "tw",
        "key":     GOOGLE_MAPS_API_KEY,
    }
    resp = requests.get(GEOCODE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "OK" or not data.get("results"):
        print(f"  WARNING: Geocoding failed for '{name}' ({network}): {data.get('status')}")
        print(data)
        input()
        result = {"stop_lat": 0.0, "stop_lon": 0.0, "place_id": ""}
    else:
        loc = data["results"][0]["geometry"]["location"]
        result = {
            "stop_lat": loc["lat"],
            "stop_lon": loc["lng"],
            "place_id": data["results"][0].get("place_id", ""),
        }

    cache[cache_key] = result
    save_cache(cache)
    return result


def main():
    if not GOOGLE_MAPS_API_KEY:
        print("ERROR: GOOGLE_MAPS_API_KEY environment variable is not set.")
        sys.exit(1)

    with open(STATIONS_JSON, encoding="utf-8") as f:
        stations = json.load(f)

    print(f"Geocoding {len(stations)} stations ...")
    min_delay = 1.0 / API_RATE_LIMIT_RPS
    cache = load_cache()

    geocoded = []
    for i, station in enumerate(stations):
        name    = station["name"]
        network = station["network"]
        stop_id = slugify(name, network)

        cache_key = f"{network}__{name}"
        was_cached = cache_key in cache

        geo = geocode_station(name, network, cache)

        record = {**station, "stop_id": stop_id, **geo}
        geocoded.append(record)

        status = f"({geo['stop_lat']:.4f}, {geo['stop_lon']:.4f})" if geo["stop_lat"] else "FAILED"
        source = "cached" if was_cached else "API"
        print(f"  [{i+1}/{len(stations)}] {name} ({network}) → {status} [{source}]")

        if not was_cached:
            time.sleep(min_delay)

    with open(STATIONS_GEOCODED_JSON, "w", encoding="utf-8") as f:
        json.dump(geocoded, f, ensure_ascii=False, indent=2)

    succeeded = sum(1 for s in geocoded if s["stop_lat"] != 0.0)
    print(f"\nWrote {len(geocoded)} records to {STATIONS_GEOCODED_JSON}")
    print(f"  Succeeded: {succeeded}  Failed: {len(geocoded) - succeeded}")


if __name__ == "__main__":
    main()
