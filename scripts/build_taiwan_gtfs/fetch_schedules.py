"""
Step 3: Query the Google Maps Routes API to sample transit departure schedules.

Strategy
--------
For each rail line (THSR, TRA Western Main Line, …):
  1. Identify the two terminal stations (northmost / southmost).
  2. For each non-terminal station on the line, at each sample hour:
       - Query Routes API: station → SOUTH terminal, departure_time = that hour
       - Query Routes API: station → NORTH terminal, departure_time = that hour
  3. Extract from each response:
       departureStop name, arrivalStop name, departureTime, arrivalTime,
       headsign, transitLine name/short_name, vehicle type
  4. Cache every raw API response to avoid re-querying on reruns.

Output
------
  data/taiwan_raw/raw_trips_cache.json  — all raw API responses (cache)
  data/taiwan_raw/raw_trips.json        — list of parsed departure records

Each departure record:
  {
    "network":        "THSR" | "TRA",
    "line":           "<line name>",
    "origin_stop_id": "<slug>",
    "origin_name":    "<Google Maps stop name>",
    "dest_stop_id":   "<slug of terminal>",
    "dest_name":      "<Google Maps terminal name>",
    "departure_time": "HH:MM:SS",    # local Taiwan time
    "arrival_time":   "HH:MM:SS",
    "headsign":       "<headsign text>",
    "route_name":     "<transit line name>",
    "route_short":    "<short name, e.g. '自強'>",
    "vehicle_type":   "HEAVY_RAIL" | "HIGH_SPEED_TRAIN" | ...
  }

Run: python3 scripts/build_taiwan_gtfs/fetch_schedules.py
Requires: GOOGLE_MAPS_API_KEY environment variable
"""

import json
import sys
import time
import pathlib
import datetime
from collections import defaultdict
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import requests

from config import (
    GOOGLE_MAPS_API_KEY,
    STATIONS_GEOCODED_JSON,
    RAW_TRIPS_CACHE_JSON,
    RAW_TRIPS_JSON,
    SAMPLE_HOURS,
    SAMPLE_DATE,
    TAIWAN_TZ,
    API_RATE_LIMIT_RPS,
)

DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

TZ = ZoneInfo(TAIWAN_TZ)
MIN_DELAY = 1.0 / API_RATE_LIMIT_RPS


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _departure_timestamp(hour: int) -> int:
    """Return Unix timestamp for SAMPLE_DATE at `hour` Taiwan local time."""
    year, month, day = map(int, SAMPLE_DATE.split("-"))
    dt_local = datetime.datetime(year, month, day, hour, 0, 0, tzinfo=TZ)
    return int(dt_local.timestamp())


def _call_routes_api(origin_lat: float, origin_lon: float,
                     dest_lat: float, dest_lon: float,
                     departure_ts: int) -> dict:
    """Call the Directions API (v1) and return the raw JSON response."""
    params = {
        "origin":         f"{origin_lat},{origin_lon}",
        "destination":    f"{dest_lat},{dest_lon}",
        "mode":           "transit",
        "transit_mode":   "rail",
        "departure_time": departure_ts,
        "key":            GOOGLE_MAPS_API_KEY,
    }
    resp = requests.get(DIRECTIONS_URL, params=params, timeout=20)
    resp.raise_for_status()
    time.sleep(.100)
    return resp.json()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _extract_departures(raw: dict, origin_stop_id: str,
                         network: str, line: str) -> list[dict]:
    """
    Walk a Directions API v1 response and return departure records for each
    transit step that uses rail.
    """
    records = []
    status = raw.get("status", "")
    if status not in ("OK", "ZERO_RESULTS"):
        print(f"    [API status: {status}]")
    routes = raw.get("routes") or []
    if not routes:
        return records

    for route in routes:
        for leg in route.get("legs", []):
            for step in leg.get("steps", []):
                if step.get("travel_mode") != "TRANSIT":
                    continue
                td = step.get("transit_details")
                if not td:
                    continue

                line_info   = td.get("line", {})
                vehicle     = line_info.get("vehicle", {}).get("type", "")
                headsign    = td.get("headsign", "")
                route_name  = line_info.get("name", "")
                route_short = line_info.get("short_name", "")

                dep_stop = td.get("departure_stop", {})
                arr_stop = td.get("arrival_stop", {})

                # Times come back as {"text": "...", "value": <unix_ts>, "time_zone": "..."}
                dep_local = _unix_to_local_hms(td.get("departure_time", {}).get("value"))
                arr_local = _unix_to_local_hms(td.get("arrival_time", {}).get("value"))

                if not dep_local or not arr_local:
                    continue

                records.append({
                    "network":        network,
                    "line":           line,
                    "origin_stop_id": origin_stop_id,
                    "origin_name":    dep_stop.get("name", ""),
                    "dest_name":      arr_stop.get("name", ""),
                    "departure_time": dep_local,
                    "arrival_time":   arr_local,
                    "headsign":       headsign,
                    "route_name":     route_name,
                    "route_short":    route_short,
                    "vehicle_type":   vehicle,
                })
                # Take only the first transit step per route
                break

    return records


def _unix_to_local_hms(unix_ts) -> str:
    """Convert a Unix timestamp (int/float) to HH:MM:SS in Taiwan local time."""
    if unix_ts is None:
        return ""
    try:
        dt = datetime.datetime.fromtimestamp(int(unix_ts), tz=TZ)
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_stations_by_line() -> dict[str, list[dict]]:
    """Return {line_key: [stations in order]} from the geocoded stations file."""
    with open(STATIONS_GEOCODED_JSON, encoding="utf-8") as f:
        stations = json.load(f)

    by_line: dict[str, list[dict]] = defaultdict(list)
    for s in stations:
        key = (s["network"], s["line"])
        by_line[key].append(s)

    # Sort each line by station order
    for key in by_line:
        by_line[key].sort(key=lambda s: s.get("order", 0))

    return dict(by_line)


def load_cache() -> dict:
    if RAW_TRIPS_CACHE_JSON.exists():
        with open(RAW_TRIPS_CACHE_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(RAW_TRIPS_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def main():
    if not GOOGLE_MAPS_API_KEY:
        print("ERROR: GOOGLE_MAPS_API_KEY environment variable is not set.")
        sys.exit(1)

    lines = load_stations_by_line()
    cache = load_cache()

    all_records: list[dict] = []
    total_queries = 0
    cache_hits    = 0

    for (network, line), stations in lines.items():
        if len(stations) < 2:
            print(f"  Skipping {network}/{line}: fewer than 2 stations.")
            continue

        north_terminal = stations[0]
        south_terminal = stations[-1]
        print(f"\n=== {network} / {line} ({len(stations)} stations) ===")
        print(f"  North terminal: {north_terminal['name']}")
        print(f"  South terminal: {south_terminal['name']}")

        for station in stations:
            name    = station["name"]
            s_id    = station["stop_id"]
            s_lat   = station["stop_lat"]
            s_lon   = station["stop_lon"]

            if s_lat == 0.0 and s_lon == 0.0:
                print(f"  Skipping {name}: no geocode.")
                continue

            # Query to south terminal (unless this IS the south terminal)
            for terminal, direction in [
                (south_terminal, "S"),
                (north_terminal, "N"),
            ]:
                if station["stop_id"] == terminal["stop_id"]:
                    continue  # Don't query terminal → itself
                t_lat = terminal["stop_lat"]
                t_lon = terminal["stop_lon"]
                if t_lat == 0.0 and t_lon == 0.0:
                    continue

                for hour in SAMPLE_HOURS:
                    dep_ts = _departure_timestamp(hour)
                    cache_key = f"{s_id}__{terminal['stop_id']}__{dep_ts}"

                    if cache_key in cache:
                        raw = cache[cache_key]
                        cache_hits += 1
                    else:
                        print(
                            f"  Querying {name}→{terminal['name']} "
                            f"at {hour:02d}:00 ({direction}) ...",
                            end=" ", flush=True,
                        )
                        try:
                            raw = _call_routes_api(s_lat, s_lon, t_lat, t_lon, dep_ts)
                            cache[cache_key] = raw
                            save_cache(cache)
                            total_queries += 1
                            print("OK", raw)
                        except Exception as e:
                            print(f"ERROR: {e}")
                            input()
                            continue
                        time.sleep(MIN_DELAY)

                    records = _extract_departures(raw, s_id, network, line)
                    all_records.extend(records)

    with open(RAW_TRIPS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  API queries made: {total_queries}")
    print(f"  Cache hits:       {cache_hits}")
    print(f"  Departure records:{len(all_records)}")
    print(f"  Wrote {RAW_TRIPS_JSON}")


if __name__ == "__main__":
    main()
