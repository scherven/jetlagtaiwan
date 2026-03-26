"""
Test run: build a GTFS feed from a small subset of ~10 stations.

Uses 5 THSR + 5 TRA stations to validate the full pipeline end-to-end
without burning through API quota.

Output goes to data/taiwan_test/ so it won't overwrite the real feeds.

Usage:
  export GOOGLE_MAPS_API_KEY="AIza..."
  python3 scripts/build_taiwan_gtfs/run_test.py
"""

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import config as _config

# --- Override config paths to point at test directories ---
RAW_DIR_TEST  = _config.ROOT / "data" / "taiwan_test_raw"
OUT_DIR_TEST  = _config.ROOT / "data" / "taiwan_test_gtfs"
RAW_DIR_TEST.mkdir(parents=True, exist_ok=True)
OUT_DIR_TEST.mkdir(parents=True, exist_ok=True)

_config.RAW_DIR                = RAW_DIR_TEST
_config.STATIONS_JSON          = RAW_DIR_TEST / "stations.json"
_config.STATIONS_GEOCODED_JSON = RAW_DIR_TEST / "stations_geocoded.json"
_config.GEOCODE_CACHE_JSON     = RAW_DIR_TEST / "geocode_cache.json"
_config.RAW_TRIPS_CACHE_JSON   = RAW_DIR_TEST / "raw_trips_cache.json"
_config.RAW_TRIPS_JSON         = RAW_DIR_TEST / "raw_trips.json"
_config.TRA_GTFS_DIR          = OUT_DIR_TEST / "tra_gtfs"
_config.THSR_GTFS_DIR         = OUT_DIR_TEST / "thsr_gtfs"

# Only sample 3 hours instead of 18 to keep API calls minimal
_config.SAMPLE_HOURS = [8, 12, 17]

# --- Test station subset ---
TEST_STATIONS = [
    # THSR: 5 stations (full N→S spread)
    {"name": "Nangang",   "network": "THSR", "line": "THSR Main Line", "order": 1},
    {"name": "Taipei",    "network": "THSR", "line": "THSR Main Line", "order": 2},
    {"name": "Taichung",  "network": "THSR", "line": "THSR Main Line", "order": 7},
    {"name": "Tainan",    "network": "THSR", "line": "THSR Main Line", "order": 11},
    {"name": "Zuoying",   "network": "THSR", "line": "THSR Main Line", "order": 12},
    # TRA: 5 stations along the Western Main Line
    {"name": "Taipei",    "network": "TRA",  "line": "Western Main Line", "order": 6},
    {"name": "Taoyuan",   "network": "TRA",  "line": "Western Main Line", "order": 14},
    {"name": "Hsinchu",   "network": "TRA",  "line": "Western Main Line", "order": 22},
    {"name": "Taichung",  "network": "TRA",  "line": "Western Main Line", "order": 31},
    {"name": "Kaohsiung", "network": "TRA",  "line": "Western Main Line", "order": 46},
]


def step1_write_stations():
    print("\n=== Step 1: Write test station list ===")
    with open(_config.STATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(TEST_STATIONS, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {len(TEST_STATIONS)} stations to {_config.STATIONS_JSON}")


def step2_geocode():
    print("\n=== Step 2: Geocode stations ===")
    import geocode_stations
    geocode_stations.main()


def step3_fetch():
    print("\n=== Step 3: Fetch schedules ===")
    import fetch_schedules
    fetch_schedules.main()


def step4_build():
    print("\n=== Step 4: Build GTFS ===")
    import build_gtfs
    # Override the output directories inside the module
    build_gtfs.TRA_GTFS_DIR  = _config.TRA_GTFS_DIR
    build_gtfs.THSR_GTFS_DIR = _config.THSR_GTFS_DIR
    build_gtfs.main()


def display_raw_trips():
    """Pretty-print every parsed departure record from the raw trips JSON."""
    print("\n=== Raw departure records (pre-GTFS) ===")
    if not _config.RAW_TRIPS_JSON.exists():
        print("  raw_trips.json not found — did step 3 run?")
        return

    with open(_config.RAW_TRIPS_JSON, encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        print("  No departure records found.")
        return

    # Group by network → line → origin station
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[r["network"]][r["line"]].append(r)

    for network, lines in sorted(grouped.items()):
        print(f"\n  ┌─ {network}")
        for line, recs in sorted(lines.items()):
            # Sub-group by origin stop
            by_origin = defaultdict(list)
            for r in recs:
                by_origin[r["origin_name"] or r["origin_stop_id"]].append(r)

            print(f"  │  └─ {line} ({len(recs)} departures across {len(by_origin)} stops)")
            for origin, deps in sorted(by_origin.items()):
                deps_sorted = sorted(deps, key=lambda d: d["departure_time"])
                print(f"  │       {origin}:")
                for d in deps_sorted:
                    print(
                        f"  │         {d['departure_time']} → {d['dest_name']:30s}"
                        f"  arr {d['arrival_time']}"
                        f"  [{d['route_short'] or d['route_name'] or d['vehicle_type']}]"
                        f"  headsign: {d['headsign']}"
                    )

    print(f"\n  Total: {len(records)} departure records")


def verify():
    print("\n=== Verify: load GTFS with RailNetwork ===")
    import sys
    sys.path.insert(0, str(_config.ROOT))
    from engine.rail_network import RailNetwork

    feeds = []
    for d in [_config.THSR_GTFS_DIR, _config.TRA_GTFS_DIR]:
        if (d / "stops.txt").exists():
            feeds.append(str(d))

    if not feeds:
        print("  No GTFS output found — did step 4 produce any files?")
        return

    net = RailNetwork(feeds, merge_strategy="name")
    print(f"  Stations: {len(net.stations)}")
    total_departures = sum(len(v) for v in net.schedules.values())
    print(f"  Total departure records: {total_departures}")

    print()
    for sid, node in sorted(net.stations.items(), key=lambda x: x[1].name):
        deps = sorted(net.schedules.get(sid, []), key=lambda d: d.departure_minute)
        if not deps:
            print(f"  {node.name:30s} ({sid})  — no departures")
            continue
        print(f"  {node.name:30s} ({sid})  {len(deps)} departures:")
        for dep in deps:
            dest_name = net.stations[dep.destination_stop_id].name if dep.destination_stop_id in net.stations else dep.destination_stop_id
            dep_hhmm  = f"{dep.departure_minute // 60:02d}:{dep.departure_minute % 60:02d}"
            # print(dep, dep.arrival_minutes)
            arr_mins  = dep.arrival_minutes[dep.intermediate_stops.index(dep.destination_stop_id)]#.get(dep.destination_stop_id, dep.departure_minute)
            arr_hhmm  = f"{arr_mins // 60:02d}:{arr_mins % 60:02d}"
            print(f"    {dep_hhmm} → {dest_name:30s}  arr {arr_hhmm}  route: {dep.route_id}")


if __name__ == "__main__":
    if not _config.GOOGLE_MAPS_API_KEY:
        print("ERROR: GOOGLE_MAPS_API_KEY environment variable is not set.")
        sys.exit(1)

    step1_write_stations()
    step2_geocode()
    step3_fetch()
    display_raw_trips()   # show parsed departures before GTFS conversion
    step4_build()
    verify()              # show final GTFS-loaded network

    print(f"\nTest complete. Output in {OUT_DIR_TEST}")
    print("Expected API calls: ~(10 stations × 2 directions × 3 hours) = ~60 queries")
