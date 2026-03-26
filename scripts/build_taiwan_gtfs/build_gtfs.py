"""
Step 4: Build GTFS files from raw departure records.

Reads
  data/taiwan_raw/stations_geocoded.json
  data/taiwan_raw/raw_trips.json

Writes (per network)
  data/tra_gtfs/   → stops.txt, trips.txt, stop_times.txt, routes.txt, agency.txt, calendar.txt
  data/thsr_gtfs/  → same

Trip stitching
--------------
The Routes API only returns the boarding and alighting stop of each transit leg,
not all intermediate stops.  We reconstruct full trips by:

  1. Grouping departure records by (route_name, headsign, arrival_time_at_terminal).
  2. Each group represents one trip: its members are departures from different stations
     that all belong to the same physical train run.
  3. Sorting members by their station's `order` field (position along line).
  4. The resulting ordered sequence gives stop_times with known departure/arrival times
     at each stop where we happened to have a record.
  5. Gaps (stations between recorded stops) are filled by linear interpolation of minutes.

GTFS output conventions
-----------------------
  - service_id = "1"  (one service pattern: every day)
  - route_type = 2    (rail) for TRA, 101 (high speed) for THSR
  - All times in HH:MM:SS, may exceed 24:00:00 for overnight (not expected here)
  - location_type = 1 for all stops (station-level only, no platforms)
  - parent_station = ""

Run: python3 scripts/build_taiwan_gtfs/build_gtfs.py
(No API key needed — entirely offline.)
"""

import csv
import json
import re
import sys
import pathlib
from collections import defaultdict
from itertools import groupby

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from config import (
    STATIONS_GEOCODED_JSON,
    RAW_TRIPS_JSON,
    TRA_GTFS_DIR,
    THSR_GTFS_DIR,
    CALENDAR_START,
    CALENDAR_END,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hms_to_minutes(hms: str) -> float:
    """Convert HH:MM:SS to minutes since midnight (float)."""
    parts = hms.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 60 + m + s / 60.0


def minutes_to_hms(minutes: float) -> str:
    """Convert minutes since midnight back to HH:MM:SS."""
    total_sec = round(minutes * 60)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def slugify(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

class GTFSBuilder:
    def __init__(self, network: str, out_dir: pathlib.Path,
                 stations: list[dict], records: list[dict]):
        self.network   = network
        self.out_dir   = out_dir
        self.stations  = [s for s in stations if s["network"] == network]
        self.records   = [r for r in records  if r["network"] == network]

        # station_order[stop_id] = order integer (position along line)
        self.station_order: dict[str, int] = {
            s["stop_id"]: s.get("order", 0) for s in self.stations
        }
        # stop lookup by stop_id
        self.stop_by_id: dict[str, dict] = {s["stop_id"]: s for s in self.stations}

    # ------------------------------------------------------------------
    # Trip stitching
    # ------------------------------------------------------------------

    def _stitch_trips(self) -> list[dict]:
        """
        Return a list of trip dicts:
          {
            trip_id, route_id, route_name, route_short, headsign,
            stops: [ {stop_id, departure_min, arrival_min, seq} ]
          }
        """
        # Group records: (route_name, headsign, arrival_time) → [records]
        # arrival_time here is the time at the *terminal* (arival stop of each query).
        # Records going to the south terminal and north terminal are separate groups.
        group_key = lambda r: (
            r.get("route_name", ""),
            r.get("headsign", ""),
            r.get("arrival_time", ""),
        )
        grouped = defaultdict(list)
        for rec in self.records:
            grouped[group_key(rec)].append(rec)

        trips = []
        trip_seen = set()

        for (route_name, headsign, arr_at_terminal), members in grouped.items():
            if not members:
                continue

            # Sort members by station order (ascending = north → south)
            members_sorted = sorted(
                members,
                key=lambda r: self.station_order.get(r["origin_stop_id"], 999),
            )

            # Build ordered stop sequence
            stop_seq = []
            for m in members_sorted:
                sid = m["origin_stop_id"]
                dep_min = hms_to_minutes(m["departure_time"])
                arr_min = hms_to_minutes(m["arrival_time"])   # at terminal
                stop_seq.append({
                    "stop_id":     sid,
                    "dep_min":     dep_min,
                    "arr_at_term": arr_min,
                })

            # Add the terminal itself (arrival time only)
            # The last member's arrival_time is at the terminal
            if stop_seq:
                last = members_sorted[-1]
                # terminal stop_id: find by dest_name match
                term_id = self._find_stop_id_by_name(last["dest_name"])
                if term_id and term_id not in {s["stop_id"] for s in stop_seq}:
                    stop_seq.append({
                        "stop_id": term_id,
                        "dep_min": hms_to_minutes(last["arrival_time"]),
                        "arr_at_term": hms_to_minutes(last["arrival_time"]),
                    })

            # Fill any intermediate stations (with linear interpolation)
            stop_seq = self._fill_intermediate_stops(stop_seq)

            # Deduplicate trips: same route+headsign+first_departure = same trip
            if not stop_seq:
                continue
            trip_key = (route_name, headsign, stop_seq[0]["dep_min"])
            if trip_key in trip_seen:
                continue
            trip_seen.add(trip_key)

            route_id  = slugify(route_name or self.network)
            trip_id   = f"{route_id}__{slugify(headsign)}__{int(stop_seq[0]['dep_min'])}"

            # Assign stop_sequence numbers
            for i, s in enumerate(stop_seq):
                s["seq"] = i + 1

            trips.append({
                "trip_id":     trip_id,
                "route_id":    route_id,
                "route_name":  route_name,
                "route_short": members[0].get("route_short", ""),
                "headsign":    headsign,
                "vehicle_type": members[0].get("vehicle_type", ""),
                "stops":       stop_seq,
            })

        return trips

    def _find_stop_id_by_name(self, name: str) -> str | None:
        """Find the stop_id whose name best matches `name` (Google Maps name)."""
        if not name:
            return None
        name_lower = name.lower()
        # Exact match first
        for s in self.stations:
            if s["name"].lower() == name_lower:
                return s["stop_id"]
        # Partial match
        for s in self.stations:
            if s["name"].lower() in name_lower or name_lower in s["name"].lower():
                return s["stop_id"]
        return None

    def _fill_intermediate_stops(self, stop_seq: list[dict]) -> list[dict]:
        """
        For each pair of consecutive recorded stops, insert any stations
        (from the ordered station list) that lie between them, with departure
        minutes linearly interpolated.
        """
        if len(stop_seq) < 2:
            return stop_seq

        all_stops_ordered = sorted(
            self.stations, key=lambda s: s.get("order", 0)
        )
        stop_ids_ordered = [s["stop_id"] for s in all_stops_ordered]

        filled: list[dict] = []
        for i in range(len(stop_seq) - 1):
            a = stop_seq[i]
            b = stop_seq[i + 1]
            filled.append(a)

            # Find stations between a and b in line order
            try:
                idx_a = stop_ids_ordered.index(a["stop_id"])
                idx_b = stop_ids_ordered.index(b["stop_id"])
            except ValueError:
                continue

            if abs(idx_b - idx_a) <= 1:
                continue  # adjacent, nothing to fill

            direction = 1 if idx_b > idx_a else -1
            between = stop_ids_ordered[idx_a + direction: idx_b: direction]
            n = len(between) + 1
            for j, sid in enumerate(between):
                frac = (j + 1) / n
                interp_min = a["dep_min"] + frac * (b["dep_min"] - a["dep_min"])
                filled.append({
                    "stop_id": sid,
                    "dep_min": interp_min,
                    "arr_at_term": b["arr_at_term"],
                })

        filled.append(stop_seq[-1])
        return filled

    # ------------------------------------------------------------------
    # Route type
    # ------------------------------------------------------------------

    def _route_type(self, vehicle_type: str) -> int:
        mapping = {
            "HIGH_SPEED_TRAIN": 101,
            "HEAVY_RAIL":       2,
            "RAIL":             2,
            "COMMUTER_TRAIN":   2,
            "METRO_RAIL":       1,
            "MONORAIL":         12,
            "TRAM":             0,
            "BUS":              3,
        }
        return mapping.get(vehicle_type.upper(), 2)

    # ------------------------------------------------------------------
    # Write GTFS files
    # ------------------------------------------------------------------

    def write(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        trips = self._stitch_trips()

        if not trips:
            print(f"  WARNING: No trips generated for {self.network}. Skipping.")
            return

        self._write_agency()
        self._write_calendar()
        self._write_stops()
        self._write_routes(trips)
        self._write_trips_and_stop_times(trips)

        print(f"  {self.network}: {len(trips)} trips, "
              f"{len(self.stations)} stops → {self.out_dir}")

    def _write_agency(self):
        agency_map = {
            "THSR": ("THSR", "Taiwan High Speed Rail", "https://www.thsrc.com.tw"),
            "TRA":  ("TRA",  "Taiwan Railways Administration", "https://www.railway.gov.tw"),
        }
        agency_id, agency_name, agency_url = agency_map.get(
            self.network, (self.network, self.network, "")
        )
        with open(self.out_dir / "agency.txt", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["agency_id", "agency_name", "agency_url",
                        "agency_timezone", "agency_lang", "agency_phone"])
            w.writerow([agency_id, agency_name, agency_url,
                        "Asia/Taipei", "zh", ""])

    def _write_calendar(self):
        with open(self.out_dir / "calendar.txt", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["service_id", "monday", "tuesday", "wednesday", "thursday",
                        "friday", "saturday", "sunday", "start_date", "end_date"])
            w.writerow(["1", "1", "1", "1", "1", "1", "1", "1",
                        CALENDAR_START, CALENDAR_END])

    def _write_stops(self):
        with open(self.out_dir / "stops.txt", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stop_id", "stop_name", "stop_lat", "stop_lon",
                        "location_type", "parent_station"])
            for s in self.stations:
                w.writerow([
                    s["stop_id"],
                    s["name"],
                    s.get("stop_lat", 0.0),
                    s.get("stop_lon", 0.0),
                    "1",
                    "",
                ])

    def _write_routes(self, trips: list[dict]):
        seen_routes = {}
        for t in trips:
            rid = t["route_id"]
            if rid not in seen_routes:
                seen_routes[rid] = t

        agency_map = {"THSR": "THSR", "TRA": "TRA"}
        agency_id  = agency_map.get(self.network, self.network)

        with open(self.out_dir / "routes.txt", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["route_id", "agency_id", "route_short_name",
                        "route_long_name", "route_desc", "route_type"])
            for rid, t in seen_routes.items():
                route_type = self._route_type(t.get("vehicle_type", ""))
                w.writerow([
                    rid,
                    agency_id,
                    t.get("route_short", rid),
                    t.get("route_name", rid),
                    "",
                    route_type,
                ])

    def _write_trips_and_stop_times(self, trips: list[dict]):
        with (
            open(self.out_dir / "trips.txt", "w", newline="", encoding="utf-8") as tf,
            open(self.out_dir / "stop_times.txt", "w", newline="", encoding="utf-8") as sf,
        ):
            tw = csv.writer(tf)
            sw = csv.writer(sf)

            tw.writerow(["route_id", "trip_id", "service_id",
                         "trip_headsign", "direction_id", "shape_id"])
            sw.writerow(["trip_id", "stop_id", "arrival_time",
                         "departure_time", "stop_sequence"])

            for trip in trips:
                tw.writerow([
                    trip["route_id"],
                    trip["trip_id"],
                    "1",
                    trip["headsign"],
                    "",
                    "",
                ])
                for stop in trip["stops"]:
                    time_hms = minutes_to_hms(stop["dep_min"])
                    sw.writerow([
                        trip["trip_id"],
                        stop["stop_id"],
                        time_hms,   # arrival = departure (no dwell modelled)
                        time_hms,
                        stop["seq"],
                    ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not STATIONS_GEOCODED_JSON.exists():
        print(f"ERROR: {STATIONS_GEOCODED_JSON} not found. Run geocode_stations.py first.")
        sys.exit(1)
    if not RAW_TRIPS_JSON.exists():
        print(f"ERROR: {RAW_TRIPS_JSON} not found. Run fetch_schedules.py first.")
        sys.exit(1)

    with open(STATIONS_GEOCODED_JSON, encoding="utf-8") as f:
        stations = json.load(f)
    with open(RAW_TRIPS_JSON, encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(stations)} stations, {len(records)} raw departure records.")

    for network, out_dir in [("THSR", THSR_GTFS_DIR), ("TRA", TRA_GTFS_DIR)]:
        net_records = [r for r in records if r["network"] == network]
        net_stations = [s for s in stations if s["network"] == network]
        if not net_stations:
            print(f"  No stations for {network}, skipping.")
            continue
        print(f"\nBuilding {network} GTFS ({len(net_stations)} stations, "
              f"{len(net_records)} records) ...")
        builder = GTFSBuilder(network, out_dir, stations, records)
        builder.write()

    print("\nDone. Next step: run verify_network.py to confirm the feed loads correctly.")


if __name__ == "__main__":
    main()
