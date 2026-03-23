"""
GTFS parsing and schedule lookup for any rail/transit network.

Accepts one or more GTFS feed directories (each containing stops.txt, trips.txt,
stop_times.txt, routes.txt).  Two merge strategies are supported:

  "parent" — collapse directional platform stops into their parent station using
             the parent_station column (e.g. NYC Subway "101N"/"101S" → "101").
             Use this for single-agency feeds that already encode station hierarchy.

  "name"   — merge stops from different feeds that share the same normalised name
             into one canonical station node.  Use this when combining multiple
             agencies (e.g. TRA + THSR) that may share station names at interchanges.
"""

from __future__ import annotations

import bisect
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StopNode:
    """A unified station node."""
    id: str
    name: str
    lat: float
    lon: float
    source_ids: List[str] = field(default_factory=list)  # all GTFS stop_ids mapped here


@dataclass
class Departure:
    """A single train departure from a station."""
    trip_id: str
    route_id: str
    departure_minute: int        # minutes since midnight
    destination_stop_id: str     # canonical station id of final stop on this boarding
    intermediate_stops: List[str]  # ordered canonical station ids after boarding (incl. destination)
    arrival_minutes: List[int]   # arrival minute at each stop in intermediate_stops


class RailNetwork:
    """
    Unified rail network built from one or more GTFS feeds.

    After construction:
      self.stations  : Dict[str, StopNode]       — keyed by canonical station ID
      self.schedules : Dict[str, List[Departure]] — keyed by station ID, sorted by departure_minute
    """

    def __init__(self, feed_dirs: List[str], merge_strategy: str = "parent"):
        """
        Parameters
        ----------
        feed_dirs        : list of directories, each a GTFS feed root.
        merge_strategy   : "parent" or "name"  (see module docstring).
        """
        if merge_strategy not in ("parent", "name"):
            raise ValueError(f"merge_strategy must be 'parent' or 'name', got {merge_strategy!r}")

        self.stations: Dict[str, StopNode] = {}
        self.schedules: Dict[str, List[Departure]] = defaultdict(list)
        self._stop_id_map: Dict[str, str] = {}   # raw GTFS stop_id → canonical station id

        self._merge_strategy = merge_strategy
        self._load(feed_dirs)

    # ------------------------------------------------------------------
    # Top-level load
    # ------------------------------------------------------------------

    def _load(self, feed_dirs: List[str]) -> None:
        all_stops: List[pd.DataFrame] = []
        all_stop_times: List[pd.DataFrame] = []
        all_trips: List[pd.DataFrame] = []

        for i, directory in enumerate(feed_dirs):
            prefix = f"F{i}"
            stops, trips, stop_times = self._load_feed(directory, prefix)
            all_stops.append(stops)
            all_stop_times.append(stop_times)
            all_trips.append(trips)

        combined_stops = pd.concat(all_stops, ignore_index=True)
        if self._merge_strategy == "parent":
            self._build_station_index_parent(combined_stops)
        else:
            self._build_station_index_name(combined_stops)

        for trips, stop_times in zip(all_trips, all_stop_times):
            self._build_schedules(stop_times, trips)

        for sid in self.schedules:
            self.schedules[sid].sort(key=lambda d: d.departure_minute)

        # Cache lat/lon bounds so encode_observation doesn't recompute per call
        lats = [s.lat for s in self.stations.values()]
        lons = [s.lon for s in self.stations.values()]
        self.latlon_bounds = (
            (min(lats), max(lats), min(lons), max(lons))
            if lats else (0.0, 1.0, 0.0, 1.0)
        )

        logger.info(
            "RailNetwork loaded: %d stations, %d schedule entries (strategy=%s)",
            len(self.stations),
            sum(len(v) for v in self.schedules.values()),
            self._merge_strategy,
        )

    # ------------------------------------------------------------------
    # Feed loading
    # ------------------------------------------------------------------

    def _load_feed(
        self, directory: str, prefix: str
    ):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"GTFS directory not found: {directory}")

        stops = pd.read_csv(os.path.join(directory, "stops.txt"), dtype=str, low_memory=False)
        trips = pd.read_csv(os.path.join(directory, "trips.txt"), dtype=str, low_memory=False)
        stop_times = pd.read_csv(os.path.join(directory, "stop_times.txt"), dtype=str, low_memory=False)

        # Namespace IDs to avoid collisions when merging multiple feeds
        stops["_orig_stop_id"] = stops["stop_id"]
        stops["stop_id"] = prefix + "_" + stops["stop_id"]
        if "parent_station" in stops.columns:
            mask = stops["parent_station"].notna() & (stops["parent_station"] != "")
            stops.loc[mask, "parent_station"] = prefix + "_" + stops.loc[mask, "parent_station"]

        stop_times["stop_id"] = prefix + "_" + stop_times["stop_id"]
        stop_times["trip_id"] = prefix + "_" + stop_times["trip_id"]
        trips["trip_id"] = prefix + "_" + trips["trip_id"]
        trips["route_id"] = prefix + "_" + trips["route_id"]

        stops["_name_norm"] = stops["stop_name"].str.strip().str.lower()

        logger.info(
            "Feed %s (%s): %d stops, %d trips, %d stop_time rows",
            prefix, directory, len(stops), len(trips), len(stop_times),
        )
        return stops, trips, stop_times

    # ------------------------------------------------------------------
    # Station index — "parent" strategy
    # ------------------------------------------------------------------

    def _build_station_index_parent(self, stops: pd.DataFrame) -> None:
        """
        Use parent_station column to collapse platforms into stations.
        Stops with location_type == '1' (or no parent) become canonical nodes.
        Stops with a parent_station are mapped to their parent.
        """
        # Identify parent/station rows: location_type == '1' OR no parent_station
        has_parent_col = "parent_station" in stops.columns

        for _, row in stops.iterrows():
            sid = row["stop_id"]
            parent = row.get("parent_station", "") if has_parent_col else ""
            loc_type = str(row.get("location_type", "")).strip()

            if has_parent_col and parent and pd.notna(parent) and parent != "":
                # This is a platform/entrance — map to parent
                self._stop_id_map[sid] = parent
            elif loc_type == "1" or not (has_parent_col and parent and pd.notna(parent)):
                # This is a station node
                try:
                    lat = float(row.get("stop_lat", 0))
                    lon = float(row.get("stop_lon", 0))
                except (ValueError, TypeError):
                    lat, lon = 0.0, 0.0

                if sid not in self.stations:
                    self.stations[sid] = StopNode(
                        id=sid,
                        name=row["stop_name"].strip(),
                        lat=lat,
                        lon=lon,
                        source_ids=[sid],
                    )
                self._stop_id_map[sid] = sid

        # Second pass: ensure all parent references point to existing stations
        # (sometimes parent_station IDs are listed without their own row)
        for _, row in stops.iterrows():
            sid = row["stop_id"]
            parent = row.get("parent_station", "") if has_parent_col else ""
            if has_parent_col and parent and pd.notna(parent) and parent != "":
                if parent not in self.stations:
                    # Parent not seen — promote this stop as the canonical node
                    try:
                        lat = float(row.get("stop_lat", 0))
                        lon = float(row.get("stop_lon", 0))
                    except (ValueError, TypeError):
                        lat, lon = 0.0, 0.0
                    self.stations[parent] = StopNode(
                        id=parent,
                        name=row["stop_name"].strip(),
                        lat=lat,
                        lon=lon,
                        source_ids=[sid],
                    )
                else:
                    if sid not in self.stations[parent].source_ids:
                        self.stations[parent].source_ids.append(sid)
                self._stop_id_map[sid] = parent

        logger.info("Station index (parent strategy): %d stations", len(self.stations))

    # ------------------------------------------------------------------
    # Station index — "name" strategy
    # ------------------------------------------------------------------

    def _build_station_index_name(self, stops: pd.DataFrame) -> None:
        """
        Merge stops with identical normalised names across all feeds.
        First occurrence of each name wins for lat/lon/canonical ID.
        """
        groups: Dict[str, List] = defaultdict(list)
        for _, row in stops.iterrows():
            groups[row["_name_norm"]].append(row)

        for norm_name, rows in groups.items():
            canonical_id = rows[0]["stop_id"]
            canonical_name = rows[0]["stop_name"].strip()
            try:
                lat = float(rows[0].get("stop_lat", 0))
                lon = float(rows[0].get("stop_lon", 0))
            except (ValueError, TypeError):
                lat, lon = 0.0, 0.0

            node = StopNode(
                id=canonical_id,
                name=canonical_name,
                lat=lat,
                lon=lon,
                source_ids=[r["stop_id"] for r in rows],
            )
            self.stations[canonical_id] = node
            for r in rows:
                self._stop_id_map[r["stop_id"]] = canonical_id

        logger.info("Station index (name strategy): %d stations", len(self.stations))

    # ------------------------------------------------------------------
    # Schedule building
    # ------------------------------------------------------------------

    def _build_schedules(self, stop_times: pd.DataFrame, trips: pd.DataFrame) -> None:
        """Build Departure objects from a single feed's stop_times + trips."""
        st = stop_times.copy()
        st["dep_min"] = st["departure_time"].apply(self._hms_to_minutes)
        st["arr_min"] = st["arrival_time"].apply(self._hms_to_minutes)
        st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
        st = st.dropna(subset=["dep_min", "arr_min", "stop_sequence"])
        st = st.sort_values(["trip_id", "stop_sequence"])

        trip_route = trips.set_index("trip_id")["route_id"].to_dict()

        for trip_id, group in st.groupby("trip_id"):
            group = group.sort_values("stop_sequence")
            stop_ids = list(group["stop_id"])
            dep_mins = list(group["dep_min"])
            arr_mins = list(group["arr_min"])

            if len(stop_ids) < 2:
                continue

            route_id = trip_route.get(trip_id, "unknown")

            for i in range(len(stop_ids) - 1):
                board_raw = stop_ids[i]
                canonical_board = self._stop_id_map.get(board_raw)
                if canonical_board is None or canonical_board not in self.stations:
                    continue

                remaining_canonical = []
                remaining_arr = []
                for j in range(i + 1, len(stop_ids)):
                    c = self._stop_id_map.get(stop_ids[j])
                    if c is not None and c in self.stations:
                        # Skip consecutive duplicates (same platform → same station)
                        if not remaining_canonical or remaining_canonical[-1] != c:
                            remaining_canonical.append(c)
                            remaining_arr.append(int(arr_mins[j]))

                if not remaining_canonical:
                    continue

                dep = Departure(
                    trip_id=trip_id,
                    route_id=route_id,
                    departure_minute=int(dep_mins[i]),
                    destination_stop_id=remaining_canonical[-1],
                    intermediate_stops=remaining_canonical,
                    arrival_minutes=remaining_arr,
                )
                self.schedules[canonical_board].append(dep)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def departures_from(
        self,
        station_id: str,
        from_minute: int,
        to_minute: int,
    ) -> List[Departure]:
        """Return departures from station_id with departure_minute in [from_minute, to_minute).

        Uses binary search on the pre-sorted schedule list for O(log N + k) performance
        instead of O(N) linear scan.
        """
        deps = self.schedules.get(station_id, [])
        if not deps:
            return []
        # Binary-search for first departure >= from_minute
        lo = bisect.bisect_left(deps, from_minute, key=lambda d: d.departure_minute)
        result = []
        for d in deps[lo:]:
            if d.departure_minute >= to_minute:
                break
            result.append(d)
        return result

    def station_by_name(self, name: str) -> Optional[StopNode]:
        """Case-insensitive station lookup by name."""
        norm = name.strip().lower()
        for node in self.stations.values():
            if node.name.strip().lower() == norm:
                return node
        return None

    def summary(self) -> str:
        total_deps = sum(len(v) for v in self.schedules.values())
        lines = [
            f"Stations       : {len(self.stations)}",
            f"With schedules : {len(self.schedules)}",
            f"Total departures: {total_deps}",
            "Sample stations:",
        ]
        for s in list(self.stations.values())[:8]:
            n_deps = len(self.schedules.get(s.id, []))
            lines.append(f"  {s.name!r:40s}  lat={s.lat:.4f} lon={s.lon:.4f}  deps={n_deps}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _hms_to_minutes(time_str: str) -> Optional[float]:
        """Convert HH:MM:SS to minutes since midnight. Handles times > 24h."""
        try:
            parts = str(time_str).strip().split(":")
            if len(parts) != 3:
                return None
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 60 + m + s / 60.0
        except (ValueError, AttributeError):
            return None
