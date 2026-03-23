"""
Synthetic 10×10 grid network for fast RL training.

Topology:
  - 100 stations arranged in a 10×10 grid, IDs "G_R{row}C{col}".
  - Each station connects to its 4 cardinal neighbours (no diagonals).
  - One trip per directed edge per departure slot.
  - Departures run every `interval_minutes` from 07:30 to the last slot
    whose arrival fits before 17:30.

This class exposes the same interface as RailNetwork so it can be
dropped in transparently: .stations, .schedules, departures_from(),
station_by_name().
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from engine.clock import DAY_END_MINUTE, DAY_START_MINUTE
from engine.rail_network import Departure, StopNode


class GridNetwork:
    """Synthetic 10×10 grid rail network."""

    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        interval_minutes: int = 5,
        travel_time: int = 5,
    ):
        self.rows = rows
        self.cols = cols
        self.stations: Dict[str, StopNode] = {}
        self.schedules: Dict[str, List[Departure]] = defaultdict(list)
        self._build(interval_minutes, travel_time)
        # Cache lat/lon bounds so encode_observation doesn't recompute per call
        lats = [s.lat for s in self.stations.values()]
        lons = [s.lon for s in self.stations.values()]
        self.latlon_bounds = (min(lats), max(lats), min(lons), max(lons))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _sid(self, r: int, c: int) -> str:
        return f"G_R{r:02d}C{c:02d}"

    def _build(self, interval_minutes: int, travel_time: int) -> None:
        rows, cols = self.rows, self.cols

        # Stations — use a simple lat/lon grid for display purposes
        for r in range(rows):
            for c in range(cols):
                sid = self._sid(r, c)
                self.stations[sid] = StopNode(
                    id=sid,
                    name=f"R{r:02d}C{c:02d}",
                    lat=40.70 + r * 0.02,
                    lon=-74.00 + c * 0.02,
                    source_ids=[sid],
                )

        # Schedules: one directed edge per neighbour pair
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for r in range(rows):
            for c in range(cols):
                src = self._sid(r, c)
                for dr, dc in neighbours:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    dst = self._sid(nr, nc)
                    route_id = f"grid_{src}_{dst}"
                    dep_min = DAY_START_MINUTE
                    trip_num = 0
                    while dep_min + travel_time <= DAY_END_MINUTE:
                        arr_min = dep_min + travel_time
                        self.schedules[src].append(
                            Departure(
                                trip_id=f"{route_id}_t{trip_num}",
                                route_id=route_id,
                                departure_minute=dep_min,
                                destination_stop_id=dst,
                                intermediate_stops=[dst],
                                arrival_minutes=[arr_min],
                            )
                        )
                        dep_min += interval_minutes
                        trip_num += 1

        for sid in self.schedules:
            self.schedules[sid].sort(key=lambda d: d.departure_minute)

    # ------------------------------------------------------------------
    # Query interface (mirrors RailNetwork)
    # ------------------------------------------------------------------

    def departures_from(
        self,
        station_id: str,
        from_minute: int,
        to_minute: int,
    ) -> List[Departure]:
        deps = self.schedules.get(station_id, [])
        if not deps:
            return []
        lo = bisect.bisect_left(deps, from_minute, key=lambda d: d.departure_minute)
        result = []
        for d in deps[lo:]:
            if d.departure_minute >= to_minute:
                break
            result.append(d)
        return result

    def station_by_name(self, name: str) -> Optional[StopNode]:
        norm = name.strip().lower()
        for node in self.stations.values():
            if node.name.strip().lower() == norm:
                return node
        return None

    def summary(self) -> str:
        total = sum(len(v) for v in self.schedules.values())
        return (
            f"GridNetwork {self.rows}×{self.cols}: "
            f"{len(self.stations)} stations, {total} schedule entries"
        )
