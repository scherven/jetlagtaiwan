"""
Observation encoding for the RL agent.

Vector layout (all float32):
  18   scalars:
         us_coins, opp_coins, sim_time, day,
         us_lat, us_lon,
         us_dest_lat, us_dest_lon, us_arrival,   ← -1 when not in transit
         opp_lat, opp_lon,
         opp_dest_lat, opp_dest_lon, opp_arrival,
         is_challenge_here, n_valid_departures,
         our_controlled_fraction, opp_controlled_fraction
  K*8  departure slots (0-padded for unused slots):
         dest_lat, dest_lon, dep_time, duration, route_cost,
         dest_ownership (-1/0/1), dest_our_chips, dest_opp_chips
  C*5  challenge slots (0-padded):
         station_lat, station_lon, value, our_chips, opp_chips

Total: 18 + K*8 + C_MAX*5
  = 148 for K=10, C_MAX=10

Station index (arbitrary sort order, no spatial meaning) has been
replaced throughout by normalised lat/lon so the agent can reason
about geographic proximity.  All lat/lon values are normalised to
[0, 1] using the per-episode bounding box of all stations.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from engine.game_state import GameState
from engine.rail_network import Departure, RailNetwork
from engine.rules import compute_route_chip_cost, count_controlled_stations
from engine.clock import DAY_DURATION, DAY_END_MINUTE


C_MAX_DEFAULT = 10  # max challenge slots encoded in the observation


def _latlon_bounds(game_state: GameState) -> Tuple[float, float, float, float]:
    """Return (min_lat, max_lat, min_lon, max_lon) across all stations."""
    lats = [s.lat for s in game_state.stations.values()]
    lons = [s.lon for s in game_state.stations.values()]
    return min(lats), max(lats), min(lons), max(lons)


def _norm_lat(lat: float, min_lat: float, max_lat: float) -> float:
    span = max_lat - min_lat
    return (lat - min_lat) / span if span > 1e-9 else 0.5


def _norm_lon(lon: float, min_lon: float, max_lon: float) -> float:
    span = max_lon - min_lon
    return (lon - min_lon) / span if span > 1e-9 else 0.5


def encode_observation(
    game_state: GameState,
    rail_network,           # RailNetwork or GridNetwork — duck-typed
    team_id: str,
    departures: List[Departure],
    starting_station_id: str,
    k: int = 10,
    starting_coins: int = 50,
    max_chips_per_station: int = 5,
    c_max: int = C_MAX_DEFAULT,
) -> np.ndarray:
    """Encode the full observation vector for `team_id`."""
    opponent_id = "B" if team_id == "A" else "A"
    us  = game_state.teams[team_id]
    opp = game_state.teams[opponent_id]

    chip_attr_us  = f"chips_team_{team_id.lower()}"
    chip_attr_opp = f"chips_team_{opponent_id.lower()}"

    station_list = list(game_state.stations.values())
    N = len(station_list)

    min_lat, max_lat, min_lon, max_lon = _latlon_bounds(game_state)

    def nlat(lat: float) -> float:
        return _norm_lat(lat, min_lat, max_lat)

    def nlon(lon: float) -> float:
        return _norm_lon(lon, min_lon, max_lon)

    # Station lat/lon lookup
    def station_latlon(sid: str) -> Tuple[float, float]:
        s = game_state.stations.get(sid)
        if s is None:
            return 0.5, 0.5
        return nlat(s.lat), nlon(s.lon)

    max_chips = max(
        max((getattr(s, chip_attr_us) for s in station_list), default=0),
        max((getattr(s, chip_attr_opp) for s in station_list), default=0),
        1,
    )

    # Challenge value normalisation
    max_chal_val = max((c.current_value() for c in game_state.challenges), default=1.0)
    max_chal_val = max(max_chal_val, 1.0)

    # Global board control
    our_count = count_controlled_stations(game_state, team_id,    starting_station_id)
    opp_count = count_controlled_stations(game_state, opponent_id, starting_station_id)
    total_stations = max(N - 1, 1)

    # ── Scalars (18) ──────────────────────────────────────────────────────────
    total_days = 5
    is_ch_here = 1.0 if any(c.station_id == us.current_station
                             for c in game_state.challenges) else 0.0
    n_valid = min(len(departures), k)

    us_lat_n,  us_lon_n  = station_latlon(us.current_station)
    opp_lat_n, opp_lon_n = station_latlon(opp.current_station)

    if us.destination_station:
        us_dest_lat_n, us_dest_lon_n = station_latlon(us.destination_station)
    else:
        us_dest_lat_n = us_dest_lon_n = -1.0

    if opp.destination_station:
        opp_dest_lat_n, opp_dest_lon_n = station_latlon(opp.destination_station)
    else:
        opp_dest_lat_n = opp_dest_lon_n = -1.0

    scalar = np.array([
        max(us.coins, 0) / max(starting_coins, 1),
        max(opp.coins, 0) / max(starting_coins, 1),
        game_state.sim_minute / max(DAY_DURATION * total_days, 1),
        (game_state.day - 1) / max(total_days - 1, 1),
        us_lat_n,
        us_lon_n,
        us_dest_lat_n,
        us_dest_lon_n,
        (us.arrival_time / max(DAY_DURATION * total_days, 1))
            if us.arrival_time is not None else -1.0,
        opp_lat_n,
        opp_lon_n,
        opp_dest_lat_n,
        opp_dest_lon_n,
        (opp.arrival_time / max(DAY_DURATION * total_days, 1))
            if opp.arrival_time is not None else -1.0,
        is_ch_here,
        n_valid / max(k, 1),
        our_count / total_stations,
        opp_count / total_stations,
    ], dtype=np.float32)

    # ── Departure slots (K × 8) ───────────────────────────────────────────────
    dep_vec = np.zeros((k, 8), dtype=np.float32)
    for j, dep in enumerate(departures[:k]):
        dest_id      = dep.destination_stop_id
        dest_station = game_state.stations.get(dest_id)
        d_lat, d_lon = station_latlon(dest_id)
        duration     = (dep.arrival_minutes[-1] - dep.departure_minute
                        ) if dep.arrival_minutes else 0
        cost = compute_route_chip_cost(
            game_state.stations,
            dep.intermediate_stops,
            team_id,
            starting_station_id,
            max_chips_per_station=max_chips_per_station,
        )
        if dest_station:
            uc = getattr(dest_station, chip_attr_us)
            oc = getattr(dest_station, chip_attr_opp)
            dest_own = 1.0 if uc > oc else (-1.0 if oc > uc else 0.0)
            dest_uc  = uc / max_chips
            dest_oc  = oc / max_chips
        else:
            dest_own = dest_uc = dest_oc = 0.0

        dep_vec[j] = [
            d_lat,
            d_lon,
            dep.departure_minute / max(DAY_END_MINUTE + 1, 1),
            duration / max(DAY_DURATION, 1),
            cost / max(starting_coins, 1),
            dest_own,
            dest_uc,
            dest_oc,
        ]

    # ── Challenge slots (C_MAX × 5) ──────────────────────────────────────────
    chal_vec = np.zeros((c_max, 5), dtype=np.float32)
    for j, c in enumerate(game_state.challenges[:c_max]):
        chal_station = game_state.stations.get(c.station_id)
        c_lat, c_lon = station_latlon(c.station_id)
        val = c.current_value() / max_chal_val
        if chal_station:
            uc = getattr(chal_station, chip_attr_us)
            oc = getattr(chal_station, chip_attr_opp)
        else:
            uc = oc = 0
        chal_vec[j] = [
            c_lat,
            c_lon,
            val,
            uc / max_chips,
            oc / max_chips,
        ]

    return np.concatenate([scalar, dep_vec.flatten(), chal_vec.flatten()])


def observation_size(n_stations: int = 0, k: int = 10,
                     c_max: int = C_MAX_DEFAULT) -> int:
    """Return flat observation vector length.

    n_stations is accepted for API compatibility but is unused —
    the encoding is independent of the total number of stations.
    """
    return 18 + k * 8 + c_max * 5
