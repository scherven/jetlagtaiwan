"""
Observation encoding for the RL agent.

Vector layout (all float32):
  14   scalars:
         us_coins, opp_coins, sim_time, day,
         us_station_idx, us_dest_idx, us_arrival,
         opp_station_idx, opp_dest_idx, opp_arrival,
         is_challenge_here, n_valid_departures,
         our_controlled_fraction, opp_controlled_fraction
  K*7  departure slots (0-padded for unused slots):
         dest_idx, dep_time, duration, route_cost,
         dest_ownership (-1/0/1), dest_our_chips, dest_opp_chips
  C*5  challenge slots (0-padded):
         station_idx, value, our_chips, opp_chips, is_reachable

Total: 14 + K*7 + C_MAX*5
  = 134 for K=10, C_MAX=10  (down from N*5+12+K*4 = 2612 with K=30, N=496)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from engine.game_state import GameState
from engine.rail_network import Departure, RailNetwork
from engine.rules import compute_route_chip_cost, count_controlled_stations
from engine.clock import DAY_DURATION, DAY_END_MINUTE


C_MAX_DEFAULT = 10  # max challenge slots encoded in the observation


def encode_observation(
    game_state: GameState,
    rail_network: RailNetwork,
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

    station_list  = game_state.station_list()
    N             = len(station_list)
    station_index = {s.id: i for i, s in enumerate(station_list)}

    chip_attr_us  = f"chips_team_{team_id.lower()}"
    chip_attr_opp = f"chips_team_{opponent_id.lower()}"

    # Max chips across all stations for normalisation
    max_chips = max(
        (max(getattr(s, chip_attr_us), getattr(s, chip_attr_opp)) for s in station_list),
        default=1,
    )
    max_chips = max(max_chips, 1)

    # Challenge lookup: station_id → current value
    challenge_map: Dict[str, float] = {}
    max_chal_val = 1.0
    for c in game_state.challenges:
        v = c.current_value()
        challenge_map[c.station_id] = v
        max_chal_val = max(max_chal_val, v)

    # Reachable destinations (for challenge reachability flag)
    reachable_dests = {dep.destination_stop_id for dep in departures[:k]}

    # Global board control
    our_count = count_controlled_stations(game_state, team_id,    starting_station_id)
    opp_count = count_controlled_stations(game_state, opponent_id, starting_station_id)
    total_stations = max(N - 1, 1)  # exclude starting station

    # ── Scalars (14) ──────────────────────────────────────────────────────────
    total_days   = 5
    us_coins     = max(us.coins, 0)
    is_ch_here   = 1.0 if any(c.station_id == us.current_station
                               for c in game_state.challenges) else 0.0
    n_valid      = min(len(departures), k)

    scalar = np.array([
        us_coins / max(starting_coins, 1),
        max(opp.coins, 0) / max(starting_coins, 1),
        game_state.sim_minute / max(DAY_DURATION * total_days, 1),
        (game_state.day - 1) / max(total_days - 1, 1),
        station_index.get(us.current_station,  0) / max(N - 1, 1),
        (station_index.get(us.destination_station, -1) / max(N - 1, 1))
            if us.destination_station else -1.0,
        (us.arrival_time / max(DAY_DURATION * total_days, 1))
            if us.arrival_time is not None else -1.0,
        station_index.get(opp.current_station, 0) / max(N - 1, 1),
        (station_index.get(opp.destination_station, -1) / max(N - 1, 1))
            if opp.destination_station else -1.0,
        (opp.arrival_time / max(DAY_DURATION * total_days, 1))
            if opp.arrival_time is not None else -1.0,
        is_ch_here,
        n_valid / max(k, 1),
        our_count / total_stations,
        opp_count / total_stations,
    ], dtype=np.float32)

    # ── Departure slots (K × 7) ───────────────────────────────────────────────
    dep_vec = np.zeros((k, 7), dtype=np.float32)
    for j, dep in enumerate(departures[:k]):
        dest_id      = dep.destination_stop_id
        dest_idx     = station_index.get(dest_id, 0)
        dest_station = game_state.stations.get(dest_id)
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
            dest_idx / max(N - 1, 1),
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
        sidx         = station_index.get(c.station_id, 0)
        val          = c.current_value() / max_chal_val
        if chal_station:
            uc = getattr(chal_station, chip_attr_us)
            oc = getattr(chal_station, chip_attr_opp)
        else:
            uc = oc = 0
        is_reachable = 1.0 if c.station_id in reachable_dests else 0.0
        chal_vec[j] = [
            sidx / max(N - 1, 1),
            val,
            uc / max_chips,
            oc / max_chips,
            is_reachable,
        ]

    return np.concatenate([scalar, dep_vec.flatten(), chal_vec.flatten()])


def observation_size(n_stations: int = 0, k: int = 10,
                     c_max: int = C_MAX_DEFAULT) -> int:
    """Return flat observation vector length.

    n_stations is accepted for API compatibility but is no longer used —
    the new encoding is independent of the total number of stations.
    """
    return 14 + k * 7 + c_max * 5
