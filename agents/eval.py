"""
Observation encoding for the RL agent.
Also provides route chip cost pre-computation used by both heuristic and RL agents.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from engine.game_state import GameState, Station
from engine.rail_network import Departure, RailNetwork
from engine.rules import compute_route_chip_cost
from engine.clock import DAY_DURATION, DAY_END_MINUTE, sim_minute_to_wall_clock


def encode_observation(
    game_state: GameState,
    rail_network: RailNetwork,
    team_id: str,
    departures: List[Departure],
    starting_station_id: str,
    k: int = 10,
    starting_coins: int = 50,
) -> np.ndarray:
    """
    Encode the full observation vector for `team_id`.

    Vector layout (all float32):
      N*3  station ownership  (1=we control, -1=opp controls, 0=neutral)
      N*2  chip counts        (our chips, opp chips) normalised by max observed
      N*2  challenge presence (bool) + current value (normalised)
      1    our coin count     (normalised by starting_coins)
      1    opp coin count
      1    current sim time   (0-1)
      1    current day        (0-1)
      1    our current station index (normalised)
      1    our destination station index (-1 if not in transit, else normalised)
      1    our arrival time   (-1 or normalised sim minute)
      1    opp current station index
      1    opp destination station index
      1    opp arrival time
      K*4  departures: dest_idx, dep_time, duration, route_cost  (all normalised)

    Total: N*7 + 10 + K*4
    """
    opponent_id = "B" if team_id == "A" else "A"
    us = game_state.teams[team_id]
    opp = game_state.teams[opponent_id]

    station_list = game_state.station_list()
    N = len(station_list)
    station_index = {s.id: i for i, s in enumerate(station_list)}

    chip_attr_us  = f"chips_team_{team_id.lower()}"
    chip_attr_opp = f"chips_team_{opponent_id.lower()}"

    # Max chips across all stations for normalisation
    max_chips = max(
        (max(getattr(s, chip_attr_us), getattr(s, chip_attr_opp)) for s in station_list),
        default=1,
    )
    max_chips = max(max_chips, 1)

    # Challenge lookup: station_id → challenge value
    challenge_map: Dict[str, float] = {}
    max_chal_val = 1.0
    for c in game_state.challenges:
        v = c.current_value()
        challenge_map[c.station_id] = v
        max_chal_val = max(max_chal_val, v)

    ownership = np.zeros(N, dtype=np.float32)
    our_chips  = np.zeros(N, dtype=np.float32)
    opp_chips  = np.zeros(N, dtype=np.float32)
    chal_present = np.zeros(N, dtype=np.float32)
    chal_value   = np.zeros(N, dtype=np.float32)

    for i, s in enumerate(station_list):
        uc = getattr(s, chip_attr_us)
        oc = getattr(s, chip_attr_opp)
        if uc > oc:
            ownership[i] = 1.0
        elif oc > uc:
            ownership[i] = -1.0
        our_chips[i]  = uc / max_chips
        opp_chips[i]  = oc / max_chips
        if s.id in challenge_map:
            chal_present[i] = 1.0
            chal_value[i]   = challenge_map[s.id] / max_chal_val

    total_days = 5  # spec default; passed in via num_days param if needed

    scalar = np.array([
        us.coins  / starting_coins,
        opp.coins / starting_coins,
        game_state.sim_minute / (DAY_DURATION * total_days),
        (game_state.day - 1) / (total_days - 1) if total_days > 1 else 0.0,
        station_index.get(us.current_station,  0) / max(N - 1, 1),
        (station_index.get(us.destination_station,  -1) / max(N - 1, 1))
            if us.destination_station else -1.0,
        (us.arrival_time / (DAY_DURATION * total_days))
            if us.arrival_time is not None else -1.0,
        station_index.get(opp.current_station, 0) / max(N - 1, 1),
        (station_index.get(opp.destination_station, -1) / max(N - 1, 1))
            if opp.destination_station else -1.0,
        (opp.arrival_time / (DAY_DURATION * total_days))
            if opp.arrival_time is not None else -1.0,
    ], dtype=np.float32)

    # Departure slots
    dep_vec = np.zeros((k, 4), dtype=np.float32)
    max_duration = DAY_DURATION
    for j, dep in enumerate(departures[:k]):
        dest_idx = station_index.get(dep.destination_stop_id, 0)
        duration = (dep.arrival_minutes[-1] - dep.departure_minute) if dep.arrival_minutes else 0
        cost = compute_route_chip_cost(
            game_state.stations,
            dep.intermediate_stops,
            team_id,
            starting_station_id,
        )
        dep_vec[j] = [
            dest_idx / max(N - 1, 1),
            dep.departure_minute / (DAY_END_MINUTE + 1),
            duration / max(max_duration, 1),
            cost / max(starting_coins, 1),
        ]

    return np.concatenate([
        ownership, our_chips, opp_chips, chal_present, chal_value,
        scalar,
        dep_vec.flatten(),
    ])


def observation_size(n_stations: int, k: int = 10) -> int:
    return n_stations * 5 + 10 + k * 4
