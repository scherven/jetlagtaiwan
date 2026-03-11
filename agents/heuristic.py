"""
Greedy heuristic agent — rule-based baseline.

Decision logic (evaluated in priority order):
1. If at a challenge station: always attempt it.
2. If a reachable uncontrolled station exists: board the departure whose route
   reaches the nearest uncontrolled station (fewest stops), provided affordable.
3. If all reachable stations are controlled by us: board toward highest-value challenge.
4. If the opponent controls a reachable station and we can afford to contest it: board toward it.
5. Otherwise: wait.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from engine.game_state import GameState, Team
from engine.rail_network import Departure, RailNetwork
from engine.rules import compute_route_chip_cost, get_valid_departures

logger = logging.getLogger(__name__)

# Action indices
ACTION_CHALLENGE = 10   # K = 10
ACTION_WAIT = 11


class HeuristicAgent:
    """
    Greedy heuristic agent implementing the same action interface as the RL agent.
    choose_action(obs, game_state, team_id, departures) -> int (0..11)
    """

    def __init__(self, config: dict):
        self.k = config["agents"]["max_departures_k"]
        self.reachability_window = config["agents"]["heuristic_reachability_window"]
        self.starting_station_id: Optional[str] = None

    def choose_action(
        self,
        game_state: GameState,
        rail_network: RailNetwork,
        team_id: str,
        available_departures: List[Departure],
    ) -> int:
        team = game_state.teams[team_id]
        opponent_id = "B" if team_id == "A" else "A"
        opponent = game_state.teams[opponent_id]
        stations = game_state.stations
        starting_id = self.starting_station_id

        # 1. Challenge at current station
        challenge_here = next(
            (c for c in game_state.challenges if c.station_id == team.current_station),
            None,
        )
        if challenge_here and not team.is_in_transit():
            return ACTION_CHALLENGE

        if team.is_in_transit():
            return ACTION_WAIT  # already moving, no decision

        # Compute route costs for each departure
        scored: List[Tuple[int, Departure, int]] = []  # (idx, dep, cost)
        for idx, dep in enumerate(available_departures[:self.k]):
            cost = compute_route_chip_cost(
                stations, dep.intermediate_stops, team_id, starting_id or ""
            )
            scored.append((idx, dep, cost))

        affordable = [(i, d, c) for i, d, c in scored if team.coins >= c or team.coins <= 0]

        # 2. Nearest uncontrolled station
        best: Optional[Tuple[int, int]] = None  # (action_idx, num_stops)
        for idx, dep, cost in affordable:
            for hop, stop_id in enumerate(dep.intermediate_stops):
                st = stations.get(stop_id)
                if st is None or stop_id == starting_id:
                    continue
                if st.controlling_team() is None:
                    if best is None or hop < best[1]:
                        best = (idx, hop)
                    break
        if best is not None:
            return best[0]

        # 3. Toward highest-value challenge
        if game_state.challenges:
            best_challenge_action: Optional[Tuple[float, int]] = None  # (value, idx)
            for idx, dep, cost in affordable:
                for stop_id in dep.intermediate_stops:
                    for ch in game_state.challenges:
                        if ch.station_id == stop_id:
                            val = ch.current_value()
                            if best_challenge_action is None or val > best_challenge_action[0]:
                                best_challenge_action = (val, idx)
                            break
            if best_challenge_action is not None:
                return best_challenge_action[1]

        # 4. Contest opponent-controlled station
        for idx, dep, cost in affordable:
            for stop_id in dep.intermediate_stops:
                st = stations.get(stop_id)
                if st and st.controlling_team() == opponent_id:
                    return idx

        # 5. Wait
        return ACTION_WAIT
