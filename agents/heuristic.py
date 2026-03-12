"""
Greedy heuristic agent — rule-based baseline.

Decision logic (evaluated in priority order):
1. If at a challenge station: always attempt it.
2. If a reachable uncontrolled station exists: board the departure whose route
   reaches the nearest uncontrolled station (fewest stops), provided affordable.
3. Board directly to the highest-value challenge station (departure must END there).
4. Contest the nearest opponent-controlled station we can afford to visit.
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
    choose_action(game_state, rail_network, team_id, departures) -> int (0..11)
    """

    def __init__(self, config: dict):
        self.k = config["agents"]["max_departures_k"]
        self.reachability_window = config["agents"]["heuristic_reachability_window"]
        self.extra_chips: int = config["agents"].get("heuristic_extra_chips", 0)
        self.max_chips: int = config["game"].get("max_chips_per_station", 5)
        self.challenge_window: int = config["agents"].get("heuristic_challenge_window", 180)
        self.challenge_k: int = config["agents"].get("heuristic_challenge_k", 50)
        self.low_coins_threshold: int = config["agents"].get("heuristic_low_coins_threshold", 15)
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
        stations = game_state.stations
        starting_id = self.starting_station_id

        # Sync desired_extra_chips onto team so simulation uses the right placement.
        team.desired_extra_chips = self.extra_chips

        # 1. Challenge at current station (only when idle)
        if not team.is_in_transit():
            challenge_here = next(
                (c for c in game_state.challenges if c.station_id == team.current_station),
                None,
            )
            if challenge_here:
                return ACTION_CHALLENGE

        if team.is_in_transit():
            return ACTION_WAIT  # already moving, no decision

        # Compute route costs for each departure (using the same extra_chips the
        # simulation will actually spend, so affordability is accurate).
        scored: List[Tuple[int, Departure, int]] = []  # (idx, dep, cost)
        for idx, dep in enumerate(available_departures[:self.k]):
            cost = compute_route_chip_cost(
                stations,
                dep.intermediate_stops,
                team_id,
                starting_id or "",
                extra_chips=self.extra_chips,
                max_chips_per_station=self.max_chips,
            )
            scored.append((idx, dep, cost))

        # Affordable: we have enough coins, OR we are broke (free travel, but
        # we can't claim anything in that case — handled per step below).
        affordable = [(i, d, c) for i, d, c in scored if team.coins >= c or team.coins <= 0]

        # Helper: find the best departure ending directly at a challenge station.
        # First checks the pre-computed departure list, then if nothing is found
        # widens the search to challenge_window with challenge_k departures so
        # challenges far from the current station (e.g. across the network) can
        # still be found.
        def _find_challenge_action(deps: List[Departure], allow_extended: bool = True) -> Optional[int]:
            best: Optional[Tuple[float, int]] = None  # (value, departure_index_in_deps)
            ch_ids = {ch.station_id: ch for ch in game_state.challenges}
            for i, dep in enumerate(deps):
                dest = dep.destination_stop_id
                if dest not in ch_ids:
                    continue
                cost = compute_route_chip_cost(
                    stations, dep.intermediate_stops, team_id,
                    starting_id or "",
                    extra_chips=self.extra_chips,
                    max_chips_per_station=self.max_chips,
                )
                if not (team.coins >= cost or team.coins <= 0):
                    continue
                val = ch_ids[dest].current_value()
                if best is None or val > best[0]:
                    best = (val, i)

            if best is not None:
                return best[1]

            # Widen search window and use a larger K so distant challenge
            # stations (e.g. in another borough) are also discovered.
            if allow_extended and self.challenge_window > self.reachability_window:
                ext_deps = get_valid_departures(
                    rail_network,
                    team.current_station,
                    game_state.sim_minute,
                    self.challenge_window,
                    k=self.challenge_k,
                )
                return _find_challenge_action(ext_deps, allow_extended=False)
            return None

        # 2. When coins are very low, seek a challenge FIRST to refill before
        # getting stuck.  This supersedes expansion so the agent never depletes
        # its budget so far from a challenge that it can no longer move.
        if 0 < team.coins <= self.low_coins_threshold and game_state.challenges:
            ch_action = _find_challenge_action(available_departures)
            if ch_action is not None:
                return ch_action

        # 3. Board toward nearest uncontrolled station — only when we have
        # coins to actually claim it.  With zero coins the team can travel
        # for free but would never place chips, so step 3 is skipped and we
        # fall through to challenge-seeking (step 4) to replenish coins.
        if team.coins > 0:
            best_hop: Optional[Tuple[int, int]] = None  # (action_idx, hop_distance)
            for idx, dep, cost in affordable:
                for hop, stop_id in enumerate(dep.intermediate_stops):
                    st = stations.get(stop_id)
                    if st is None or stop_id == starting_id:
                        continue
                    if st.controlling_team() is None:
                        if best_hop is None or hop < best_hop[1]:
                            best_hop = (idx, hop)
                        break  # only the first uncontrolled stop per route matters
            if best_hop is not None:
                return best_hop[0]

        # 4. Board directly to a challenge station (departure must end there).
        # Uses the wider challenge_window if needed.
        # Prioritised over contesting when broke so the agent can recover coins.
        if game_state.challenges:
            ch_action = _find_challenge_action(available_departures)
            if ch_action is not None:
                return ch_action

        # 5. Contest nearest opponent-controlled station we can afford.
        if team.coins > 0:
            for idx, dep, cost in affordable:
                for stop_id in dep.intermediate_stops:
                    st = stations.get(stop_id)
                    if st and st.controlling_team() == opponent_id:
                        return idx

        # 6. Fallback: keep moving to avoid getting stranded.
        #
        # When no useful action is found the agent would otherwise wait
        # indefinitely at an end-of-line station.  Instead it returns to
        # the starting hub (usually a major transfer station) which opens
        # up departure options to the rest of the network.  If the starting
        # station is not reachable in the current window, take any affordable
        # move so the agent at least reaches new territory.
        if affordable:
            # Prefer routes that end at or pass through the starting hub.
            for idx, dep, cost in affordable:
                if dep.destination_stop_id == starting_id:
                    return idx
                if starting_id in dep.intermediate_stops:
                    return idx
            # Otherwise take any affordable departure.
            return affordable[0][0]

        # 7. Truly nothing to do — wait.
        return ACTION_WAIT
