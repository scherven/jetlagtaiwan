"""
Greedy heuristic agent — rule-based baseline.

Decision logic (evaluated in priority order):
1. If at a challenge station: always attempt it.
2. If coins are low: seek the highest-value reachable challenge first.
3. Board toward the nearest uncontrolled station (most expansion value), avoiding backtrack.
4. Board directly to the highest-value challenge station (departure must END there).
5. Contest the nearest opponent-controlled station we can afford to visit.
6. Fallback: keep moving (prefer routes through starting hub, then most new territory).
7. Wait.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from engine.game_state import GameState, Team
from engine.rail_network import Departure, RailNetwork
from engine.rules import compute_route_chip_cost, get_valid_departures

logger = logging.getLogger(__name__)


class HeuristicAgent:
    """
    Greedy heuristic agent implementing the same action interface as the RL agent.
    choose_action(game_state, rail_network, team_id, departures) -> int (0..K+1)
    """

    def __init__(self, config: dict):
        self.k = config["agents"]["max_departures_k"]
        self.reachability_window = config["agents"]["heuristic_reachability_window"]
        self.extra_chips: int = config["agents"].get("heuristic_extra_chips", 0)
        self.max_chips: int = config["game"].get("max_chip_differential", 5)
        self.challenge_window: int = config["agents"].get("heuristic_challenge_window", 180)
        self.challenge_k: int = config["agents"].get("heuristic_challenge_k", 50)
        self.low_coins_threshold: int = config["agents"].get("heuristic_low_coins_threshold", 15)
        self.starting_station_id: Optional[str] = None
        # Anti-bounce: track where we most recently departed FROM so we don't
        # immediately turn around and go back.  Updated every time any departure
        # action is chosen (steps 3-6).
        self._last_departed_from: Optional[str] = None

    def reset(self) -> None:
        """Clear per-episode state.  Call between games when reusing an instance."""
        self._last_departed_from = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def choose_action(
        self,
        game_state: GameState,
        rail_network: RailNetwork,
        team_id: str,
        available_departures: List[Departure],
    ) -> int:
        ACTION_CHALLENGE = self.k
        ACTION_WAIT = self.k + 1

        team = game_state.teams[team_id]
        opponent_id = "B" if team_id == "A" else "A"
        stations = game_state.stations
        starting_id = self.starting_station_id

        if team.is_in_transit():
            return ACTION_WAIT  # already moving, no decision needed

        # ----------------------------------------------------------
        # 1. Challenge at current station
        # ----------------------------------------------------------
        challenge_here = next(
            (c for c in game_state.challenges if c.station_id == team.current_station),
            None,
        )
        if challenge_here:
            return ACTION_CHALLENGE

        # ----------------------------------------------------------
        # Pre-compute costs and affordable list
        # ----------------------------------------------------------
        scored: List[Tuple[int, Departure, int]] = []  # (idx, dep, cost)
        for idx, dep in enumerate(available_departures[:self.k]):
            n = len(dep.intermediate_stops)
            cost = compute_route_chip_cost(
                stations,
                dep.intermediate_stops,
                team_id,
                starting_id or "",
                chips_per_stop=[self.extra_chips] * n,
                max_chip_differential=self.max_chips,
            )
            scored.append((idx, dep, cost))

        # Affordable: enough coins to cover cost, OR broke (free travel, but can't claim).
        affordable = [(i, d, c) for i, d, c in scored if team.coins >= c or team.coins <= 0]

        # Non-backtrack affordable: prefer not returning to the station we just left.
        def _no_backtrack(items: List[Tuple[int, Departure, int]]) -> List[Tuple[int, Departure, int]]:
            nb = [x for x in items if x[1].destination_stop_id != self._last_departed_from]
            return nb if nb else items

        def _take_departure(idx: int) -> int:
            """Record departure for anti-bounce, stamp chips_per_stop, return action."""
            self._last_departed_from = team.current_station
            if 0 <= idx < len(available_departures):
                dep = available_departures[idx]
                dep.chips_per_stop = [self.extra_chips] * len(dep.intermediate_stops)
            return idx

        # ----------------------------------------------------------
        # 2. Low-coins: seek a challenge FIRST to avoid coin deadlock
        # ----------------------------------------------------------
        if 0 < team.coins <= self.low_coins_threshold and game_state.challenges:
            ch_action = self._find_challenge_action(
                available_departures, game_state, rail_network, team_id, stations, starting_id
            )
            if ch_action is not None:
                return _take_departure(ch_action)

        # ----------------------------------------------------------
        # 3. Expand to nearest uncontrolled station (only when solvent)
        # ----------------------------------------------------------
        if team.coins > 0:
            best_hop: Optional[Tuple[int, int]] = None  # (action_idx, hop_distance)
            best_hop_nb: Optional[Tuple[int, int]] = None  # non-backtrack best
            for idx, dep, cost in affordable:
                is_backtrack = dep.destination_stop_id == self._last_departed_from
                for hop, stop_id in enumerate(dep.intermediate_stops):
                    st = stations.get(stop_id)
                    if st is None or stop_id == starting_id:
                        continue
                    if st.controlling_team() is None:
                        candidate = (idx, hop)
                        if best_hop is None or hop < best_hop[1]:
                            best_hop = candidate
                        if not is_backtrack and (best_hop_nb is None or hop < best_hop_nb[1]):
                            best_hop_nb = candidate
                        break  # only the first uncontrolled stop per route matters
            # Prefer non-backtrack; fall back to any if needed
            chosen = best_hop_nb if best_hop_nb is not None else best_hop
            if chosen is not None:
                return _take_departure(chosen[0])

        # ----------------------------------------------------------
        # 4. Head directly to the highest-value challenge station
        # ----------------------------------------------------------
        if game_state.challenges:
            ch_action = self._find_challenge_action(
                available_departures, game_state, rail_network, team_id, stations, starting_id
            )
            if ch_action is not None:
                return _take_departure(ch_action)

        # ----------------------------------------------------------
        # 5. Contest nearest opponent-controlled station
        # ----------------------------------------------------------
        if team.coins > 0:
            for idx, dep, cost in _no_backtrack(affordable):
                for stop_id in dep.intermediate_stops:
                    st = stations.get(stop_id)
                    if st and st.controlling_team() == opponent_id:
                        return _take_departure(idx)

        # ----------------------------------------------------------
        # 6. Fallback: keep moving to avoid end-of-line deadlock
        #
        # Priority:
        #   a) Route ending at or passing through the starting hub (opens options)
        #   b) Route with most new territory (uncontrolled stops), longest trip
        #   c) Any non-backtrack affordable departure
        #   d) Any affordable departure
        # ----------------------------------------------------------
        if affordable:
            # (a) Return to starting hub
            nb = _no_backtrack(affordable)
            candidates = nb if nb else affordable
            for idx, dep, cost in candidates:
                if dep.destination_stop_id == starting_id:
                    return _take_departure(idx)
                if starting_id and starting_id in dep.intermediate_stops:
                    return _take_departure(idx)

            # (b) Best-scoring non-backtrack (or any if all are backtracks)
            def _score(item: Tuple[int, Departure, int]) -> Tuple[int, int]:
                _, dep, _ = item
                uncontrolled = sum(
                    1 for s in dep.intermediate_stops
                    if s in stations and stations[s].controlling_team() is None
                    and s != starting_id
                )
                return (uncontrolled, len(dep.intermediate_stops))

            best = max(candidates, key=_score)
            return _take_departure(best[0])

        # ----------------------------------------------------------
        # 7. Nothing to do — wait
        # ----------------------------------------------------------
        return ACTION_WAIT

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_challenge_action(
        self,
        deps: List[Departure],
        game_state: GameState,
        rail_network: RailNetwork,
        team_id: str,
        stations: dict,
        starting_id: Optional[str],
        allow_extended: bool = True,
    ) -> Optional[int]:
        """Return the index into deps of the best departure ending at a challenge station.
        Returns None if no such affordable departure exists.
        """
        team = game_state.teams[team_id]
        ch_ids = {ch.station_id: ch for ch in game_state.challenges}
        best: Optional[Tuple[float, int]] = None  # (value, departure_index)
        for i, dep in enumerate(deps):
            dest = dep.destination_stop_id
            if dest not in ch_ids:
                continue
            cost = compute_route_chip_cost(
                stations, dep.intermediate_stops, team_id,
                starting_id or "",
                chips_per_stop=[self.extra_chips] * len(dep.intermediate_stops),
                max_chip_differential=self.max_chips,
            )
            if not (team.coins >= cost or team.coins <= 0):
                continue
            val = ch_ids[dest].current_value()
            if best is None or val > best[0]:
                best = (val, i)

        if best is not None:
            return best[1]

        # Widen search window to find distant challenge stations.
        if allow_extended and self.challenge_window > self.reachability_window:
            ext_deps = get_valid_departures(
                rail_network,
                game_state.teams[team_id].current_station,
                game_state.sim_minute,
                self.challenge_window,
                k=self.challenge_k,
            )
            return self._find_challenge_action(
                ext_deps, game_state, rail_network, team_id, stations, starting_id,
                allow_extended=False,
            )
        return None
