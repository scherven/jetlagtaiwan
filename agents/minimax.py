"""
Minimax agent with alpha-beta pruning.

Decision structure:
  1. Always attempt a challenge if present at current station.
  2. Run a depth-limited minimax search over candidate departures.
       Nodes alternate: our team (maximise) → opponent (minimise).
       Each node simulates one departure: chip placement along route
       plus position advance.  No GTFS step-loop overhead.
  3. Return the departure index with the best minimax value.

Depth guide:
  depth=1  — my move then evaluate (fast, still better than pure heuristic)
  depth=2  — my move → opponent best response → evaluate  (recommended)
  depth=3+ — expensive; keep branch_factor ≤ 3

Departures are pre-ranked by a fast heuristic score (uncontrolled stations
first, then challenge proximity) to maximise alpha-beta pruning efficiency.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Dict, List, Optional, Tuple

from engine.clock import DAY_DURATION, DAY_END_MINUTE, wall_clock_to_sim_minute
from engine.game_state import GameState, Station, Team
from engine.rail_network import Departure, RailNetwork
from engine.rules import (
    compute_route_chip_cost,
    count_controlled_stations,
    get_valid_departures,
    place_chips_at_stop,
)

logger = logging.getLogger(__name__)

NEG_INF = float("-inf")
POS_INF = float("inf")


class MinimaxAgent:
    """
    Minimax agent implementing the same action interface as HeuristicAgent and
    the RL agent.

    choose_action(game_state, rail_network, team_id, departures) -> int
    """

    def __init__(
        self,
        config: dict,
        depth: int = 2,
        branch_factor: int = 5,
    ):
        self.k = config["agents"]["max_departures_k"]
        self.depth = depth
        self.branch_factor = branch_factor
        self.max_chips: int = config["game"].get("max_chips_per_station", 5)
        self.reachability_window: int = config["agents"]["heuristic_reachability_window"]
        self.starting_coins: int = config["game"]["starting_coins"]
        self.starting_station_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Public entry point
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
        if team.is_in_transit():
            return ACTION_WAIT

        # Always attempt a challenge immediately if one is here.
        if any(c.station_id == team.current_station for c in game_state.challenges):
            return ACTION_CHALLENGE

        if not available_departures:
            return ACTION_WAIT

        opponent_id = "B" if team_id == "A" else "A"

        best_action = ACTION_WAIT
        best_score = NEG_INF
        alpha = NEG_INF
        beta = POS_INF

        # Rank and limit candidates to control branching factor.
        candidates = self._rank_departures(
            game_state, team_id, available_departures
        )[: self.branch_factor]

        for action_idx, dep in candidates:
            cost = compute_route_chip_cost(
                game_state.stations,
                dep.intermediate_stops,
                team_id,
                self.starting_station_id or "",
                max_chips_per_station=self.max_chips,
            )
            if team.coins > 0 and cost > team.coins:
                continue

            child = self._simulate_departure(game_state, team_id, dep)
            score = self._search(
                child,
                rail_network,
                opponent_id,
                self.depth - 1,
                alpha,
                beta,
                team_id,  # root team — we always evaluate from this perspective
            )

            if score > best_score:
                best_score = score
                best_action = action_idx

            alpha = max(alpha, score)
            if beta <= alpha:
                break  # root is always a max node, no need for beta cutoff here

        return best_action

    # ------------------------------------------------------------------
    # Alpha-beta search
    # ------------------------------------------------------------------

    def _search(
        self,
        state: GameState,
        rail_network: RailNetwork,
        current_team: str,
        depth: int,
        alpha: float,
        beta: float,
        root_team: str,
    ) -> float:
        """
        Recursive minimax with alpha-beta pruning.

        current_team — whose turn it is this ply.
        root_team    — the agent we are optimising for (perspective of evaluation).
        """
        if depth == 0:
            return self._evaluate(state, root_team)

        opponent = "B" if current_team == "A" else "A"
        team = state.teams[current_team]

        # Check for challenge at current position first.
        ch = next(
            (c for c in state.challenges if c.station_id == team.current_station),
            None,
        )
        if ch is not None:
            child = self._simulate_challenge(state, current_team, ch)
            return self._search(child, rail_network, opponent, depth - 1,
                                alpha, beta, root_team)

        deps = get_valid_departures(
            rail_network,
            team.current_station,
            state.sim_minute,
            self.reachability_window,
            k=self.branch_factor,
        )

        if not deps:
            return self._evaluate(state, root_team)

        candidates = self._rank_departures(state, current_team, deps)[: self.branch_factor]
        maximising = (current_team == root_team)

        if maximising:
            best = NEG_INF
            for _, dep in candidates:
                cost = compute_route_chip_cost(
                    state.stations, dep.intermediate_stops, current_team,
                    self.starting_station_id or "",
                    max_chips_per_station=self.max_chips,
                )
                if state.teams[current_team].coins > 0 and cost > state.teams[current_team].coins:
                    continue
                child = self._simulate_departure(state, current_team, dep)
                val = self._search(child, rail_network, opponent,
                                   depth - 1, alpha, beta, root_team)
                best = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best if best > NEG_INF else self._evaluate(state, root_team)
        else:
            best = POS_INF
            for _, dep in candidates:
                cost = compute_route_chip_cost(
                    state.stations, dep.intermediate_stops, current_team,
                    self.starting_station_id or "",
                    max_chips_per_station=self.max_chips,
                )
                if state.teams[current_team].coins > 0 and cost > state.teams[current_team].coins:
                    continue
                child = self._simulate_departure(state, current_team, dep)
                val = self._search(child, rail_network, opponent,
                                   depth - 1, alpha, beta, root_team)
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best if best < POS_INF else self._evaluate(state, root_team)

    # ------------------------------------------------------------------
    # State evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, state: GameState, our_team: str) -> float:
        """
        Board-position score from our_team's perspective.

        Primary: station control difference.
        Secondary (small weight): coin lead — more coins → more future moves.
        Tertiary: proximity bonus for challenges near our current station.
        """
        opp = "B" if our_team == "A" else "A"
        sid = self.starting_station_id or ""

        our_stations = count_controlled_stations(state, our_team, sid)
        opp_stations = count_controlled_stations(state, opp, sid)

        our_coins = max(state.teams[our_team].coins, 0)
        opp_coins = max(state.teams[opp].coins, 0)
        coin_score = (our_coins - opp_coins) / max(self.starting_coins, 1)

        return float(our_stations - opp_stations) + 0.05 * coin_score

    # ------------------------------------------------------------------
    # Departure ranking (search-order heuristic for better pruning)
    # ------------------------------------------------------------------

    def _rank_departures(
        self,
        state: GameState,
        team_id: str,
        departures: List[Departure],
    ) -> List[Tuple[int, Departure]]:
        """
        Score each departure quickly so the most promising ones come first,
        improving alpha-beta pruning efficiency.

        Scoring:
          +2.0 per uncontrolled station along the route
          +0.5 per opponent-controlled station (contesting is good too)
          +3.0 if the destination has a challenge
        """
        opp_id = "B" if team_id == "A" else "A"
        sid = self.starting_station_id or ""
        ch_stations = {c.station_id for c in state.challenges}

        scored: List[Tuple[float, int, Departure]] = []
        for idx, dep in enumerate(departures[: self.k]):
            score = 0.0
            for stop_id in dep.intermediate_stops:
                if stop_id == sid:
                    continue
                st = state.stations.get(stop_id)
                if st is None:
                    continue
                ctrl = st.controlling_team()
                if ctrl is None:
                    score += 2.0
                elif ctrl == opp_id:
                    score += 1.5

            if dep.destination_stop_id in ch_stations:
                score += 3.0

            scored.append((score, idx, dep))

        scored.sort(key=lambda x: -x[0])
        return [(idx, dep) for _, idx, dep in scored]

    # ------------------------------------------------------------------
    # State simulation helpers
    # ------------------------------------------------------------------

    def _simulate_departure(
        self,
        state: GameState,
        team_id: str,
        dep: Departure,
    ) -> GameState:
        """
        Clone state, place chips at each valid stop along dep, advance
        team's position to the last reachable stop before day end.
        Sim clock advances to the arrival time at that stop.
        """
        new_state = self._clone_state(state)
        team = new_state.teams[team_id]
        sid = self.starting_station_id or ""

        last_valid_stop: Optional[str] = None
        last_valid_arr: Optional[int] = None

        for stop_id, arr_minute in zip(dep.intermediate_stops, dep.arrival_minutes):
            if arr_minute > DAY_END_MINUTE:
                break
            station = new_state.stations.get(stop_id)
            if station is None:
                continue
            place_chips_at_stop(
                station,
                team,
                sid,
                extra_chips=0,
                max_chips_per_station=self.max_chips,
                game_state=new_state,
            )
            last_valid_stop = stop_id
            last_valid_arr = arr_minute

        if last_valid_stop is not None:
            team.current_station = last_valid_stop
            team.destination_station = None
            team.arrival_time = None
            new_state.sim_minute = wall_clock_to_sim_minute(last_valid_arr)  # type: ignore[arg-type]

        return new_state

    def _simulate_challenge(
        self,
        state: GameState,
        team_id: str,
        challenge,
    ) -> GameState:
        """
        Clone state and apply the expected outcome of completing a challenge.
        Challenges are removed from the board after completion.
        """
        new_state = self._clone_state(state)
        team = new_state.teams[team_id]

        if challenge.type == "chip_gain":
            gain = int(math.floor(challenge.current_value()))
            team.coins += gain
        elif challenge.type == "steal":
            opp_id = team.opponent_id()
            opp = new_state.teams[opp_id]
            if opp.coins > 0:
                stolen = int(opp.coins * challenge.base_value)
                opp.coins -= stolen
                team.coins += stolen

        new_state.challenges = [
            c for c in new_state.challenges if c.id != challenge.id
        ]
        return new_state

    def _clone_state(self, state: GameState) -> GameState:
        """
        Lightweight clone of the mutable game state.

        Copies station chip counts, team balances/positions, and challenge
        list.  The rail network and config are never copied (they are
        read-only for the purposes of lookahead).
        """
        # Directly construct Station/Team objects instead of copy.copy() —
        # avoids the generic copy machinery (573k copy.copy calls per eval).
        new_stations = {
            sid: Station(
                id=s.id, name=s.name, lat=s.lat, lon=s.lon,
                chips_team_a=s.chips_team_a, chips_team_b=s.chips_team_b,
            )
            for sid, s in state.stations.items()
        }
        new_teams = {
            tid: Team(
                id=t.id, coins=t.coins,
                current_station=t.current_station,
                destination_station=t.destination_station,
                arrival_time=t.arrival_time,
                remaining_stops=list(t.remaining_stops),
                desired_extra_chips=t.desired_extra_chips,
            )
            for tid, t in state.teams.items()
        }

        return GameState(
            day=state.day,
            sim_minute=state.sim_minute,
            stations=new_stations,
            teams=new_teams,
            challenges=list(state.challenges),
            # _ctrl_cache left empty — count_controlled_stations recomputes lazily
        )
