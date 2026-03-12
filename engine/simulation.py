"""
Core simulation engine.

Advances the clock in 1-minute increments, processes arrivals,
queries agents at decision points, and enforces all game rules.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import yaml

from engine.clock import (
    DAY_DURATION,
    DAY_END_MINUTE,
    DAY_START_MINUTE,
    sim_minute_to_wall_clock,
    sim_minute_to_str,
    wall_clock_to_sim_minute,
)
from engine.game_state import Challenge, GameState, Station, Team
from engine.rail_network import Departure, RailNetwork
from engine.rules import (
    compute_route_chip_cost,
    count_controlled_stations,
    get_valid_departures,
    place_chips_at_stop,
    resolve_challenge,
    spawn_challenges,
)

logger = logging.getLogger(__name__)

# Agent callback type: (game_state, rail_network, team_id, departures) -> action int
AgentFn = Callable[[GameState, RailNetwork, str, List[Departure]], int]


@dataclass
class _TransitEntry:
    """Upcoming stops for a team currently in transit."""
    trip_id: str
    stops: List[str]            # canonical station IDs remaining (incl. destination)
    wall_minutes: List[int]     # wall-clock arrival minute at each stop


@dataclass
class _ChallengeAttempt:
    """Tracks an in-progress challenge attempt."""
    challenge_id: str
    complete_wall_minute: int


class Simulation:
    """
    Manages one complete game (5 days).

    Usage:
        sim = Simulation(config, rail_network, agent_a, agent_b)
        for event_log in sim.run():
            render(sim.state)
    """

    def __init__(
        self,
        config: dict,
        rail_network: RailNetwork,
        agent_a: AgentFn,
        agent_b: AgentFn,
    ):
        self.config = config
        self.net = rail_network
        self.agents: Dict[str, AgentFn] = {"A": agent_a, "B": agent_b}

        self._transit: Dict[str, _TransitEntry] = {}
        self._challenge_attempts: Dict[str, _ChallengeAttempt] = {}

        self.state: GameState = self._new_game_state()
        self.log: List[str] = []          # human-readable event log
        self.done: bool = False
        # When True, step() skips agent queries (AEC env drives them externally)
        self.skip_agent_queries: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> GameState:
        self._transit.clear()
        self._challenge_attempts.clear()
        self.log.clear()
        self.done = False
        self.state = self._new_game_state()
        return self.state

    def step(self) -> List[str]:
        """
        Advance the simulation by one minute.
        Returns a list of event strings that occurred this minute.
        Returns empty list if game is paused or already done.
        """
        if self.done or self.state.is_paused:
            return []

        events: List[str] = []
        current_wall = sim_minute_to_wall_clock(self.state.sim_minute)

        # 1. Process arrivals for teams in transit
        for team_id in ("A", "B"):
            events += self._process_arrivals(team_id, current_wall)

        # 2. Process completed challenge attempts
        for team_id in ("A", "B"):
            events += self._process_challenge_completion(team_id, current_wall)

        # 3. Query idle agents for decisions (skipped when AEC env drives externally)
        if not self.skip_agent_queries:
            for team_id in ("A", "B"):
                events += self._query_agent_if_idle(team_id)

        # 4. Advance clock
        self.state.sim_minute += 1

        # 5. Check day boundary
        if self.state.sim_minute >= DAY_DURATION:
            events += self._end_day()

        self.log.extend(events)
        return events

    def run(self):
        """
        Generator — yields after each sim step.
        Caller is responsible for timing (wall-clock speed control).
        """
        while not self.done:
            events = self.step()
            yield events

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _new_game_state(self) -> GameState:
        gcfg = self.config["game"]
        start_name = gcfg["starting_station"]
        start_node = self.net.station_by_name(start_name)
        if start_node is None:
            raise ValueError(
                f"Starting station {start_name!r} not found in rail network. "
                "Check config.yaml starting_station."
            )
        self._starting_station_id = start_node.id

        # Pre-compute the full-day reachable set from the starting station.
        # Challenges are only spawned here so agents can always reach them in
        # a single-leg trip (no transfers required).
        self._starting_reachable: set = self._compute_reachable(start_node.id)
        logger.info(
            "Reachable from starting station: %d / %d stations.",
            len(self._starting_reachable), len(self.net.stations),
        )

        stations: Dict[str, Station] = {
            sid: Station(id=sid, name=node.name, lat=node.lat, lon=node.lon)
            for sid, node in self.net.stations.items()
        }

        coins = gcfg["starting_coins"]
        teams = {
            "A": Team(
                id="A",
                coins=coins,
                current_station=self._starting_station_id,
            ),
            "B": Team(
                id="B",
                coins=coins,
                current_station=self._starting_station_id,
            ),
        }

        state = GameState(
            day=1,
            sim_minute=0,
            stations=stations,
            teams=teams,
            challenges=[],
        )

        # Spawn initial challenges (restricted to single-leg reachable stations)
        chal_cfg = self.config["challenges"]
        initial = spawn_challenges(
            state,
            self._starting_station_id,
            chip_gain_base=chal_cfg["chip_gain_base"],
            steal_fraction=chal_cfg["steal_fraction"],
            steal_probability=chal_cfg["steal_probability"],
            count=chal_cfg["initial_count"],
            reachable_stations=self._starting_reachable,
        )
        state.challenges.extend(initial)
        logger.info("Game initialised. %d challenges spawned.", len(initial))
        return state

    # ------------------------------------------------------------------
    # Reachability helpers
    # ------------------------------------------------------------------

    def _compute_reachable(self, station_id: str) -> set:
        """Return the set of station IDs reachable from station_id in a single trip
        over the full operating day (no time-window restriction)."""
        from engine.clock import DAY_DURATION
        deps = get_valid_departures(
            self.net, station_id, 0,
            window_minutes=DAY_DURATION,
            k=2000,
        )
        return {d.destination_stop_id for d in deps}

    def _compute_reachable_from_teams(self) -> set:
        """Return the union of stations reachable from each team's current position."""
        reachable: set = set()
        for team in self.state.teams.values():
            reachable |= self._compute_reachable(team.current_station)
        return reachable

    # ------------------------------------------------------------------
    # Transit processing
    # ------------------------------------------------------------------

    def _process_arrivals(self, team_id: str, current_wall: int) -> List[str]:
        """Deliver chips at any stops whose wall-clock arrival minute == current_wall."""
        entry = self._transit.get(team_id)
        if entry is None:
            return []

        team = self.state.teams[team_id]
        events = []
        tcfg = self.config["travel"]

        while entry.stops and entry.wall_minutes[0] <= current_wall:
            stop_id = entry.stops.pop(0)
            arr_min = entry.wall_minutes.pop(0)

            # Skip if past day end (cancelled journey)
            if arr_min > DAY_END_MINUTE:
                break

            station = self.state.stations.get(stop_id)
            if station is None:
                continue

            place_chips_at_stop(
                station, team, self._starting_station_id,
                own_station_penalty=tcfg["own_station_penalty"],
                enemy_station_penalty=tcfg["enemy_station_penalty"],
                extra_chips=team.desired_extra_chips,
                max_chips_per_station=self.config["game"].get("max_chips_per_station", 5),
            )

            team.current_station = stop_id
            events.append(
                f"Team {team_id} arrived at {station.name!r} "
                f"(coins={team.coins})"
            )

            if not entry.stops:
                # Reached final destination
                team.destination_station = None
                team.arrival_time = None
                team.remaining_stops = []
                del self._transit[team_id]
                events.append(f"Team {team_id} completed journey at {station.name!r}.")
                break

        return events

    # ------------------------------------------------------------------
    # Challenge handling
    # ------------------------------------------------------------------

    def _process_challenge_completion(self, team_id: str, current_wall: int) -> List[str]:
        attempt = self._challenge_attempts.get(team_id)
        if attempt is None or current_wall < attempt.complete_wall_minute:
            return []

        del self._challenge_attempts[team_id]
        chal = next(
            (c for c in self.state.challenges if c.id == attempt.challenge_id), None
        )
        if chal is None:
            return []

        team = self.state.teams[team_id]
        opponent = self.state.teams[team.opponent_id()]

        result = resolve_challenge(
            chal, team, opponent,
            self.state.day,
            daily_multiplier=self.config["challenges"]["daily_multiplier"],
        )
        self.state.challenges.remove(chal)

        # Spawn new challenges at stations reachable from current team positions
        # so the replacement challenges are always findable.
        chal_cfg = self.config["challenges"]
        n_spawn = 2 if len(self.state.challenges) < chal_cfg["spawn_threshold"] else 1
        reachable = self._compute_reachable_from_teams()
        new_chals = spawn_challenges(
            self.state,
            self._starting_station_id,
            chip_gain_base=chal_cfg["chip_gain_base"],
            steal_fraction=chal_cfg["steal_fraction"],
            steal_probability=chal_cfg["steal_probability"],
            count=n_spawn,
            reachable_stations=reachable,
        )
        self.state.challenges.extend(new_chals)

        return [result, f"Spawned {len(new_chals)} new challenge(s)."]

    # ------------------------------------------------------------------
    # Agent querying
    # ------------------------------------------------------------------

    def _query_agent_if_idle(self, team_id: str) -> List[str]:
        """Query the agent for an action if the team is idle (not in transit, not doing challenge)."""
        team = self.state.teams[team_id]
        if team_id in self._transit or team_id in self._challenge_attempts:
            return []

        k = self.config["agents"]["max_departures_k"]
        window = self.config["agents"]["heuristic_reachability_window"]
        current_wall = sim_minute_to_wall_clock(self.state.sim_minute)

        departures = get_valid_departures(
            self.net,
            team.current_station,
            self.state.sim_minute,
            window,
            day_end_wall=DAY_END_MINUTE,
            k=k,
        )

        action = self.agents[team_id](self.state, self.net, team_id, departures)

        k_val = self.config["agents"]["max_departures_k"]
        ACTION_CHALLENGE = k_val
        ACTION_WAIT = k_val + 1

        if action == ACTION_WAIT:
            return []

        if action == ACTION_CHALLENGE:
            return self._attempt_challenge(team_id, current_wall)

        if 0 <= action < len(departures):
            return self._board_train(team_id, departures[action], current_wall)

        return []  # invalid action — ignore

    def _attempt_challenge(self, team_id: str, current_wall: int) -> List[str]:
        team = self.state.teams[team_id]
        chal = next(
            (c for c in self.state.challenges if c.station_id == team.current_station),
            None,
        )
        if chal is None:
            return []

        # Simultaneous arrival: if other team is already attempting this challenge, skip
        other_id = team.opponent_id()
        other_attempt = self._challenge_attempts.get(other_id)
        if other_attempt and other_attempt.challenge_id == chal.id:
            return [f"Team {team_id} arrived at challenge but Team {other_id} is already attempting it."]

        # Coin flip if both arrive simultaneously (handled by order A then B)
        duration = self.config["challenges"]["completion_time_minutes"]
        self._challenge_attempts[team_id] = _ChallengeAttempt(
            challenge_id=chal.id,
            complete_wall_minute=current_wall + duration,
        )
        return [f"Team {team_id} started challenge at {self.state.stations[team.current_station].name!r} (completes in {duration} min)."]

    def _board_train(
        self, team_id: str, departure: Departure, current_wall: int
    ) -> List[str]:
        team = self.state.teams[team_id]
        tcfg = self.config["travel"]

        # Filter stops to those before day end
        valid_stops = []
        valid_arr = []
        for stop, arr in zip(departure.intermediate_stops, departure.arrival_minutes):
            if arr <= DAY_END_MINUTE:
                valid_stops.append(stop)
                valid_arr.append(arr)

        if not valid_stops:
            return []

        # Check affordability (team may board if balance is negative already)
        cost = compute_route_chip_cost(
            self.state.stations, valid_stops, team_id, self._starting_station_id,
            extra_chips=team.desired_extra_chips,
            max_chips_per_station=self.config["game"].get("max_chips_per_station", 5),
        )
        if team.coins > 0 and cost > team.coins:
            return [f"Team {team_id} cannot afford train to {self.state.stations.get(valid_stops[-1], type('', (), {'name': valid_stops[-1]})()).name!r} (cost={cost}, coins={team.coins})."]

        # Board
        dest_id = valid_stops[-1]
        dest_name = self.state.stations[dest_id].name if dest_id in self.state.stations else dest_id
        dep_h, dep_m = divmod(departure.departure_minute, 60)

        team.destination_station = dest_id
        team.arrival_time = wall_clock_to_sim_minute(valid_arr[-1])
        team.remaining_stops = list(valid_stops)

        self._transit[team_id] = _TransitEntry(
            trip_id=departure.trip_id,
            stops=list(valid_stops),
            wall_minutes=list(valid_arr),
        )

        return [
            f"Team {team_id} boarded train {departure.trip_id[-12:]!r} "
            f"at {dep_h:02d}:{dep_m:02d} → {dest_name!r} ({len(valid_stops)} stops, cost={cost})."
        ]

    # ------------------------------------------------------------------
    # Day boundary
    # ------------------------------------------------------------------

    def _end_day(self) -> List[str]:
        events = [f"=== Day {self.state.day} ended ==="]

        # Cancel in-progress journeys; teams return to last completed stop
        for team_id in ("A", "B"):
            entry = self._transit.pop(team_id, None)
            if entry:
                team = self.state.teams[team_id]
                team.destination_station = None
                team.arrival_time = None
                team.remaining_stops = []
                events.append(
                    f"Team {team_id}'s journey cancelled; returned to {self.state.stations.get(team.current_station, type('', (), {'name': team.current_station})()).name!r}."
                )
            self._challenge_attempts.pop(team_id, None)

        if self.state.day >= self.config["game"]["num_days"]:
            self._end_game(events)
        else:
            self.state.day += 1
            self.state.sim_minute = 0
            events.append(f"=== Day {self.state.day} started ===")

        return events

    def _end_game(self, events: List[str]) -> None:
        self.done = True
        a = count_controlled_stations(self.state, "A", self._starting_station_id)
        b = count_controlled_stations(self.state, "B", self._starting_station_id)
        if a > b:
            winner = "Team A"
        elif b > a:
            winner = "Team B"
        else:
            winner = "Tie"
        events.append(f"=== GAME OVER: {winner} wins! (A={a}, B={b}) ===")
        logger.info("Game over. %s. A=%d B=%d", winner, a, b)

    # ------------------------------------------------------------------
    # Helpers for UI/agents
    # ------------------------------------------------------------------

    def get_available_departures(self, team_id: str) -> List[Departure]:
        team = self.state.teams[team_id]
        k = self.config["agents"]["max_departures_k"]
        window = self.config["agents"]["heuristic_reachability_window"]
        return get_valid_departures(
            self.net,
            team.current_station,
            self.state.sim_minute,
            window,
            day_end_wall=DAY_END_MINUTE,
            k=k,
        )
