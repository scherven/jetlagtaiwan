"""
Core game rules: chip placement, challenge resolution, win conditions.
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from typing import Dict, List, Optional, Tuple

from engine.game_state import Challenge, GameState, Station, Team
from engine.rail_network import Departure, RailNetwork
from engine.clock import DAY_END_MINUTE, sim_minute_to_wall_clock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chip cost calculation
# ---------------------------------------------------------------------------

def compute_route_chip_cost(
    stations: Dict[str, Station],
    stop_ids: List[str],
    team_id: str,
    starting_station_id: str,
) -> int:
    """
    Compute total chip cost for travelling through stop_ids (in order).
    Returns the number of chips needed at each stop:
      - 1 chip for neutral or own station
      - opponent_chips + 1 for opponent-controlled stations
    Starting station is free (skip it from cost).
    """
    total = 0
    opponent_id = "B" if team_id == "A" else "A"
    chip_attr = f"chips_team_{team_id.lower()}"
    opp_attr = f"chips_team_{opponent_id.lower()}"

    for stop_id in stop_ids:
        if stop_id == starting_station_id:
            continue
        station = stations.get(stop_id)
        if station is None:
            continue
        our_chips = getattr(station, chip_attr)
        opp_chips = getattr(station, opp_attr)
        if opp_chips > our_chips:
            # Opponent controls: need to exceed
            cost = opp_chips - our_chips + 1
        else:
            cost = 1  # neutral or we already control: minimum 1 chip
        total += cost
    return total


# ---------------------------------------------------------------------------
# Chip placement
# ---------------------------------------------------------------------------

def place_chips_at_stop(
    station: Station,
    team: Team,
    starting_station_id: str,
    own_station_penalty: int = 1,
    enemy_station_penalty: int = 3,
) -> None:
    """
    Place chips for team at a single stop according to the game rules.
    Modifies station.chips_team_* and team.coins in place.
    """
    if station.id == starting_station_id:
        return  # starting station is always free and neutral

    chip_attr = f"chips_team_{team.id.lower()}"
    opp_attr = f"chips_team_{'b' if team.id == 'A' else 'a'}"

    our_chips = getattr(station, chip_attr)
    opp_chips = getattr(station, opp_attr)

    if team.coins <= 0:
        # Zero-coins travel rule: pay coin penalty only — cannot contest.
        # if opp_chips > our_chips:
        #     team.coins -= enemy_station_penalty   # bleed 3 coins, place nothing
        #     chips_to_place = 0
        # else:
        #     team.coins -= own_station_penalty     # bleed 1 coin, place 1 chip minimum
        #     chips_to_place = 1
        chips_to_place = 0
    else:
        if opp_chips > our_chips:
            chips_to_place = opp_chips - our_chips + 1
        elif opp_chips == our_chips:
            chips_to_place = 1
        else:
            chips_to_place = 0
        team.coins -= chips_to_place

    if chips_to_place > 0:
        setattr(station, chip_attr, our_chips + chips_to_place)
    logger.debug(
        "Team %s placed %d chips at %s (balance now %d)",
        team.id, chips_to_place, station.name, team.coins,
    )


# ---------------------------------------------------------------------------
# Challenge resolution
# ---------------------------------------------------------------------------

def resolve_challenge(
    challenge: Challenge,
    acting_team: Team,
    opponent: Team,
    current_day: int,
    daily_multiplier: float = 1.10,
) -> str:
    """
    Resolve a challenge for acting_team. Returns a human-readable result string.
    Modifies team balances in place.
    """
    value = challenge.current_value(daily_multiplier)

    if challenge.type == "chip_gain":
        gain = int(math.floor(value))
        acting_team.coins += gain
        return f"Team {acting_team.id} gained {gain} coins from chip_gain challenge."

    elif challenge.type == "steal":
        if opponent.coins > 0:
            stolen = int(math.floor(opponent.coins * value))
            opponent.coins -= stolen
            acting_team.coins += stolen
            return (
                f"Team {acting_team.id} stole {stolen} coins from Team {opponent.id}."
            )
        else:
            return f"Team {acting_team.id} attempted steal but opponent has non-positive balance."

    return f"Unknown challenge type: {challenge.type}"


def spawn_challenges(
    game_state: GameState,
    starting_station_id: str,
    chip_gain_base: int = 50,
    steal_fraction: float = 0.20,
    steal_probability: float = 0.25,
    count: int = 1,
) -> List[Challenge]:
    """
    Spawn `count` new challenges on random stations (no duplicates, not on starting station).
    Returns list of newly spawned challenges.
    """
    occupied = {c.station_id for c in game_state.challenges}
    candidates = [
        sid
        for sid in game_state.stations
        if sid not in occupied and sid != starting_station_id
    ]
    random.shuffle(candidates)
    spawned = []
    for sid in candidates[:count]:
        ctype = "steal" if random.random() < steal_probability else "chip_gain"
        base = steal_fraction if ctype == "steal" else chip_gain_base
        c = Challenge(
            id=str(uuid.uuid4()),
            station_id=sid,
            type=ctype,
            base_value=base if ctype == "chip_gain" else int(base * 100),
            day=game_state.day,
        )
        spawned.append(c)
    return spawned


# ---------------------------------------------------------------------------
# Win condition
# ---------------------------------------------------------------------------

def count_controlled_stations(
    game_state: GameState,
    team_id: str,
    starting_station_id: str,
) -> int:
    """Count how many stations team controls (excluding starting station)."""
    count = 0
    for sid, station in game_state.stations.items():
        if sid == starting_station_id:
            continue
        if station.controlling_team() == team_id:
            count += 1
    return count


def compute_winner(
    game_state: GameState,
    starting_station_id: str,
) -> Optional[str]:
    """Return 'A', 'B', 'tie', or None if game not over."""
    a = count_controlled_stations(game_state, "A", starting_station_id)
    b = count_controlled_stations(game_state, "B", starting_station_id)
    if a > b:
        return "A"
    elif b > a:
        return "B"
    return "tie"


# ---------------------------------------------------------------------------
# Departure filtering
# ---------------------------------------------------------------------------

def get_valid_departures(
    rail_network: RailNetwork,
    station_id: str,
    current_sim_minute: int,
    window_minutes: int,
    day_end_wall: int = DAY_END_MINUTE,
    k: int = 10,
) -> List[Departure]:
    """
    Return up to k departure options from station_id within the next window_minutes.

    Each option is a (train, chosen-destination) pair — teams ride only as far
    as the chosen destination, not to the train's terminus. This allows agents to
    make short hops without being forced onto long expensive trips.

    One option is generated per unique reachable destination (earliest departure
    wins). Options are sorted by arrival time so the nearest destinations come first.
    """
    from engine.clock import sim_minute_to_wall_clock

    current_wall = sim_minute_to_wall_clock(current_sim_minute)
    until_wall = current_wall + window_minutes

    raw = rail_network.departures_from(station_id, current_wall, until_wall)

    # Build one truncated departure per unique destination: earliest departure,
    # covering only the stops up to and including that destination.
    by_dest: Dict[str, Departure] = {}
    for dep in raw:
        for j, (stop, arr) in enumerate(
            zip(dep.intermediate_stops, dep.arrival_minutes)
        ):
            if arr > day_end_wall:
                break
            if stop == station_id:
                continue
            if stop not in by_dest:
                by_dest[stop] = Departure(
                    trip_id=dep.trip_id,
                    route_id=dep.route_id,
                    departure_minute=dep.departure_minute,
                    destination_stop_id=stop,
                    intermediate_stops=list(dep.intermediate_stops[: j + 1]),
                    arrival_minutes=[int(a) for a in dep.arrival_minutes[: j + 1]],
                )

    # Sort by arrival time (nearest first), then by departure time
    options = sorted(
        by_dest.values(),
        key=lambda d: (d.arrival_minutes[-1], d.departure_minute),
    )
    return options[:k]
