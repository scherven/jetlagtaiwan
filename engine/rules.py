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
    chips_per_stop: Optional[List[int]] = None,
    max_chip_differential: int = 5,
) -> int:
    """
    Compute total chip cost for travelling through stop_ids (in order).

    The cap rule: after placing, our chips may not exceed opp_chips + max_chip_differential.
    So the maximum chips we can place at a stop = max(0, opp_chips + max_chip_differential - our_chips).

    Per-stop cost:
      - 0               if we already control (our > opp) and no extra requested
      - 1 + extra       if neutral (tie), capped by the differential rule
      - (opp-our+1)+extra  if opponent controls, capped by the differential rule

    chips_per_stop: extra coins above the minimum at each stop, parallel to stop_ids.
    Starting station is always free.
    """
    total = 0
    opponent_id = "B" if team_id == "A" else "A"
    chip_attr = f"chips_team_{team_id.lower()}"
    opp_attr = f"chips_team_{opponent_id.lower()}"

    for i, stop_id in enumerate(stop_ids):
        if stop_id == starting_station_id:
            continue
        station = stations.get(stop_id)
        if station is None:
            continue
        our_chips = getattr(station, chip_attr)
        opp_chips = getattr(station, opp_attr)
        extra = (chips_per_stop[i] if chips_per_stop and i < len(chips_per_stop) else 0)

        # Maximum we are allowed to add (differential cap)
        cap = max(0, opp_chips + max_chip_differential - our_chips)

        if our_chips > opp_chips:
            cost = min(extra, cap)
        elif opp_chips > our_chips:
            min_cost = opp_chips - our_chips + 1
            cost = min(min_cost + extra, cap)
        else:
            cost = min(1 + extra, cap)

        total += max(0, cost)
    return total


# ---------------------------------------------------------------------------
# Chip placement
# ---------------------------------------------------------------------------

def place_chips_at_stop(
    station: Station,
    team: Team,
    starting_station_id: str,
    own_station_penalty: int = 1,   # kept for signature compat, not currently used
    enemy_station_penalty: int = 3,  # kept for signature compat, not currently used
    extra_chips: int = 0,
    max_chip_differential: int = 5,
    game_state=None,  # GameState | None — pass to keep control cache consistent
) -> None:
    """
    Place chips for team at a single stop according to the game rules.
    Modifies station.chips_team_* and team.coins in place.

    Cap rule: after placing, our chips ≤ opp_chips + max_chip_differential.
    So the most we can ever add = max(0, opp_chips + max_chip_differential - our_chips).

    Placement rules:
      - Starting station: always free, skip.
      - Own station (our > opp): free entry; can optionally add extra_chips.
      - Neutral station (tie): 1 chip to claim, plus up to extra_chips more.
      - Opponent station (opp > our): minimum to take control, plus up to extra_chips more.
      - Zero-balance: no chips placed (team is broke).
    """
    if station.id == starting_station_id:
        return  # starting station is always free and neutral

    chip_attr = f"chips_team_{team.id.lower()}"
    opp_attr = f"chips_team_{'b' if team.id == 'A' else 'a'}"

    our_chips = getattr(station, chip_attr)
    opp_chips = getattr(station, opp_attr)
    old_controller = station.controlling_team()

    if team.coins <= 0:
        chips_to_place = 0
    else:
        if opp_chips > our_chips:
            min_needed = opp_chips - our_chips + 1
            chips_to_place = min_needed + extra_chips
        elif opp_chips == our_chips:
            chips_to_place = 1 + extra_chips
        else:
            chips_to_place = extra_chips

        # Differential cap: we cannot exceed opp_chips + max_chip_differential.
        cap = max(0, opp_chips + max_chip_differential - our_chips)
        chips_to_place = max(0, min(chips_to_place, cap))
        # Can't spend more than we have.
        chips_to_place = min(chips_to_place, team.coins)
        team.coins -= chips_to_place

    if chips_to_place > 0:
        setattr(station, chip_attr, our_chips + chips_to_place)
        # Update incremental control cache if GameState provided.
        if game_state is not None:
            new_controller = station.controlling_team()
            if new_controller != old_controller:
                cache = game_state._ctrl_cache
                sid = game_state._ctrl_cache_starting_id
                if station.id != sid:
                    if old_controller is not None:
                        cache[old_controller] = cache.get(old_controller, 1) - 1
                    if new_controller is not None:
                        cache[new_controller] = cache.get(new_controller, 0) + 1

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
    reachable_stations: Optional[set] = None,
) -> List[Challenge]:
    """
    Spawn `count` new challenges on random stations (no duplicates, not on starting station).

    reachable_stations: if provided, challenges are only placed at stations in
    this set.  Pass the union of stations reachable from current team positions
    so agents can always navigate to the challenge using a direct single-leg trip.
    Returns list of newly spawned challenges.
    """
    occupied = {c.station_id for c in game_state.challenges}
    candidates = [
        sid
        for sid in game_state.stations
       # if sid != starting_station_id
       #and   sid not in occupied and (reachable_stations is None or sid in reachable_stations)
    ]
    random.shuffle(candidates)
    spawned = []
    for sid in candidates[:count]:
        ctype = "steal" if random.random() < steal_probability else "chip_gain"
        # chip_gain base_value = coins reward (e.g. 50)
        # steal base_value     = fraction of opponent coins (e.g. 0.20 = 20%)
        base_value: float = steal_fraction if ctype == "steal" else float(chip_gain_base)
        c = Challenge(
            id=str(uuid.uuid4()),
            station_id=sid,
            type=ctype,
            base_value=base_value,
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
    """Count how many stations team controls (excluding starting station).

    Uses the incremental cache on GameState when available and warm;
    falls back to a full scan (and warms the cache) otherwise.
    """
    # Warm cache on first call or if starting station changed.
    if (game_state._ctrl_cache_starting_id != starting_station_id
            or not game_state._ctrl_cache):
        counts: dict = {"A": 0, "B": 0}
        for sid, station in game_state.stations.items():
            if sid == starting_station_id:
                continue
            ctrl = station.controlling_team()
            if ctrl is not None:
                counts[ctrl] = counts.get(ctrl, 0) + 1
        game_state._ctrl_cache = counts
        game_state._ctrl_cache_starting_id = starting_station_id

    return game_state._ctrl_cache.get(team_id, 0)


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
                stops_slice = list(dep.intermediate_stops[: j + 1])
                by_dest[stop] = Departure(
                    trip_id=dep.trip_id,
                    route_id=dep.route_id,
                    departure_minute=dep.departure_minute,
                    destination_stop_id=stop,
                    intermediate_stops=stops_slice,
                    arrival_minutes=[int(a) for a in dep.arrival_minutes[: j + 1]],
                    chips_per_stop=[0] * len(stops_slice),
                )

    # Sort by arrival time (nearest first), then by departure time
    options = sorted(
        by_dest.values(),
        key=lambda d: (d.arrival_minutes[-1], d.departure_minute),
    )
    return options[:k]
