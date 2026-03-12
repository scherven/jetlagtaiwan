from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Station:
    id: str
    name: str
    lat: float
    lon: float
    chips_team_a: int = 0
    chips_team_b: int = 0

    def controlling_team(self) -> Optional[str]:
        """Return 'A', 'B', or None if neutral."""
        if self.chips_team_a > self.chips_team_b:
            return "A"
        elif self.chips_team_b > self.chips_team_a:
            return "B"
        return None


@dataclass
class Team:
    id: str                              # "A" or "B"
    coins: int                           # starts at 50
    current_station: str
    destination_station: Optional[str] = None
    arrival_time: Optional[int] = None   # simulation minute of final destination
    remaining_stops: List[str] = field(default_factory=list)  # ordered upcoming stop IDs on current journey
    desired_extra_chips: int = 0         # extra chips to place above minimum on neutral/contested stops

    def is_in_transit(self) -> bool:
        return self.destination_station is not None

    def opponent_id(self) -> str:
        return "B" if self.id == "A" else "A"


@dataclass
class Challenge:
    id: str
    station_id: str
    type: str          # "chip_gain" | "steal"
    base_value: float  # coins for chip_gain (e.g. 50); fraction for steal (e.g. 0.20)
    day: int           # which day it was spawned (for multiplier)

    def current_value(self, daily_multiplier: float = 1.10) -> float:
        return self.base_value * (daily_multiplier ** (self.day - 1))


@dataclass
class GameState:
    day: int          # 1–5
    sim_minute: int   # 0–599 (7:30 AM = 0)
    stations: Dict[str, Station]
    teams: Dict[str, Team]
    challenges: List[Challenge]
    is_paused: bool = False
    speed_multiplier: float = 1.0
    # Cached controlled-station counts (excluding starting station).
    # Updated incrementally by place_chips_at_stop via _update_control_cache().
    _ctrl_cache: Dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 0})
    _ctrl_cache_starting_id: Optional[str] = None

    def sim_minute_to_clock(self) -> str:
        """Convert sim_minute to HH:MM string (0 = 07:30)."""
        total = 7 * 60 + 30 + self.sim_minute
        h = total // 60
        m = total % 60
        return f"{h:02d}:{m:02d}"

    def station_list(self) -> List[Station]:
        """Sorted station list for consistent indexing."""
        return sorted(self.stations.values(), key=lambda s: s.id)
