"""Simulation clock management."""

from __future__ import annotations

DAY_START_MINUTE = 7 * 60 + 30   # 07:30 = 450 minutes since midnight
DAY_END_MINUTE = 17 * 60 + 30    # 17:30 = 1050 minutes since midnight
DAY_DURATION = DAY_END_MINUTE - DAY_START_MINUTE  # 600 minutes


def wall_clock_to_sim_minute(wall_minute: int) -> int:
    """Convert absolute minute-of-day (from midnight) to sim_minute (0 = 07:30)."""
    return wall_minute - DAY_START_MINUTE


def sim_minute_to_wall_clock(sim_minute: int) -> int:
    """Convert sim_minute to absolute minute-of-day."""
    return sim_minute + DAY_START_MINUTE


def sim_minute_to_str(sim_minute: int) -> str:
    """Convert sim_minute to HH:MM display string."""
    total = DAY_START_MINUTE + sim_minute
    h = total // 60
    m = total % 60
    return f"{h:02d}:{m:02d}"


def is_valid_departure(
    dep_wall_minute: int,
    current_sim_minute: int,
    first_arrival_wall_minute: int,
) -> bool:
    """
    A departure is valid if:
    - It hasn't departed yet (dep_wall_minute >= current wall clock)
    - The train reaches at least one more stop before 17:30
    """
    current_wall = sim_minute_to_wall_clock(current_sim_minute)
    return (
        dep_wall_minute >= current_wall
        and first_arrival_wall_minute <= DAY_END_MINUTE
    )
