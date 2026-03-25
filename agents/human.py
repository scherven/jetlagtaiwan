"""
Human player agent.

Integrates with the agent callback interface but pauses the simulation
when a decision is needed so the Pygame UI can collect input.

Usage in main.py:
    agent = HumanAgent(config)
    agent_fn = agent.choose_action   # pass as agent callback

    # Each frame in the UI loop:
    if agent.needs_input():
        sim.state.is_paused = True
        ctx = agent.get_context()   # (game_state, team_id, departures)
        # ... render overlay, handle clicks ...
        # On click:
        agent.queue_action(dep_idx, extra_chips=2)
        sim.state.is_paused = False
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from engine.game_state import GameState
from engine.rail_network import Departure, RailNetwork


class HumanAgent:
    def __init__(self, config: dict):
        k = config["agents"]["max_departures_k"]
        self.ACTION_CHALLENGE: int = k
        self.ACTION_WAIT: int = k + 1

        self._pending_context: Optional[Tuple[GameState, str, List[Departure]]] = None
        self._queued_action: Optional[int] = None
        self._queued_extra_chips: int = 0
        self._skip_remaining: int = 0

    # ------------------------------------------------------------------
    # Agent callback (called by Simulation._query_agent_if_idle)
    # ------------------------------------------------------------------

    def choose_action(
        self,
        game_state: GameState,
        rail_network: RailNetwork,
        team_id: str,
        departures: List[Departure],
    ) -> int:
        # Burning through a skip — return WAIT each step
        if self._skip_remaining > 0:
            self._skip_remaining -= 1
            return self.ACTION_WAIT

        # Consume a queued action (set by the UI after the player clicks)
        if self._queued_action is not None:
            game_state.teams[team_id].desired_extra_chips = self._queued_extra_chips
            action = self._queued_action
            self._queued_action = None
            self._pending_context = None
            return action

        # No decision yet — store context for the UI and wait this step
        self._pending_context = (game_state, team_id, departures)
        return self.ACTION_WAIT

    # ------------------------------------------------------------------
    # UI interface
    # ------------------------------------------------------------------

    def needs_input(self) -> bool:
        """True when the player needs to make a decision this frame."""
        return (
            self._pending_context is not None
            and self._queued_action is None
            and self._skip_remaining == 0
        )

    def get_context(self) -> Optional[Tuple[GameState, str, List[Departure]]]:
        """Return (game_state, team_id, departures) or None."""
        return self._pending_context

    def queue_action(self, action: int, extra_chips: int = 0) -> None:
        """Queue a departure / challenge action chosen by the player."""
        self._queued_action = action
        self._queued_extra_chips = extra_chips
        self._pending_context = None

    def queue_skip(self, minutes: int = 30) -> None:
        """Skip forward the given number of sim minutes by waiting."""
        self._skip_remaining = minutes
        self._pending_context = None
