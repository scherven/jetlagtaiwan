"""
PettingZoo AEC + Parallel environment wrappers for the rail strategy game.

Observation space : flat float32 vector, size = 14 + K*7 + C_MAX*5
Action space      : Discrete(K+2)  — 0..K-1 board departure, K challenge, K+1 wait

Both environments expose action_masks() for use with MaskablePPO:
  - RailGameEnv (AEC)      → action_masks() returns (K+2,) bool array for current agent
  - RailGameParallelEnv    → action_masks() returns {agent: (K+2,) bool array}
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import wrappers

from agents.eval import C_MAX_DEFAULT, encode_observation, observation_size
from engine.clock import DAY_DURATION, DAY_END_MINUTE, sim_minute_to_wall_clock
from engine.rail_network import Departure, RailNetwork
from engine.rules import count_controlled_stations, get_valid_departures
from engine.simulation import Simulation


def make_env(config: dict, rail_network: RailNetwork) -> AECEnv:
    """Factory: create a fully wrapped PettingZoo AEC environment."""
    env = RailGameEnv(config, rail_network)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class RailGameEnv(AECEnv):
    """
    PettingZoo AEC environment for the rail strategy game.

    The simulation runs internally.  When a team becomes idle it is added
    to the decision queue and control passes to the AEC framework.
    """

    metadata = {
        "render_modes": [],
        "name": "rail_game_v0",
        "is_parallelizable": True,
    }

    def __init__(self, config: dict, rail_network: RailNetwork):
        super().__init__()
        self.config   = config
        self.net      = rail_network
        self.possible_agents = ["A", "B"]
        self.render_mode     = None

        K     = config["agents"]["max_departures_k"]
        c_max = config["agents"].get("c_max", C_MAX_DEFAULT)
        obs_sz = observation_size(k=K, c_max=c_max)

        self.observation_spaces = {
            a: gym.spaces.Box(low=-1.0, high=20.0, shape=(obs_sz,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: gym.spaces.Discrete(K + 2)
            for a in self.possible_agents
        }

        self._K             = K
        self._c_max         = c_max
        self._starting_coins: int        = config["game"]["starting_coins"]
        self._starting_station_id: Optional[str] = None
        self._sim: Optional[Simulation]  = None

        self._decision_queue: List[str]            = []
        self._departures: Dict[str, List[Departure]] = {}

    # ------------------------------------------------------------------
    # PettingZoo AEC API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._init_sim()

        self.agents              = list(self.possible_agents)
        self.rewards             = {a: 0.0  for a in self.agents}
        self._cumulative_rewards = {a: 0.0  for a in self.agents}
        self.terminations        = {a: False for a in self.agents}
        self.truncations         = {a: False for a in self.agents}
        self.infos               = {a: {}    for a in self.agents}
        self._prev_counts: dict      = {"A": 0, "B": 0}
        self._prev_counts_snap: dict = {"A": 0, "B": 0}
        self._challenge_bonus: dict  = {}
        self._revisit_penalty: dict  = {}

        self._decision_queue = []
        self._departures     = {}

        self._advance_to_decision()

        obs = {a: self.observe(a) for a in self.agents}
        return obs, self.infos

    def step(self, action: int):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._clear_rewards()

        departures = self._departures.get(agent, [])
        # Snapshot counts before applying the action so the reward delta is
        # computed relative to the state *before* this decision.
        self._prev_counts_snap = dict(self._prev_counts)
        self._apply_action(agent, action, departures)

        self.rewards[agent] = self._compute_step_reward(agent)
        # Keep _prev_counts up-to-date with current reality.
        sid = self._starting_station_id
        self._prev_counts["A"] = count_controlled_stations(self._sim.state, "A", sid)
        self._prev_counts["B"] = count_controlled_stations(self._sim.state, "B", sid)
        self._accumulate_rewards()

        self._decision_queue = [a for a in self._decision_queue if a != agent]

        if self._decision_queue:
            self.agent_selection = self._decision_queue[0]
        else:
            if not self._sim.done:
                self._sim.step()
            self._advance_to_decision()

    def observe(self, agent: str) -> np.ndarray:
        return encode_observation(
            self._sim.state,
            self.net,
            agent,
            self._departures.get(agent, []),
            self._starting_station_id,
            k=self._K,
            starting_coins=self._starting_coins,
            c_max=self._c_max,
        )

    def observation_space(self, agent: str) -> gym.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]

    def render(self):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Action masking (used by MaskablePPO at inference time)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return valid-action mask for the *current* agent, shape (K+2,)."""
        return self._compute_action_mask(self.agent_selection)

    def _compute_action_mask(self, agent: str) -> np.ndarray:
        K    = self._K
        mask = np.zeros(K + 2, dtype=bool)

        deps    = self._departures.get(agent, [])
        n_valid = min(len(deps), K)
        mask[:n_valid] = True

        # Challenge only if agent is physically at a challenge station
        if self._sim is not None:
            team = self._sim.state.teams[agent]
            has_challenge = any(
                c.station_id == team.current_station
                for c in self._sim.state.challenges
            )
            mask[K] = has_challenge

        mask[K + 1] = True   # wait is always valid
        return mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sim(self):
        def noop(state, net, tid, deps):
            return self._K + 1   # wait — AEC env drives decisions

        self._sim = Simulation(self.config, self.net, noop, noop)
        self._sim.skip_agent_queries  = True
        self._starting_station_id     = self._sim._starting_station_id

    def _idle_teams(self) -> List[str]:
        result = []
        for tid in ("A", "B"):
            if tid not in self.agents:
                continue
            if (tid not in self._sim._transit
                    and tid not in self._sim._challenge_attempts):
                result.append(tid)
        return result

    def _advance_to_decision(self):
        max_steps = DAY_DURATION * self.config["game"]["num_days"] + 100
        for _ in range(max_steps):
            if self._sim.done:
                self._end_episode()
                return
            idle = self._idle_teams()
            if idle:
                self._decision_queue = idle
                for tid in idle:
                    self._departures[tid] = self._sim.get_available_departures(tid)
                self.agent_selection = idle[0]
                return
            self._sim.step()
        self._end_episode()

    def _apply_action(self, team_id: str, action: int, departures: List[Departure]):
        K    = self._K
        wall = sim_minute_to_wall_clock(self._sim.state.sim_minute)
        if action == K:
            team = self._sim.state.teams[team_id]
            ch = next(
                (c for c in self._sim.state.challenges
                 if c.station_id == team.current_station),
                None,
            )
            if ch is not None:
                self._sim._attempt_challenge(team_id, wall)
                # Bonus for initiating a challenge — bridges the 30-min gap
                # before the station-control delta reward fires.
                self._challenge_bonus[team_id] = (
                    self._challenge_bonus.get(team_id, 0.0) + 0.05
                )
        elif action == K + 1:
            pass   # explicit wait
        elif 0 <= action < len(departures):
            dest_id = departures[action].destination_stop_id
            dest_st = self._sim.state.stations.get(dest_id)
            if dest_st and dest_st.controlling_team() == team_id:
                self._revisit_penalty[team_id] = (
                    self._revisit_penalty.get(team_id, 0.0) - 0.02
                )
            self._sim._board_train(team_id, departures[action], wall)
        # else: out-of-range index → silent wait

    def _compute_step_reward(self, agent: str) -> float:
        state = self._sim.state
        opp   = "B" if agent == "A" else "A"
        sid   = self._starting_station_id
        our_now = count_controlled_stations(state, agent, sid)
        opp_now = count_controlled_stations(state, opp,   sid)
        # Read deltas using the snapshot taken before any agent's reward was
        # computed this step (set by the caller via _snapshot_prev_counts).
        delta_our = our_now - self._prev_counts_snap.get(agent, 0)
        delta_opp = opp_now - self._prev_counts_snap.get(opp,   0)
        # Accumulate completed-challenge bonus if flagged by _apply_action.
        ch_bonus     = self._challenge_bonus.pop(agent, 0.0)
        revisit_pen  = self._revisit_penalty.pop(agent, 0.0)
        return 0.1 * delta_our - 0.1 * delta_opp + ch_bonus + revisit_pen

    def _end_episode(self):
        state = self._sim.state
        sid   = self._starting_station_id
        total = max(sum(1 for s in state.stations if s != sid), 1)
        for agent in list(self.agents):
            opp     = "B" if agent == "A" else "A"
            our     = count_controlled_stations(state, agent, sid)
            opp_ctrl = count_controlled_stations(state, opp,  sid)
            # Scale: ±1.0 for a clean sweep, ±0 for a draw.
            terminal_r = 1.0 * (our - opp_ctrl) / total
            self.rewards[agent]              = terminal_r
            self._cumulative_rewards[agent] += terminal_r
            self.terminations[agent]         = True
        self._accumulate_rewards()
        if self.agents:
            self.agent_selection = self.agents[0]


# ──────────────────────────────────────────────────────────────────────────────
# Parallel environment (used for PPO training via supersuit)
# ──────────────────────────────────────────────────────────────────────────────

def make_parallel_env(config: dict, rail_network: RailNetwork) -> ParallelEnv:
    """Factory: create a ParallelEnv for PPO training."""
    return RailGameParallelEnv(config, rail_network)


class RailGameParallelEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for PPO training.

    step() receives actions for all living agents.  Only agents in
    _decision_queue (idle this tick) have their actions applied;
    in-transit agents are auto-waited.  The sim then advances until at
    least one agent is idle again.
    """

    metadata = {
        "render_modes": [],
        "name": "rail_game_v0",
        "is_parallelizable": True,
    }

    def __init__(self, config: dict, rail_network: RailNetwork):
        super().__init__()
        self.config          = config
        self.net             = rail_network
        self.possible_agents = ["A", "B"]
        self.render_mode     = None

        K     = config["agents"]["max_departures_k"]
        c_max = config["agents"].get("c_max", C_MAX_DEFAULT)
        obs_sz = observation_size(k=K, c_max=c_max)

        self.observation_spaces = {
            a: gym.spaces.Box(low=-1.0, high=20.0, shape=(obs_sz,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: gym.spaces.Discrete(K + 2)
            for a in self.possible_agents
        }

        self._K             = K
        self._c_max         = c_max
        self._starting_coins: int        = config["game"]["starting_coins"]
        self._starting_station_id: Optional[str] = None
        self._sim: Optional[Simulation]  = None
        self._decision_queue: List[str]            = []
        self._departures: Dict[str, List[Departure]] = {}

    # ------------------------------------------------------------------
    # PettingZoo ParallelEnv API
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> gym.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._init_sim()
        self.agents              = list(self.possible_agents)
        self._decision_queue     = []
        self._departures         = {}
        self._prev_counts: dict      = {"A": 0, "B": 0}
        self._prev_counts_snap: dict = {"A": 0, "B": 0}
        self._challenge_bonus: dict  = {}
        self._revisit_penalty: dict  = {}

        self._advance_to_decision()

        obs   = {a: self._observe(a) for a in self.possible_agents}
        infos = {a: {}               for a in self.possible_agents}
        return obs, infos

    def step(self, actions):
        # Snapshot counts before applying any actions so every agent's reward
        # delta is computed against the same pre-step baseline (fixes race).
        sid = self._starting_station_id
        self._prev_counts_snap = {
            a: count_controlled_stations(self._sim.state, a, sid)
            for a in self.possible_agents
        }
        for agent in self._decision_queue:
            action = actions.get(agent, self._K + 1)
            self._apply_action(agent, action, self._departures.get(agent, []))

        rewards = {a: self._compute_step_reward(a) for a in self.possible_agents}
        # Refresh prev_counts after rewards are computed.
        for a in self.possible_agents:
            self._prev_counts[a] = count_controlled_stations(self._sim.state, a, sid)

        self._decision_queue = []
        self._departures     = {}
        if not self._sim.done:
            self._sim.step()
        self._advance_to_decision()

        terminations = {a: False for a in self.possible_agents}
        truncations  = {a: False for a in self.possible_agents}
        infos        = {a: {}    for a in self.possible_agents}

        if self._sim.done:
            self._add_terminal_rewards(rewards, terminations)
            self.agents = []

        obs = {a: self._observe(a) for a in self.possible_agents}
        return obs, rewards, terminations, truncations, infos

    def render(self):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Action masking (used by ActionMaskVecEnvWrapper during training)
    # ------------------------------------------------------------------

    def action_masks(self) -> Dict[str, np.ndarray]:
        """Return {agent: bool_array} of shape (K+2,) for every agent."""
        return {a: self._compute_action_mask(a) for a in self.possible_agents}

    def _compute_action_mask(self, agent: str) -> np.ndarray:
        K    = self._K
        mask = np.zeros(K + 2, dtype=bool)

        deps    = self._departures.get(agent, [])
        n_valid = min(len(deps), K)
        mask[:n_valid] = True

        if self._sim is not None:
            team = self._sim.state.teams[agent]
            has_challenge = any(
                c.station_id == team.current_station
                for c in self._sim.state.challenges
            )
            mask[K] = has_challenge

        mask[K + 1] = True   # wait always valid
        return mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sim(self):
        def noop(state, net, tid, deps):
            return self._K + 1

        self._sim = Simulation(self.config, self.net, noop, noop)
        self._sim.skip_agent_queries  = True
        self._starting_station_id     = self._sim._starting_station_id

    def _idle_teams(self) -> List[str]:
        result = []
        for tid in ("A", "B"):
            if (tid not in self._sim._transit
                    and tid not in self._sim._challenge_attempts):
                result.append(tid)
        return result

    def _advance_to_decision(self):
        max_steps = DAY_DURATION * self.config["game"]["num_days"] + 100
        for _ in range(max_steps):
            if self._sim.done:
                return
            idle = self._idle_teams()
            if idle:
                self._decision_queue = idle
                for tid in idle:
                    self._departures[tid] = self._sim.get_available_departures(tid)
                return
            self._sim.step()

    def _apply_action(self, team_id: str, action: int, departures: List[Departure]):
        K    = self._K
        wall = sim_minute_to_wall_clock(self._sim.state.sim_minute)
        if action == K:
            team = self._sim.state.teams[team_id]
            ch = next(
                (c for c in self._sim.state.challenges
                 if c.station_id == team.current_station),
                None,
            )
            if ch is not None:
                self._sim._attempt_challenge(team_id, wall)
                self._challenge_bonus[team_id] = (
                    self._challenge_bonus.get(team_id, 0.0) + 0.05
                )
        elif action == K + 1:
            pass
        elif 0 <= action < len(departures):
            dest_id = departures[action].destination_stop_id
            dest_st = self._sim.state.stations.get(dest_id)
            if dest_st and dest_st.controlling_team() == team_id:
                self._revisit_penalty[team_id] = (
                    self._revisit_penalty.get(team_id, 0.0) - 0.02
                )
            self._sim._board_train(team_id, departures[action], wall)

    def _compute_step_reward(self, agent: str) -> float:
        state   = self._sim.state
        opp     = "B" if agent == "A" else "A"
        sid     = self._starting_station_id
        our_now = count_controlled_stations(state, agent, sid)
        opp_now = count_controlled_stations(state, opp,   sid)
        # Use the pre-step snapshot so A and B see the same baseline.
        delta_our = our_now - self._prev_counts_snap.get(agent, 0)
        delta_opp = opp_now - self._prev_counts_snap.get(opp,   0)
        ch_bonus    = self._challenge_bonus.pop(agent, 0.0)
        revisit_pen = self._revisit_penalty.pop(agent, 0.0)
        return 0.1 * delta_our - 0.1 * delta_opp + ch_bonus + revisit_pen

    def _add_terminal_rewards(self, rewards: dict, terminations: dict):
        state = self._sim.state
        sid   = self._starting_station_id
        total = max(sum(1 for s in state.stations if s != sid), 1)
        for agent in self.possible_agents:
            opp      = "B" if agent == "A" else "A"
            our      = count_controlled_stations(state, agent, sid)
            opp_ctrl = count_controlled_stations(state, opp,   sid)
            # Scale: ±1.0 for a clean sweep, ±0 for a draw.
            rewards[agent] = rewards.get(agent, 0.0) + 1.0 * (our - opp_ctrl) / total
            terminations[agent] = True

    def _observe(self, agent: str) -> np.ndarray:
        return encode_observation(
            self._sim.state,
            self.net,
            agent,
            self._departures.get(agent, []),
            self._starting_station_id,
            k=self._K,
            starting_coins=self._starting_coins,
            c_max=self._c_max,
        )
