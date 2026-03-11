"""
PettingZoo AEC environment wrapper for the rail strategy game.

Both agents share the same observation/action space. The environment
advances the simulation clock between decision points and presents each
team's turn to the RL framework one at a time.

Observation space : flat float32 vector, size = N*5 + 10 + K*4
Action space      : Discrete(K+2)  — 0..K-1 board departure, K challenge, K+1 wait
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import wrappers

from agents.eval import encode_observation, observation_size
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

    The simulation runs internally. When a team becomes idle (not in
    transit, not doing a challenge) it is added to the decision queue and
    control passes to the AEC framework to supply an action.
    """

    metadata = {
        "render_modes": [],
        "name": "rail_game_v0",
        "is_parallelizable": True,
    }

    def __init__(self, config: dict, rail_network: RailNetwork):
        super().__init__()
        self.config = config
        self.net = rail_network
        self.possible_agents = ["A", "B"]
        self.render_mode = None

        N = len(rail_network.stations)
        K = config["agents"]["max_departures_k"]
        obs_sz = observation_size(N, K)

        self.observation_spaces = {
            a: gym.spaces.Box(low=-1.0, high=20.0, shape=(obs_sz,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: gym.spaces.Discrete(K + 2)
            for a in self.possible_agents
        }

        self._K = K
        self._starting_coins: int = config["game"]["starting_coins"]
        self._starting_station_id: Optional[str] = None
        self._sim: Optional[Simulation] = None

        # Per-step state
        self._decision_queue: List[str] = []
        self._departures: Dict[str, List[Departure]] = {}

    # ------------------------------------------------------------------
    # PettingZoo AEC API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._init_sim()

        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._decision_queue = []
        self._departures = {}

        self._advance_to_decision()

        obs = {a: self.observe(a) for a in self.agents}
        return obs, self.infos

    def step(self, action: int):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._clear_rewards()

        # Apply action
        departures = self._departures.get(agent, [])
        self._apply_action(agent, action, departures)

        # Dense shaping reward
        self.rewards[agent] = self._compute_step_reward(agent)
        self._accumulate_rewards()

        # Advance to next decision
        self._decision_queue = [a for a in self._decision_queue if a != agent]

        if self._decision_queue:
            # Another agent also needs to act this minute
            self.agent_selection = self._decision_queue[0]
        else:
            # Advance sim at least 1 minute, then find next decision point
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sim(self):
        def noop(state, net, tid, deps):
            return self._K + 1  # wait — AEC env drives decisions externally

        self._sim = Simulation(self.config, self.net, noop, noop)
        self._sim.skip_agent_queries = True
        self._starting_station_id = self._sim._starting_station_id

    def _idle_teams(self) -> List[str]:
        """Return teams that are idle (not in transit, not doing a challenge)."""
        result = []
        for tid in ("A", "B"):
            if tid not in self.agents:
                continue
            if tid not in self._sim._transit and tid not in self._sim._challenge_attempts:
                result.append(tid)
        return result

    def _advance_to_decision(self):
        """Run the sim forward until at least one team needs a decision or game ends."""
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

        # Timed out
        self._end_episode()

    def _apply_action(self, team_id: str, action: int, departures: List[Departure]):
        K = self._K
        wall = sim_minute_to_wall_clock(self._sim.state.sim_minute)
        if action == K:
            self._sim._attempt_challenge(team_id, wall)
        elif action == K + 1:
            pass  # wait
        elif 0 <= action < len(departures):
            self._sim._board_train(team_id, departures[action], wall)
        # else: invalid — noop per spec

    def _compute_step_reward(self, agent: str) -> float:
        state = self._sim.state
        opp = "B" if agent == "A" else "A"
        sid = self._starting_station_id
        our = count_controlled_stations(state, agent, sid)
        opp_ctrl = count_controlled_stations(state, opp, sid)
        return 0.01 * our - 0.01 * opp_ctrl

    def _end_episode(self):
        state = self._sim.state
        sid = self._starting_station_id
        total = max(sum(1 for s in state.stations if s != sid), 1)

        for agent in list(self.agents):
            opp = "B" if agent == "A" else "A"
            our = count_controlled_stations(state, agent, sid)
            opp_ctrl = count_controlled_stations(state, opp, sid)
            terminal_r = (our - opp_ctrl) / total
            self.rewards[agent] = terminal_r
            self._cumulative_rewards[agent] += terminal_r
            self.terminations[agent] = True

        self._accumulate_rewards()
        if self.agents:
            self.agent_selection = self.agents[0]


def make_parallel_env(config: dict, rail_network: RailNetwork) -> ParallelEnv:
    """Factory: create a ParallelEnv for PPO training (no aec_to_parallel needed)."""
    return RailGameParallelEnv(config, rail_network)


class RailGameParallelEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for PPO training.

    Each step() receives actions for all living agents.  Only agents in
    _decision_queue (idle this tick) actually have their actions applied;
    actions for agents currently in-transit are silently ignored (auto-wait).
    The sim then advances until at least one agent is idle again.
    """

    metadata = {
        "render_modes": [],
        "name": "rail_game_v0",
        "is_parallelizable": True,
    }

    def __init__(self, config: dict, rail_network: RailNetwork):
        super().__init__()
        self.config = config
        self.net = rail_network
        self.possible_agents = ["A", "B"]
        self.render_mode = None

        N = len(rail_network.stations)
        K = config["agents"]["max_departures_k"]
        obs_sz = observation_size(N, K)

        self.observation_spaces = {
            a: gym.spaces.Box(low=-1.0, high=20.0, shape=(obs_sz,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: gym.spaces.Discrete(K + 2)
            for a in self.possible_agents
        }

        self._K = K
        self._starting_coins: int = config["game"]["starting_coins"]
        self._starting_station_id: Optional[str] = None
        self._sim: Optional[Simulation] = None
        self._decision_queue: List[str] = []
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
        self.agents = list(self.possible_agents)
        self._decision_queue = []
        self._departures = {}

        self._advance_to_decision()

        obs = {a: self._observe(a) for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}
        return obs, infos

    def step(self, actions):
        # Apply actions for agents that are idle this tick
        for agent in self._decision_queue:
            action = actions.get(agent, self._K + 1)  # default: wait
            self._apply_action(agent, action, self._departures.get(agent, []))

        rewards = {a: self._compute_step_reward(a) for a in self.possible_agents}

        # Advance sim to next decision point
        self._decision_queue = []
        self._departures = {}
        if not self._sim.done:
            self._sim.step()
        self._advance_to_decision()

        terminations = {a: False for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_sim(self):
        def noop(state, net, tid, deps):
            return self._K + 1  # wait — env drives decisions externally

        self._sim = Simulation(self.config, self.net, noop, noop)
        self._sim.skip_agent_queries = True
        self._starting_station_id = self._sim._starting_station_id

    def _idle_teams(self) -> List[str]:
        result = []
        for tid in ("A", "B"):
            if tid not in self._sim._transit and tid not in self._sim._challenge_attempts:
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
        K = self._K
        wall = sim_minute_to_wall_clock(self._sim.state.sim_minute)
        if action == K:
            self._sim._attempt_challenge(team_id, wall)
        elif action == K + 1:
            pass  # wait
        elif 0 <= action < len(departures):
            self._sim._board_train(team_id, departures[action], wall)

    def _compute_step_reward(self, agent: str) -> float:
        state = self._sim.state
        opp = "B" if agent == "A" else "A"
        sid = self._starting_station_id
        our = count_controlled_stations(state, agent, sid)
        opp_ctrl = count_controlled_stations(state, opp, sid)
        return 0.01 * our - 0.01 * opp_ctrl

    def _add_terminal_rewards(self, rewards: dict, terminations: dict):
        state = self._sim.state
        sid = self._starting_station_id
        total = max(sum(1 for s in state.stations if s != sid), 1)
        for agent in self.possible_agents:
            opp = "B" if agent == "A" else "A"
            our = count_controlled_stations(state, agent, sid)
            opp_ctrl = count_controlled_stations(state, opp, sid)
            rewards[agent] = rewards.get(agent, 0.0) + (our - opp_ctrl) / total
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
        )
