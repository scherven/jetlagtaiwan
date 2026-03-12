"""
PPO self-play training for the rail strategy game.

Usage:
    python -m agents.ppo                     # train with config defaults
    python -m agents.ppo --timesteps 1000000 # override total timesteps

Training loop:
  1. Both agents share one MaskablePPO policy (self-play via supersuit).
  2. Every `eval_interval` episodes, evaluate win rate vs the heuristic agent.
  3. Log win rate to training_log.csv.
  4. Save a policy snapshot whenever a new win-rate high is reached.
  5. Stop when win rate exceeds win_rate_target over eval_episodes window.

Action masking:
  - ActionMaskVecEnvWrapper wraps the supersuit VecEnv and exposes
    action_masks() / env_method("action_masks") for MaskablePPO.
  - The mask is computed from RailGameParallelEnv.action_masks():
      · departure slots 0..len(departures)-1 → True
      · slot K (challenge)                   → True only if at challenge station
      · slot K+1 (wait)                      → always True
  - num_cpus=0 keeps everything in-process so the wrapper can traverse the
    gymnasium SyncVectorEnv to reach the live RailGameParallelEnv.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


def _check_imports():
    missing = []
    for pkg in ("stable_baselines3", "sb3_contrib", "pettingzoo", "supersuit", "gymnasium"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Run: pip install stable-baselines3 sb3-contrib pettingzoo supersuit gymnasium")
        sys.exit(1)


from .env_wrapper import make_parallel_env, RailGameEnv


# ──────────────────────────────────────────────────────────────────────────────
# ActionMaskVecEnvWrapper
# ──────────────────────────────────────────────────────────────────────────────

def _find_par_env(venv):
    """
    Traverse the supersuit + SB3 wrapper chain to reach the live
    RailGameParallelEnv instance.

    With num_cpus=0 the chain is (outer → inner):
        VecMonitor
          → ActionMaskVecEnvWrapper          (this wrapper itself)
            → SB3VecEnvWrapper    (.venv)
              → ConcatVecEnv      (.vec_envs[0])
                → MarkovVectorEnv (.par_env)
                  → RailGameParallelEnv

    Requires num_cpus=0 in concat_vec_envs_v1 so that ConcatVecEnv holds
    real MarkovVectorEnv instances rather than subprocess handles.
    """
    from agents.env_wrapper import RailGameParallelEnv

    env = venv
    # Unwrap SB3 VecEnvWrapper layers (each exposes .venv)
    while hasattr(env, "venv"):
        env = env.venv
    # env is now supersuit's ConcatVecEnv which holds .vec_envs
    if hasattr(env, "vec_envs") and env.vec_envs:
        env = env.vec_envs[0]
    # env is now MarkovVectorEnv — grab its .par_env
    if hasattr(env, "par_env") and isinstance(env.par_env, RailGameParallelEnv):
        return env.par_env
    # Fallback: gymnasium SyncVectorEnv stores instances in .envs
    if hasattr(env, "envs") and env.envs:
        candidate = env.envs[0]
        if hasattr(candidate, "par_env") and isinstance(candidate.par_env, RailGameParallelEnv):
            return candidate.par_env
    raise RuntimeError(
        f"Cannot locate RailGameParallelEnv under {type(venv).__name__}. "
        "Ensure num_cpus=0 in concat_vec_envs_v1."
    )


class ActionMaskVecEnvWrapper:
    """
    Thin wrapper around an SB3 VecEnv (produced by supersuit) that exposes
    action_masks() and overrides env_method("action_masks") so that
    sb3_contrib.MaskablePPO can retrieve per-agent masks during rollout.

    Delegates everything else transparently to the wrapped VecEnv.

    Why a manual wrapper instead of VecEnvWrapper?
    VecEnvWrapper.__init__ performs deep introspection that breaks when the
    inner venv is a gymnasium VectorEnv (not a pure SB3 VecEnv); a plain
    delegation wrapper is safer here.
    """

    def __init__(self, venv):
        self._venv     = venv
        self._par_env  = _find_par_env(venv)
        # Mirror SB3 VecEnv interface attributes
        self.num_envs         = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space      = venv.action_space
        self.render_mode       = getattr(venv, "render_mode", None)
        self.reset_infos       = getattr(venv, "reset_infos", [])

    # ── Action masking ────────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """Return per-agent masks stacked to shape (n_envs, n_actions).

        MaskablePPO can call this directly (it also tries env_method below).
        """
        agent_masks = self._par_env.action_masks()   # {agent: (K+2,) bool}
        agents = self._par_env.possible_agents
        return np.stack([agent_masks[a] for a in agents])  # (n_agents, K+2)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Intercept 'action_masks' so get_action_masks() in sb3_contrib works.

        get_action_masks(env) does:
            np.stack(env.env_method("action_masks"))
        and expects a list of n_envs arrays each of shape (n_actions,).
        """
        if method_name == "action_masks":
            masks = self.action_masks()               # (n_envs, n_actions)
            return [masks[i] for i in range(len(masks))]
        return self._venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    # ── SB3 VecEnv pass-throughs ──────────────────────────────────────────────

    def reset(self):
        return self._venv.reset()

    def step_async(self, actions):
        return self._venv.step_async(actions)

    def step_wait(self):
        return self._venv.step_wait()

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        return self._venv.close()

    def render(self, mode=None):
        return self._venv.render()

    def seed(self, seed=None):
        return self._venv.seed(seed)

    def get_attr(self, attr_name, indices=None):
        return self._venv.get_attr(attr_name, indices=indices)

    def set_attr(self, attr_name, value, indices=None):
        return self._venv.set_attr(attr_name, value, indices=indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self._venv.env_is_wrapped(wrapper_class, indices=indices)

    def has_attr(self, attr_name: str) -> bool:
        """Required by sb3_contrib.is_masking_supported() and SB3 VecEnvWrapper chain."""
        if attr_name == "action_masks":
            return True
        try:
            return self._venv.has_attr(attr_name)
        except AttributeError:
            return hasattr(self._venv, attr_name)

    # Allow VecMonitor (and other SB3 wrappers) to find attributes on the
    # inner venv transparently.
    def __getattr__(self, name):
        return getattr(self._venv, name)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(config: dict, rail_network, extra_timesteps: Optional[int] = None):
    _check_imports()

    import supersuit as ss
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import VecMonitor

    tcfg             = config["training"]
    total_timesteps  = extra_timesteps or tcfg["total_timesteps"]
    eval_interval    = tcfg["eval_interval"]
    eval_episodes    = tcfg["eval_episodes"]
    win_rate_target  = tcfg["win_rate_target"]
    opponent_swap_interval = tcfg["opponent_swap_interval"]
    log_file         = tcfg["log_file"]
    snapshot_dir     = Path(tcfg["snapshot_dir"])
    snapshot_dir.mkdir(exist_ok=True)

    print("=== Rail Game — MaskablePPO Self-Play Training ===")
    print(f"Network  : {config['network']['feeds']}")
    print(f"Stations : {len(rail_network.stations)}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Eval every {eval_interval} episodes vs heuristic")
    print(f"Target win rate: {win_rate_target:.0%}\n")

    # ── Build vectorised self-play environment ────────────────────────────────
    # num_cpus=0 → gymnasium SyncVectorEnv (same-process) so that
    # ActionMaskVecEnvWrapper can traverse wrappers to the live par_env.
    def make_raw():
        return make_parallel_env(config, rail_network)

    env = make_raw()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class="stable_baselines3")
    env = ActionMaskVecEnvWrapper(env)
    env = VecMonitor(env)

    # ── MaskablePPO model ─────────────────────────────────────────────────────
    policy_kwargs = dict(net_arch=[512, 512, 256])

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=tcfg["learning_rate"],
        n_steps=tcfg["n_steps"],
        batch_size=tcfg["batch_size"],
        n_epochs=tcfg["n_epochs"],
        gamma=tcfg["gamma"],
        clip_range=tcfg["clip_range"],
        verbose=0,
        policy_kwargs=policy_kwargs,
    )

    # ── CSV logger ────────────────────────────────────────────────────────────
    log_path    = Path(log_file)
    csv_existed = log_path.exists()
    log_fp      = open(log_path, "a", newline="")
    writer      = csv.writer(log_fp)
    if not csv_existed:
        writer.writerow(["timestep", "episodes", "win_rate", "best_win_rate", "elapsed_s"])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_win_rate  = 0.0
    episodes_done  = 0
    steps_per_block = max(eval_interval * 20, tcfg["n_steps"])
    t0             = time.time()

    print("Starting training...\n")

    while model.num_timesteps < total_timesteps:
        model.learn(
            total_timesteps=steps_per_block,
            reset_num_timesteps=False,
            progress_bar=False,
            use_masking=True,        # ← enable action masking during rollout
        )

        episodes_done += eval_interval   # approximate

        # Periodic opponent snapshot
        if (episodes_done // opponent_swap_interval) > (
            (episodes_done - eval_interval) // opponent_swap_interval
        ):
            snap_path = snapshot_dir / f"snapshot_ep{episodes_done}.zip"
            model.save(str(snap_path))

        win_rate = evaluate_vs_heuristic(model, config, rail_network,
                                         n_episodes=eval_episodes)
        elapsed  = time.time() - t0

        print(
            f"  timestep={model.num_timesteps:>8,}  episodes~{episodes_done:>5}  "
            f"win_rate={win_rate:.2%}  best={best_win_rate:.2%}  "
            f"elapsed={elapsed:.0f}s"
        )

        writer.writerow([model.num_timesteps, episodes_done,
                         f"{win_rate:.4f}", f"{best_win_rate:.4f}", f"{elapsed:.1f}"])
        log_fp.flush()

        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_path     = snapshot_dir / "best_model.zip"
            model.save(str(best_path))
            print(f"  ** New best win rate {best_win_rate:.2%} — saved to {best_path}")

        if win_rate >= win_rate_target:
            print(f"\nTarget win rate {win_rate_target:.0%} reached! Stopping.")
            break

    log_fp.close()
    final_path = snapshot_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Best win rate: {best_win_rate:.2%}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation vs heuristic
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_vs_heuristic(
    model,
    config: dict,
    rail_network,
    n_episodes: int = 100,
) -> float:
    """Run n_episodes with RL as Team A and heuristic as Team B.
    Returns Team-A win fraction.  Uses a worker pool for speed.
    """
    import multiprocessing as mp

    buf = io.BytesIO()
    model.save(buf)
    model_bytes = buf.getvalue()

    n_workers = min(n_episodes, max(1, os.cpu_count() - 1))
    args      = [(model_bytes, config, i) for i in range(n_episodes)]

    feeds = config["network"]["feeds"]
    merge = config["network"].get("merge_strategy", "parent")

    with mp.Pool(
        processes=n_workers,
        initializer=_eval_worker_init,
        initargs=(feeds, merge),
    ) as pool:
        results = pool.map(_eval_worker_run, args)

    wins = sum(1 for r in results if r == "A")
    return wins / n_episodes


# ── Worker-process state ──────────────────────────────────────────────────────

_worker_rail_network = None


def _eval_worker_init(feeds, merge_strategy):
    """Load the rail network once per worker process."""
    global _worker_rail_network
    from engine.rail_network import RailNetwork
    _worker_rail_network = RailNetwork(feeds, merge_strategy=merge_strategy)


def _eval_worker_run(args) -> str:
    """Run one eval episode.  Returns 'A', 'B', or 'tie'."""
    from sb3_contrib import MaskablePPO
    from agents.heuristic import HeuristicAgent

    model_bytes, config, _seed = args
    model     = MaskablePPO.load(io.BytesIO(model_bytes))
    heuristic = HeuristicAgent(config)
    return _run_one_episode(model, heuristic, config, _worker_rail_network)


def _run_one_episode(model, heuristic, config: dict, rail_network) -> str:
    """
    Run one game: RL model as A, heuristic as B.
    Returns 'A', 'B', or 'tie'.

    Action masks are computed from RailGameEnv.action_masks() and passed to
    model.predict() so the RL agent never samples invalid actions.
    """
    from engine.rules import count_controlled_stations

    env = RailGameEnv(config, rail_network)
    obs_dict, _ = env.reset()
    heuristic.starting_station_id = env._starting_station_id

    done = False
    while not done:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            done = (all(env.terminations.values())
                    or all(env.truncations.values()))
            continue

        if agent == "A":
            obs   = obs_dict.get(agent, env.observe(agent))
            masks = env.action_masks()                        # (K+2,) bool
            action, _ = model.predict(
                obs.reshape(1, -1),
                action_masks=masks.reshape(1, -1),
                deterministic=True,
            )
            action = int(action[0]) if hasattr(action, "__len__") else int(action)
        else:
            departures = env._departures.get(agent, [])
            action     = heuristic.choose_action(
                env._sim.state, rail_network, agent, departures
            )

        env.step(action)
        obs_dict[agent] = env.observe(agent)

        done = (all(env.terminations.values())
                or all(env.truncations.values()))

    state = env._sim.state
    sid   = env._starting_station_id
    a     = count_controlled_stations(state, "A", sid)
    b     = count_controlled_stations(state, "B", sid)
    if a > b:
        return "A"
    elif b > a:
        return "B"
    return "tie"


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train RL agents for Rail Strategy Game")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps from config")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    from engine.rail_network import RailNetwork
    print("Loading rail network...")
    net = RailNetwork(
        config["network"]["feeds"],
        config["network"].get("merge_strategy", "parent"),
    )
    print(f"  {len(net.stations)} stations loaded\n")

    train(config, net, extra_timesteps=args.timesteps)


if __name__ == "__main__":
    import yaml
    main()
