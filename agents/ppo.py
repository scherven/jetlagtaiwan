"""
PPO self-play training for the rail strategy game.

Usage:
    python -m agents.ppo                     # train with config defaults
    python -m agents.ppo --timesteps 1000000 # override total timesteps

Training loop:
  1. Both agents share one PPO policy (self-play via supersuit vectorisation).
  2. Every `eval_interval` episodes, evaluate win rate vs the heuristic agent.
  3. Log win rate to training_log.csv.
  4. Save a policy snapshot whenever a new win-rate high is reached.
  5. Stop when win rate exceeds win_rate_target over eval_episodes window.

Opponent snapshots (anti-forgetting):
  Every `opponent_swap_interval` training episodes, the current policy is
  saved as a snapshot. Evaluation always uses the CURRENT policy vs heuristic.
"""

from __future__ import annotations

import argparse
import csv
import io
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import supersuit as ss
import yaml
from pettingzoo.utils.conversions import aec_to_parallel
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from .env_wrapper import RailGameEnv, make_parallel_env
from agents.heuristic import HeuristicAgent
from engine.rail_network import RailNetwork
from engine.rules import count_controlled_stations


def train(config: dict, rail_network, extra_timesteps: Optional[int] = None):
    tcfg = config["training"]
    total_timesteps = extra_timesteps or tcfg["total_timesteps"]
    eval_interval = tcfg["eval_interval"]
    eval_episodes = tcfg["eval_episodes"]
    win_rate_target = tcfg["win_rate_target"]
    opponent_swap_interval = tcfg["opponent_swap_interval"]
    log_file = tcfg["log_file"]
    snapshot_dir = Path(tcfg["snapshot_dir"])
    snapshot_dir.mkdir(exist_ok=True)

    print("=== Rail Game — PPO Self-Play Training ===")
    print(f"Network: {config['network']['feeds']}")
    print(f"Stations: {len(rail_network.stations)}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Eval every {eval_interval} episodes vs heuristic")
    print(f"Target win rate: {win_rate_target:.0%}\n")

    # ------------------------------------------------------------------ #
    # Build vectorised self-play environment
    # supersuit requires a ParallelEnv, so convert from AEC first.
    # Both agents share one policy; SB3 collects rollouts from both.
    # ------------------------------------------------------------------ #
    def make_raw():
        return make_parallel_env(config, rail_network)

    env = make_raw()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    # ------------------------------------------------------------------ #
    # PPO model
    # ------------------------------------------------------------------ #
    policy_kwargs = dict(net_arch=[256, 256])

    model = PPO(
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

    # ------------------------------------------------------------------ #
    # CSV logger
    # ------------------------------------------------------------------ #
    log_path = Path(log_file)
    csv_existed = log_path.exists()
    log_fp = open(log_path, "a", newline="")
    writer = csv.writer(log_fp)
    if not csv_existed:
        writer.writerow(["timestep", "episodes", "win_rate", "best_win_rate", "elapsed_s"])

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    best_win_rate = 0.0
    episodes_done = 0
    steps_per_block = max(eval_interval * 20, tcfg["n_steps"])  # approx steps per eval block
    t0 = time.time()

    print("Starting training...\n")

    while model.num_timesteps < total_timesteps:
        # Train for one block
        model.learn(
            total_timesteps=steps_per_block,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        episodes_done += eval_interval  # approximate

        # Save opponent snapshot periodically
        if (episodes_done // opponent_swap_interval) > ((episodes_done - eval_interval) // opponent_swap_interval):
            snap_path = snapshot_dir / f"snapshot_ep{episodes_done}.zip"
            model.save(str(snap_path))

        # Evaluate vs heuristic
        win_rate = evaluate_vs_heuristic(model, config, rail_network, n_episodes=eval_episodes)
        elapsed = time.time() - t0

        print(
            f"  timestep={model.num_timesteps:>8,}  episodes~{episodes_done:>5}  "
            f"win_rate={win_rate:.2%}  best={best_win_rate:.2%}  "
            f"elapsed={elapsed:.0f}s"
        )

        writer.writerow([model.num_timesteps, episodes_done, f"{win_rate:.4f}",
                         f"{best_win_rate:.4f}", f"{elapsed:.1f}"])
        log_fp.flush()

        # Save if new best
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_path = snapshot_dir / "best_model.zip"
            model.save(str(best_path))
            print(f"  ** New best win rate {best_win_rate:.2%} — saved to {best_path}")

        # Stopping condition
        if win_rate >= win_rate_target:
            print(f"\nTarget win rate {win_rate_target:.0%} reached! Stopping.")
            break

    log_fp.close()
    final_path = snapshot_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Best win rate: {best_win_rate:.2%}")
    return model


def evaluate_vs_heuristic(
    model,
    config: dict,
    rail_network,
    n_episodes: int = 100,
) -> float:
    """
    Run n_episodes with the RL model as Team A and the heuristic as Team B.
    Returns the fraction of episodes won by Team A (RL agent).

    Runs games in parallel across available CPU cores for speed.
    """
    # Serialize model to bytes so workers can reconstruct it without sharing
    # a single model object across processes.
    buf = io.BytesIO()
    model.save(buf)
    model_bytes = buf.getvalue()

    n_workers = min(n_episodes, max(1, os.cpu_count() - 1))

    args = [(model_bytes, config, i) for i in range(n_episodes)]

    # Use spawn-safe pool with network loaded once per worker via initializer.
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


# ------------------------------------------------------------------
# Worker-process state (loaded once per worker via initializer)
# ------------------------------------------------------------------
_worker_rail_network = None


def _eval_worker_init(feeds, merge_strategy):
    """Called once per worker process to load the rail network."""
    global _worker_rail_network
    _worker_rail_network = RailNetwork(feeds, merge_strategy=merge_strategy)


def _eval_worker_run(args) -> str:
    """Run a single eval game in a worker process. Returns 'A', 'B', or 'tie'."""
    model_bytes, config, _seed = args
    model = PPO.load(io.BytesIO(model_bytes))
    heuristic = HeuristicAgent(config)
    return _run_one_episode(model, heuristic, config, _worker_rail_network)


def _run_one_episode(model, heuristic, config: dict, rail_network) -> str:
    """
    Run one game: RL model as A, heuristic as B.
    Returns 'A', 'B', or 'tie'.
    """
    env = RailGameEnv(config, rail_network)
    obs_dict, _ = env.reset()
    # Make sure heuristic knows the starting station so cost calculations are correct.
    heuristic.starting_station_id = env._starting_station_id

    done = False
    while not done:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            done = all(env.terminations.values()) or all(env.truncations.values())
            continue

        if agent == "A":
            obs = obs_dict.get(agent, env.observe(agent))
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            action = int(action[0]) if hasattr(action, "__len__") else int(action)
        else:
            departures = env._departures.get(agent, [])
            action = heuristic.choose_action(env._sim.state, rail_network, agent, departures)

        env.step(action)
        obs_dict[agent] = env.observe(agent)

        done = all(env.terminations.values()) or all(env.truncations.values())

    # Determine winner
    state = env._sim.state
    sid = env._starting_station_id
    a = count_controlled_stations(state, "A", sid)
    b = count_controlled_stations(state, "B", sid)
    if a > b:
        return "A"
    elif b > a:
        return "B"
    return "tie"


def main():
    parser = argparse.ArgumentParser(description="Train RL agents for Rail Strategy Game")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps from config")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    print("Loading rail network...")
    net = RailNetwork(
        config["network"]["feeds"],
        config["network"].get("merge_strategy", "parent"),
    )
    print(f"  {len(net.stations)} stations loaded\n")

    # Set heuristic starting station
    start_name = config["game"]["starting_station"]
    start_node = net.station_by_name(start_name)
    if start_node is None:
        print(f"ERROR: Starting station '{start_name}' not found.")
        sys.exit(1)

    train(config, net, extra_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
