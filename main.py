"""
Entry point for the Rail Strategy Game — Phase 1.

Usage:
    python main.py                                 # heuristic vs heuristic
    python main.py --model-a checkpoints/best_model.zip          # RL agent as A
    python main.py --model-a X.zip --model-b Y.zip               # RL vs RL
    python main.py --headless                      # no display (fast)
    python main.py --speed 100                     # faster UI

Agent flags:
    --model-a PATH   Load a trained SB3 model as Team A (default: heuristic)
    --model-b PATH   Load a trained SB3 model as Team B (default: heuristic)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import pygame
import yaml
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from agents.eval import encode_observation, C_MAX_DEFAULT
from agents.heuristic import HeuristicAgent
from engine.rail_network import RailNetwork
from engine.simulation import Simulation
from ui.display import Display

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

def _load_rl_agent(model_path: str, team_label: str, config: dict):
    """Load a MaskablePPO (or legacy PPO) model and return an agent callback."""
    k              = config["agents"]["max_departures_k"]
    c_max          = config["agents"].get("c_max", C_MAX_DEFAULT)
    starting_coins = config["game"]["starting_coins"]
    max_chips      = config["game"].get("max_chips_per_station", 5)

    # Try MaskablePPO first; fall back to plain PPO for old checkpoints.
    maskable = True
    try:
        model = MaskablePPO.load(model_path)
    except Exception as e1:
        try:
            model = PPO.load(model_path)
            maskable = False
            print(f"  NOTE: {model_path} is a plain-PPO checkpoint (no action masking at inference).")
        except Exception as e2:
            print(f"  WARNING: could not load model for Team {team_label}: {e2}")
            return None

    print(f"  Loaded {'MaskablePPO' if maskable else 'PPO'} model for Team {team_label}: {model_path} (k={k}, c_max={c_max})")

    def agent_fn(state, rail_network, team_id, departures):
        obs = encode_observation(
            state, rail_network, team_id, departures,
            _STARTING_ID, k=k, starting_coins=starting_coins,
            max_chips_per_station=max_chips, c_max=c_max,
        )
        if maskable:
            mask = np.zeros(k + 2, dtype=bool)
            mask[:min(len(departures), k)] = True
            team = state.teams[team_id]
            mask[k]     = any(c.station_id == team.current_station
                              for c in state.challenges)
            mask[k + 1] = True
            action, _ = model.predict(
                obs.reshape(1, -1),
                action_masks=mask.reshape(1, -1),
                deterministic=True,
            )
        else:
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        return int(action[0]) if hasattr(action, "__len__") else int(action)

    return agent_fn

_STARTING_ID = None   # set after network load so the RL closure can reference it

def main():
    parser = argparse.ArgumentParser(description="Rail Strategy Game")
    parser.add_argument("--headless", action="store_true", help="Run without UI")
    parser.add_argument("--speed", type=float, default=20.0, help="Simulation speed multiplier")
    parser.add_argument("--model-a", type=str, default=None, help="Path to trained SB3 model for Team A")
    parser.add_argument("--model-b", type=str, default=None, help="Path to trained SB3 model for Team B")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    # Build rail network
    # ------------------------------------------------------------------ #
    print("Loading rail network...")
    feed_dirs = config["network"]["feeds"]
    merge_strategy = config["network"].get("merge_strategy", "parent")
    net = RailNetwork(feed_dirs, merge_strategy=merge_strategy)
    print(f"  {len(net.stations)} stations, "
          f"{sum(len(v) for v in net.schedules.values())} schedule entries")

    # Find starting station
    start_name = config["game"]["starting_station"]
    start_node = net.station_by_name(start_name)
    if start_node is None:
        print(f"ERROR: Starting station {start_name!r} not found.")
        sys.exit(1)
    starting_station_id = start_node.id
    print(f"  Starting station: {start_node.name!r} ({start_node.id})")

    # ------------------------------------------------------------------ #
    # Build agents
    # ------------------------------------------------------------------ #
    global _STARTING_ID
    _STARTING_ID = starting_station_id

    heuristic_a = HeuristicAgent(config)
    heuristic_a.starting_station_id = starting_station_id
    heuristic_b = HeuristicAgent(config)
    heuristic_b.starting_station_id = starting_station_id

    rl_a = _load_rl_agent(args.model_a, "A", config) if args.model_a else None
    rl_b = _load_rl_agent(args.model_b, "B", config) if args.model_b else None

    def agent_a(state, rail_network, team_id, departures):
        if rl_a:
            return rl_a(state, rail_network, team_id, departures)
        return heuristic_a.choose_action(state, rail_network, team_id, departures)

    def agent_b(state, rail_network, team_id, departures):
        if rl_b:
            return rl_b(state, rail_network, team_id, departures)
        return heuristic_b.choose_action(state, rail_network, team_id, departures)

    a_label = f"RL ({args.model_a})" if rl_a else "Heuristic"
    b_label = f"RL ({args.model_b})" if rl_b else "Heuristic"
    print(f"  Team A: {a_label}")
    print(f"  Team B: {b_label}")

    # ------------------------------------------------------------------ #
    # Build simulation
    # ------------------------------------------------------------------ #
    sim = Simulation(config, net, agent_a, agent_b)

    # ------------------------------------------------------------------ #
    # Headless mode
    # ------------------------------------------------------------------ #
    if args.headless:
        print("\nRunning headless simulation...")
        t0 = time.time()
        for events in sim.run():
            for e in events:
                if e.startswith("==="):
                    print(e)
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.2f}s")
        return

    # ------------------------------------------------------------------ #
    # Pygame UI
    # ------------------------------------------------------------------ #
    display = Display(config, net, starting_station_id)

    # Set starting speed
    speed = args.speed
    speed_options = config["simulation"]["speed_options"]
    display.speed_idx = min(
        range(len(speed_options)),
        key=lambda i: abs(speed_options[i] - speed)
    )

    clock = pygame.time.Clock()
    TARGET_FPS = 60
    sim_minute_accumulator = 0.0   # fractional sim minutes owed

    running = True
    while running and not sim.done:
        dt_ms = clock.tick(TARGET_FPS)    # milliseconds since last frame
        dt_s  = dt_ms / 1000.0

        # Handle input
        action = display.handle_events()
        if action == "quit":
            running = False
        elif action in ("pause", "play"):
            sim.state.is_paused = not sim.state.is_paused
        elif action == "speed_up":
            display.speed_idx = min(display.speed_idx + 1, len(speed_options) - 1)
        elif action == "speed_down":
            display.speed_idx = max(display.speed_idx - 1, 0)

        # Advance simulation
        if not sim.state.is_paused:
            current_speed = speed_options[display.speed_idx]
            # At 1x, 1 sim-minute = 60 real seconds. At Nx, 1 sim-minute = 60/N real seconds.
            sim_minutes_per_real_second = current_speed / 60.0
            sim_minute_accumulator += dt_s * sim_minutes_per_real_second

            while sim_minute_accumulator >= 1.0 and not sim.done:
                sim.step()
                sim_minute_accumulator -= 1.0

        # Render
        display.draw(sim.state, sim.log[-40:], starting_station_id)

    # Final frame after game ends
    if sim.done:
        display.draw(sim.state, sim.log[-40:], starting_station_id)
        # Show for 5 seconds
        t_end = time.time() + 5
        while time.time() < t_end:
            if display.handle_events() == "quit":
                break
            pygame.time.wait(100)

    pygame.quit()


if __name__ == "__main__":
    main()
