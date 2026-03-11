"""
Entry point for the Rail Strategy Game — Phase 1.

Runs two heuristic agents against each other with a Pygame observer UI.

Usage:
    python main.py [--headless] [--speed MULTIPLIER]

Options:
    --headless      Run without display (useful for testing/profiling)
    --speed N       Starting simulation speed multiplier (default: 20)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import yaml

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

def main():
    parser = argparse.ArgumentParser(description="Rail Strategy Game")
    parser.add_argument("--headless", action="store_true", help="Run without UI")
    parser.add_argument("--speed", type=float, default=20.0, help="Simulation speed multiplier")
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
    from engine.rail_network import RailNetwork
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
    from agents.heuristic import HeuristicAgent

    agent_a_obj = HeuristicAgent(config)
    agent_a_obj.starting_station_id = starting_station_id
    agent_b_obj = HeuristicAgent(config)
    agent_b_obj.starting_station_id = starting_station_id

    def agent_a(state, rail_network, team_id, departures):
        return agent_a_obj.choose_action(state, rail_network, team_id, departures)

    def agent_b(state, rail_network, team_id, departures):
        return agent_b_obj.choose_action(state, rail_network, team_id, departures)

    # ------------------------------------------------------------------ #
    # Build simulation
    # ------------------------------------------------------------------ #
    from engine.simulation import Simulation
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
    import pygame
    from ui.display import Display

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
