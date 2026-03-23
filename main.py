"""
Entry point for the Rail Strategy Game — Phase 1.

Usage:
    python main.py                                    # heuristic vs heuristic
    python main.py --agent-a minimax                  # minimax as A
    python main.py --agent-a minimax --minimax-depth 3
    python main.py --agent-a rl:checkpoints/best.zip  # RL model as A
    python main.py --agent-a rl:X.zip --agent-b rl:Y.zip
    python main.py --config config_grid.yaml --headless
    python main.py --speed 100

Agent flags (--agent-a / --agent-b):
    heuristic          Greedy rule-based agent (default)
    minimax            Minimax agent (depth from --minimax-depth)
    rl:PATH            Load a trained SB3/MaskablePPO model from PATH

Minimax tuning:
    --minimax-depth N  Search depth (default: 2; depth 1 = one-step lookahead)
    --minimax-branch N Max departures considered per node (default: 5)

Backwards compatibility:
    --model-a PATH     Equivalent to --agent-a rl:PATH
    --model-b PATH     Equivalent to --agent-b rl:PATH
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import pygame
import yaml

from agents.eval import encode_observation, C_MAX_DEFAULT
from agents.heuristic import HeuristicAgent
from agents.minimax import MinimaxAgent
from engine.rail_network import RailNetwork
from engine.simulation import Simulation
from ui.display import Display

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

def _make_rl_agent(model_path: str, team_label: str, config: dict):
    """Load a trained MaskablePPO model and return an agent callback."""
    try:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(model_path)
        k              = config["agents"]["max_departures_k"]
        c_max          = config["agents"].get("c_max", C_MAX_DEFAULT)
        starting_coins = config["game"]["starting_coins"]
        max_chips      = config["game"].get("max_chips_per_station", 5)
        print(f"  Loaded RL model for Team {team_label}: {model_path}")

        def agent_fn(state, rail_network, team_id, departures):
            obs = encode_observation(
                state, rail_network, team_id, departures,
                _STARTING_ID, k=k, starting_coins=starting_coins,
                max_chips_per_station=max_chips, c_max=c_max,
            )
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
            return int(action[0]) if hasattr(action, "__len__") else int(action)

        return agent_fn
    except Exception as e:
        print(f"  WARNING: could not load RL model for Team {team_label}: {e}")
        return None


def _build_agent(spec: str, label: str, config: dict, starting_station_id: str,
                 minimax_depth: int, minimax_branch: int):
    """
    Parse an agent spec string and return (agent_fn, display_label).

    Spec formats:
        "heuristic"      — HeuristicAgent
        "minimax"        — MinimaxAgent (uses minimax_depth / minimax_branch)
        "rl:PATH"        — MaskablePPO loaded from PATH
        "PATH"           — treated as rl:PATH for backwards-compat (--model-a)
    """
    spec = spec.strip()

    if spec == "heuristic":
        agent = HeuristicAgent(config)
        agent.starting_station_id = starting_station_id
        return agent.choose_action, "Heuristic"

    if spec == "minimax":
        agent = MinimaxAgent(config, depth=minimax_depth, branch_factor=minimax_branch)
        agent.starting_station_id = starting_station_id
        return agent.choose_action, f"Minimax(depth={minimax_depth})"

    # "rl:PATH" or bare "PATH"
    model_path = spec[3:] if spec.startswith("rl:") else spec
    fn = _make_rl_agent(model_path, label, config)
    if fn is not None:
        return fn, f"RL({model_path})"
    # Fall back to heuristic if model load failed
    agent = HeuristicAgent(config)
    agent.starting_station_id = starting_station_id
    return agent.choose_action, "Heuristic(fallback)"


_STARTING_ID = None   # set after network load so the RL closure can reference it

def main():
    parser = argparse.ArgumentParser(description="Rail Strategy Game")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config file (default: config.yaml; use config_grid.yaml for grid)")
    parser.add_argument("--headless", action="store_true", help="Run without UI")
    parser.add_argument("--speed", type=float, default=20.0, help="Simulation speed multiplier")
    parser.add_argument("--agent-a", type=str, default="heuristic",
                        help="Team A agent: heuristic | minimax | rl:PATH")
    parser.add_argument("--agent-b", type=str, default="heuristic",
                        help="Team B agent: heuristic | minimax | rl:PATH")
    parser.add_argument("--days", type=int, default=None,
                        help="Limit simulation to this many days (overrides config)")
    parser.add_argument("--minimax-depth", type=int, default=2,
                        help="Search depth for minimax agents (default: 2)")
    parser.add_argument("--minimax-branch", type=int, default=5,
                        help="Branching factor for minimax agents (default: 5)")
    # Backwards-compat shims: --model-a/b PATH ≡ --agent-a/b rl:PATH
    parser.add_argument("--model-a", type=str, default=None,
                        help="(deprecated) Path to RL model for Team A; use --agent-a rl:PATH")
    parser.add_argument("--model-b", type=str, default=None,
                        help="(deprecated) Path to RL model for Team B; use --agent-b rl:PATH")
    args = parser.parse_args()

    # Honour legacy --model-a/b flags by overriding --agent-a/b
    if args.model_a:
        args.agent_a = f"rl:{args.model_a}"
    if args.model_b:
        args.agent_b = f"rl:{args.model_b}"

    # ------------------------------------------------------------------ #
    # Load config
    # ------------------------------------------------------------------ #
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.days is not None:
        config["game"]["num_days"] = args.days

    # ------------------------------------------------------------------ #
    # Build rail network
    # ------------------------------------------------------------------ #
    print("Loading rail network...")
    net_cfg = config["network"]
    if net_cfg.get("type") == "grid":
        from engine.grid_network import GridNetwork
        net = GridNetwork(
            rows=net_cfg.get("grid_rows", 10),
            cols=net_cfg.get("grid_cols", 10),
            interval_minutes=net_cfg.get("grid_interval_minutes", 5),
            travel_time=net_cfg.get("grid_travel_time", 5),
        )
    else:
        net = RailNetwork(
            net_cfg["feeds"],
            merge_strategy=net_cfg.get("merge_strategy", "parent"),
        )
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

    agent_a, a_label = _build_agent(
        args.agent_a, "A", config, starting_station_id,
        args.minimax_depth, args.minimax_branch,
    )
    agent_b, b_label = _build_agent(
        args.agent_b, "B", config, starting_station_id,
        args.minimax_depth, args.minimax_branch,
    )
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
        from engine.rules import count_controlled_stations
        from engine.clock import sim_minute_to_str

        print("\nRunning headless simulation...")
        t0 = time.time()

        # Per-game stats
        stats = {
            "A": {"waits": 0, "boards": 0, "challenges": 0, "cant_afford": 0},
            "B": {"waits": 0, "boards": 0, "challenges": 0, "cant_afford": 0},
        }

        MOVE_KEYWORDS = ("boarded", "arrived at", "completed journey",
                         "started challenge", "cannot afford", "cancelled",
                         "challenge but")

        for events in sim.run():
            wall = sim_minute_to_str(sim.state.sim_minute)
            for e in events:
                if e.startswith("==="):
                    print(e)
                    # Print day snapshot on day boundaries
                    if "ended" in e or "GAME OVER" in e:
                        state = sim.state
                        for tid in ("A", "B"):
                            t = state.teams[tid]
                            ctrl = count_controlled_stations(state, tid, starting_station_id)
                            loc = state.stations[t.current_station].name if t.current_station in state.stations else t.current_station
                            print(f"  Team {tid}: coins={t.coins:>4d}  controlled={ctrl:>3d}  at={loc!r}")
                        print(f"  Active challenges: {len(state.challenges)}")
                elif any(kw in e for kw in MOVE_KEYWORDS):
                    print(f"  [{wall}] {e}")

                # Tally action events
                for tid in ("A", "B"):
                    if f"Team {tid} boarded" in e:
                        stats[tid]["boards"] += 1
                    elif f"Team {tid} started challenge" in e:
                        stats[tid]["challenges"] += 1
                    elif f"Team {tid} cannot afford" in e:
                        stats[tid]["cant_afford"] += 1

        elapsed = time.time() - t0

        # Final action breakdown
        print("\n--- Agent Action Summary ---")
        for tid in ("A", "B"):
            s = stats[tid]
            total_active = s["boards"] + s["challenges"] + s["cant_afford"]
            print(f"  Team {tid} ({a_label if tid == 'A' else b_label}):")
            print(f"    Boards:      {s['boards']:>4d}")
            print(f"    Challenges:  {s['challenges']:>4d}")
            print(f"    Cant afford: {s['cant_afford']:>4d}")

        # Final board state
        state = sim.state
        print("\n--- Final Board State ---")
        for tid in ("A", "B"):
            ctrl = count_controlled_stations(state, tid, starting_station_id)
            t = state.teams[tid]
            print(f"  Team {tid}: coins={t.coins}  controlled_stations={ctrl}")
        # Top controlled stations per team
        for tid in ("A", "B"):
            opp = "B" if tid == "A" else "A"
            chip_us  = f"chips_team_{tid.lower()}"
            chip_opp = f"chips_team_{opp.lower()}"
            contested = [(s.name, getattr(s, chip_us), getattr(s, chip_opp))
                         for s in state.stations.values()
                         if s.id != starting_station_id
                         and getattr(s, chip_us) > 0 and getattr(s, chip_opp) > 0]
            print(f"  Contested stations for {tid}: {len(contested)}")
            for name, uc, oc in sorted(contested, key=lambda x: -(x[1]+x[2]))[:5]:
                print(f"    {name}: us={uc} opp={oc}")

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
