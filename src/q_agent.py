"""Simple tabular Q-learning agent for GridWorldEnv."""

import sys
import os
import argparse
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import GridWorldEnv

N_ACTIONS = 5
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
N_EPISODES = 5000
LOG_EVERY = 500

ACTION_NAMES = ["right", "up", "left", "down", "challenge"]


def obs_to_state(obs):
    # Coins bucketed into 0-10 range to keep state space finite.
    # Starting at 100 and challenge gives +10, so // 10 gives 0-10+ range.
    coins_bucket = min(int(obs["coins"]) // 10, 10)
    return (
        int(obs["agent"][0]),
        int(obs["agent"][1]),
        int(obs["challenge"][0]),
        int(obs["challenge"][1]),
        int(obs["board"].sum()),   # integer 1-25, NOT float
        coins_bucket,
    )


def choose_action(q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(N_ACTIONS)
    qs = [q[(state, a)] for a in range(N_ACTIONS)]
    return int(np.argmax(qs))


def train():
    env = GridWorldEnv()
    q = defaultdict(float)
    epsilon = EPSILON_START
    episode_rewards = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        state = obs_to_state(obs)
        prev_coverage = int(obs["board"].sum())
        total_reward = 0.0
        done = False

        while not done:
            action = choose_action(q, state, epsilon)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Dense reward shaping: small bonus per newly visited cell.
            # Keeps gradients flowing; actual episode reward is still sparse.
            new_coverage = int(obs["board"].sum())
            shaped_reward = reward #+ (new_coverage - prev_coverage) / (env.size ** 2)
            prev_coverage = new_coverage

            next_state = obs_to_state(obs)
            best_next = max(q[(next_state, a)] for a in range(N_ACTIONS))
            td_target = shaped_reward + GAMMA * best_next * (not done)
            q[(state, action)] += ALPHA * (td_target - q[(state, action)])

            state = next_state
            total_reward += reward  # log true reward, not shaped

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        if (ep + 1) % LOG_EVERY == 0:
            avg = np.mean(episode_rewards[-LOG_EVERY:])
            print(f"Episode {ep+1:5d} | avg reward: {avg:.4f} | epsilon: {epsilon:.3f} | Q-states: {len(q)}")

    env.close()
    return q


def evaluate(q, n_episodes=200):
    env = GridWorldEnv()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = obs_to_state(obs)
        total_reward = 0.0
        done = False
        while not done:
            action = choose_action(q, state, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs_to_state(obs)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    env.close()
    print(f"\nEval ({n_episodes} eps) | avg reward: {np.mean(rewards):.4f} "
          f"| min: {min(rewards):.4f} | max: {max(rewards):.4f}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

CELL = 100
PAD  = 20
HUD_H = 115

BG        = (30,  30,  30)
UNVISITED = (70,  70,  70)
VISITED   = (80,  160, 80)
AGENT_COL = (60,  120, 220)
CHAL_COL  = (220, 180, 40)
TEXT_COL  = (220, 220, 220)
ARROW_COL = (140, 140, 140)


def _draw_arrow(surface, r, c, action):
    """Small directional dot-arrow showing policy at a cell."""
    cx = PAD + c * CELL + CELL // 2
    cy = PAD + r * CELL + CELL // 2
    offsets = {0: (22, 0), 2: (-22, 0), 1: (0, -22), 3: (0, 22)}
    if action not in offsets:
        return
    dx, dy = offsets[action]
    import pygame
    pygame.draw.line(surface, ARROW_COL, (cx, cy), (cx + dx, cy + dy), 2)
    pygame.draw.circle(surface, ARROW_COL, (cx + dx, cy + dy), 4)


def visualize(q, n_episodes=5, fps=4):
    import pygame

    env = GridWorldEnv()
    size = env.size
    W = size * CELL + 2 * PAD
    H = size * CELL + 2 * PAD + HUD_H

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Q-Agent GridWorld  |  SPACE=step  F=fast  Q=quit")
    font     = pygame.font.SysFont("monospace", 17)
    font_sm  = pygame.font.SysFont("monospace", 13)
    clock    = pygame.time.Clock()

    step_mode = True   # True = wait for SPACE; False = run at fps

    for ep in range(n_episodes):
        obs, _ = env.reset()
        state = obs_to_state(obs)
        done = False
        step = 0
        total_reward = 0.0
        last_action = 0

        while not done:
            # In step mode, block until SPACE (advance), F (toggle fast), or Q (quit).
            # In fast mode, just drain the event queue and advance immediately.
            if step_mode:
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit(); env.close(); return
                        if event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_SPACE, pygame.K_RIGHT):
                                waiting = False
                            elif event.key == pygame.K_f:
                                step_mode = False
                                waiting = False
                            elif event.key == pygame.K_q:
                                pygame.quit(); env.close(); return
                    clock.tick(30)
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit(); env.close(); return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_f:
                            step_mode = True
                        elif event.key == pygame.K_q:
                            pygame.quit(); env.close(); return

            action = choose_action(q, state, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs_to_state(obs)
            done = terminated or truncated
            total_reward += reward
            last_action = action
            step += 1

            screen.fill(BG)
            board     = obs["board"]
            agent_pos = obs["agent"]
            chal_pos  = obs["challenge"]

            # --- Grid cells ---
            for r in range(size):
                for c in range(size):
                    color = VISITED if board[r, c] else UNVISITED
                    pygame.draw.rect(screen, color,
                                     (PAD + c*CELL + 2, PAD + r*CELL + 2, CELL-4, CELL-4))

            # --- Policy arrows (best movement action per cell, given current challenge pos) ---
            coverage = int(board.sum())
            coins_b  = min(int(obs["coins"]) // 10, 10)
            for r in range(size):
                for c in range(size):
                    cell_state = (r, c,
                                  int(chal_pos[0]), int(chal_pos[1]),
                                  coverage, coins_b)
                    qs = [q[(cell_state, a)] for a in range(4)]
                    if any(v != 0.0 for v in qs):
                        _draw_arrow(screen, r, c, int(np.argmax(qs)))

            # --- Challenge marker ---
            cr, cc = int(chal_pos[0]), int(chal_pos[1])
            cx = PAD + cc * CELL + CELL // 2
            cy = PAD + cr * CELL + CELL // 2
            pygame.draw.circle(screen, CHAL_COL, (cx, cy), CELL // 4)
            lbl = font_sm.render("C", True, (0, 0, 0))
            screen.blit(lbl, lbl.get_rect(center=(cx, cy)))

            # --- Agent marker ---
            ar, ac = int(agent_pos[0]), int(agent_pos[1])
            ax = PAD + ac * CELL + CELL // 2
            ay = PAD + ar * CELL + CELL // 2
            pygame.draw.circle(screen, AGENT_COL, (ax, ay), CELL // 3)
            lbl = font_sm.render("A", True, (255, 255, 255))
            screen.blit(lbl, lbl.get_rect(center=(ax, ay)))

            # --- HUD ---
            hy = size * CELL + 2 * PAD + 8
            pct = 100 * coverage // (size * size)
            mode_str = "STEP (F=fast)" if step_mode else "FAST (F=step)"
            lines = [
                f"Episode {ep+1}/{n_episodes}   Step {step:3d}   [{mode_str}]",
                f"Action : {ACTION_NAMES[last_action]:<10s}  Coins: {int(obs['coins'])}",
                f"Coverage: {coverage}/{size*size} ({pct}%)   Reward: {total_reward:.2f}",
            ]
            for i, line in enumerate(lines):
                screen.blit(font.render(line, True, TEXT_COL), (PAD, hy + i * 26))

            # --- Legend ---
            lx = W - 125
            pygame.draw.rect(screen, VISITED,   (lx, hy,      13, 13))
            screen.blit(font_sm.render("visited",   True, ARROW_COL), (lx + 17, hy))
            pygame.draw.rect(screen, UNVISITED, (lx, hy + 18, 13, 13))
            screen.blit(font_sm.render("unvisited", True, ARROW_COL), (lx + 17, hy + 18))

            pygame.display.flip()
            if not step_mode:
                clock.tick(fps)

    pygame.quit()
    env.close()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true",
                        help="Open pygame window after training")
    parser.add_argument("--fps", type=int, default=4,
                        help="Visualization speed (frames/sec)")
    parser.add_argument("--vis-episodes", type=int, default=5,
                        help="Number of episodes to visualize")
    args = parser.parse_args()

    print("Training Q-learning agent...")
    q = train()
    # evaluate(q)

    if args.visualize:
        visualize(q, n_episodes=args.vis_episodes, fps=args.fps)
