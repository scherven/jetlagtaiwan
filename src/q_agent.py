"""Simple tabular Q-learning agent for GridWorldEnv."""

import sys
import os
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


def obs_to_state(obs):
    return (int(obs["agent"][0]), int(obs["agent"][1]),
            int(obs["challenge"][0]), int(obs["challenge"][1]),
            obs["board"].sum() / 25, int(obs["coins"]))


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
        total_reward = 0.0
        done = False

        while not done:
            action = choose_action(q, state, epsilon)
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = obs_to_state(obs)
            print(action, next_state)
            done = terminated or truncated

            best_next = max(q[(next_state, a)] for a in range(N_ACTIONS))
            td_target = reward + GAMMA * best_next * (not done)
            q[(state, action)] += ALPHA * (td_target - q[(state, action)])

            state = next_state
            total_reward += reward

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        if (ep + 1) % LOG_EVERY == 0:
            avg = np.mean(episode_rewards[-LOG_EVERY:])
            print(f"Episode {ep + 1:5d} | avg reward: {avg:.4f} | epsilon: {epsilon:.3f} | Q-states: {len(q)}")

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
    print(f"\nEval ({n_episodes} eps) | avg reward: {np.mean(rewards):.4f} | min: {min(rewards):.4f} | max: {max(rewards):.4f}")


if __name__ == "__main__":
    print("Training Q-learning agent...")
    q = train()
    evaluate(q)
