from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid (5x5 by default)
        self.size = size

        self.grid = np.zeros((size, size))
        self.grid[0, 0] = 1

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self.agent_location = np.array([0, 0], dtype=np.int32)
        self.challenge_location = np.array([-1, -1], dtype=np.int32)
        self.starting_agent_coins = 30
        self.agent_coins = self.starting_agent_coins
        self.challenge_win = 10

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "challenge": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
                "board": gym.spaces.Box(0, 1, shape=(size, size), dtype=int)
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self.action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
            # 4: np.array([5, 5]), # do challenge
        }
        self.steps = 0
    def get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self.agent_location, "challenge": self.challenge_location, "board": self.grid, "coins": self.agent_coins}

    def get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "full": self.grid.sum()
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self.agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self.randomize_challenge_location()

        self.grid = np.zeros((self.size, self.size))
        self.visit()
        self.agent_coins = self.starting_agent_coins

        observation = self.get_obs()
        info = self.get_info()
        self.steps = 0
        return observation, info
    
    def randomize_challenge_location(self):
        self.challenge_location = self.agent_location
        while np.array_equal(self.challenge_location, self.agent_location):
            self.challenge_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

    def visit(self):
        self.grid[self.agent_location[0], self.agent_location[1]] = 1

    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        prev_location = self.agent_location.copy()
        prev_count = int(self.grid.sum())

        should_terminate = False
        reward = 0
        # if action == 4:
        #     ...
        #     # if np.array_equal(self.agent_location, self.challenge_location):
        #     #     self.agent_coins += self.challenge_win
        #     #     self.randomize_challenge_location()
        #     # else:
        #     #     reward -= 1
        # elif self.agent_coins > 0:
            # self.agent_coins -= 1
        direction = self.action_to_direction[action]
        self.agent_location = np.clip(
            self.agent_location + direction, 0, self.size - 1
        )
        self.visit()
        # else:
        #     should_terminate = True

        count = self.grid.sum()
        delta = int(count) - prev_count
        reward += delta / (self.size ** 2)
        # if self.grid[self.agent_location[0], self.agent_location[1]] == 1 and action != 4:
            # reward -= 0.05
        # reward += self.agent_coins / 10
        terminated = count == self.size ** 2 or should_terminate
        self.steps += 1
        truncated = self.steps == 100
        # Check if agent reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        # reward = 1 if terminated else 0
        # count = self.grid.sum()
        # reward = count // (self.size ** 2)
        # terminated = reward == 1

        observation = self.get_obs()
        info = self.get_info()

        return observation, reward, terminated, truncated, info