from typing import Optional
import numpy as np
import gymnasium as gym


# You can register the env so next time just gym.env()
class GridWordEnv(gym.Env):

    def __init__(self, size: int=5):
        self.size = size

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int)
        })

        self.action_space = gym.spaces.Discrete(4)

        #  Map above to below
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Seeding the random number generator
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def step(self, action):
        direction = self._action_to_direction[action]

        # clip to make sure, if out of bound, stay where it is, per element basis
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size-1
        )

        terminated = np.array_equal(self._agent_location, self._target_location)

        truncated = False

        reward = 1 if terminated else -0.01

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    

    def render(self):
        """Render the environment for human viewing."""
        if self.render_mode == "human":
            # Print a simple ASCII representation
            for y in range(self.size - 1, -1, -1):  # Top to bottom
                row = ""
                for x in range(self.size):
                    if np.array_equal([x, y], self._agent_location):
                        row += "A "  # Agent
                    elif np.array_equal([x, y], self._target_location):
                        row += "T "  # Target
                    else:
                        row += ". "  # Empty
                print(row)
            print()

agent = GridWordEnv(size=5)
agent.reset()
observation, reward, terminated, truncated, info = agent.step(0)
print("Initial observation: ", observation, " info: ", info)

while terminated == False:
    observation, reward, terminated, truncated, info = agent.step(np.random.randint(0,4))
    print("Observation: ", observation, "info: ", info)
