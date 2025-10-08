import gym
from gym import spaces
import numpy as np
from bsuite.environments import deep_sea


class DeepSeaGym(deep_sea.DeepSea, gym.Env):
    """Gym-compatible wrapper for bsuite's DeepSea environment."""

    def __init__(
            self,
            size=10,
            deterministic=True,
            unscaled_move_cost=0.01,
            randomize_actions=True,
            seed=None,
            mapping_seed=42,    # follows the practice from Bsuite, https://github.com/google-deepmind/bsuite/blob/main/bsuite/experiments/deep_sea/sweep.py
    ):
        # Initialize DeepSea environment
        super().__init__(
            size=size,
            deterministic=deterministic,
            unscaled_move_cost=unscaled_move_cost,
            randomize_actions=randomize_actions,
            seed=seed,
            mapping_seed=mapping_seed,
        )

        # Define observation space: N x N binary grid
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._size, self._size),
            dtype=np.float32
        )

        # Define action space: 0 (left), 1 (right)
        self.action_space = spaces.Discrete(2)

        self.num_states = self._size**2 + 1
        self.num_actions = 2

    def coordinate_to_natnum(self, obs):
        arr = obs.ravel()
        if np.sum(arr) == 0:
            return self._size**2
        else:
            return np.argmax(arr).item()

    def get_max_reward(self):
        return 1.0

    def reset(self):
        """Reset environment (Gym API)."""
        timestep = super().reset()
        obs = timestep.observation

        self.state = self.coordinate_to_natnum(obs)
        return self.state

    def step(self, action: int):
        timestep = self._step(action)
        self._reset_next_step = timestep.last()

        obs = timestep.observation
        reward = float(timestep.reward)
        done = timestep.last()
        info = self.bsuite_info()
        info['success'] = (self.state == self._size**2 - 1) and (action == self._action_mapping[self._size - 1, self._size - 1])
        if self.state == self._size**2 - self._size:
            info['location'] = "Left"
        elif self._size**2 - self._size < self.state < self._size**2 - 1:
            info['location'] = "Middle"
        elif self.state == self._size**2 - 1:
            info['location'] = "Right"

        self.state = self.coordinate_to_natnum(obs)
        return self.state, reward, done, info
