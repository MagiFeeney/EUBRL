import numpy as np
import gym
from .lazychain import LazyChain


class Chain(LazyChain, gym.Env):
    def __init__(self, left_length=0, left_reward=2, right_length=4, right_reward=10, on_chain_reward=0, p_return_to_start=0, p_error=0.2, random_state=None):
        super().__init__(
            left_length=left_length,
            left_reward=left_reward,
            right_length=right_length,
            right_reward=right_reward,
            on_chain_reward=on_chain_reward,
            p_return_to_start=p_return_to_start,
            p_error=p_error,
            random_state=random_state,
        )

        self.num_actions = 2

    def is_terminal(self):
        return False            # non-episodic

    def step(self, abstract_action: int):
        if self.p_return_to_start and self.random_state.rand() < self.p_return_to_start:
            self.reset()

        action = abstract_action
        # flip the abstract_action due to "slip"
        if self.p_error and self.random_state.random() < self.p_error:
            if abstract_action == 1:
                action = 0
            else:
                action = 1

        if action == 1:
            if self.state == self.num_states - 1:
                reward = self.right_reward
            else:
                reward = self.on_chain_reward
                self.state += 1 # not at the rightmost state
        else:
            reward = self.left_reward
            self.state = self.left_length # loop back to the leftmost state

        done = self.is_terminal()
        info = {}
        return self.observe(), reward, done, info

    def get_reward(self, use_instant_reward=True):
        if use_instant_reward:
            reward = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float32)

            reward[:, :, 0] = self.left_reward # ends at the first state
            reward[self.num_states - 1, :, self.num_states - 1] = self.right_reward # last state gets the highest reward
        else:
            reward = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

            # Right
            reward[np.arange(0, self.num_states - 1), 1] = self.on_chain_reward * (1 - self.p_error) + self.p_error * self.left_reward
            reward[self.num_states - 1, 1] = self.right_reward * (1 - self.p_error) + self.left_reward * self.p_error

            # Left
            reward[np.arange(0, self.num_states - 1), 0] = self.left_reward * (1 - self.p_error) + self.on_chain_reward * self.p_error
            reward[self.num_states - 1, 0] = self.left_reward * (1 - self.p_error) + self.right_reward * self.p_error

        return reward
