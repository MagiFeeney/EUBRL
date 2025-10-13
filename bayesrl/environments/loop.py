import gym
import numpy as np


class Loop(gym.Env):
    def __init__(self, loop_length=5, left_reward=2, right_reward=1, num_loops=2):
        """
        Loop environment from Strens, M. (2000, June). A Bayesian framework for reinforcement learning. In ICML (Vol. 2000, pp. 943-950).
        Extension supports multiple loops.
        Args:
            loop_length (int): size of one side of the loop.
            left_reward (int): reward of traversing the left loop.
            right_reward (int): reward of traversing the right loop.
        """

        self.loop_length = loop_length
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.num_loops = num_loops

        self.num_states = num_loops * (self.loop_length - 1) + 1
        self.num_actions = num_loops

    def reset(self):
        self.last_state = self.state = 0
        return self.state

    def get_max_reward(self):
        return max(self.left_reward, self.right_reward)

    def is_terminal(self):
        return False            # non-episodic

    def step(self, action: int):
        if self.state == self.num_loops * (self.loop_length - 1): # the more rewarding loop
            reward = self.left_reward
        elif self.state != 0 and self.state % (self.loop_length - 1) == 0: # the less rewarding ones
            reward = self.right_reward
        else:                   # otherwise no reward
            reward = 0

        if self.state == 0:
            self.state = action * (self.loop_length - 1) + 1
        elif 1 <= self.state <= (self.num_loops - 1) * (self.loop_length - 1):
            current_loop = self.state // self.loop_length # start with zero
            self.state = (self.state + 1) % ((current_loop + 1) * (self.loop_length - 1) + 1)
        else:
            if action == self.num_loops - 1:
                self.state = (self.state + 1) % (self.num_loops * (self.loop_length - 1) + 1)
            else:
                self.state = 0

        done = self.is_terminal()
        info = {}
        return self.state, reward, done, info
