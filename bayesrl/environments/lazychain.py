import gym
from ..utils import check_random_state


class LazyChain(gym.Env):
    def __init__(self, left_length, left_reward, right_length, right_reward, on_chain_reward, p_return_to_start=0, p_error=0, random_state=None):
        super(LazyChain, self).__init__()

        self.left_length = left_length
        self.left_reward = left_reward
        self.right_length = right_length
        self.right_reward = right_reward
        self.on_chain_reward = on_chain_reward
        self.p_return_to_start = p_return_to_start
        self.num_states = self.left_length + self.right_length + 1
        self.num_actions = 3
        self.random_state = check_random_state(random_state)
        self.p_error = p_error

        self._deterministic = p_error == 0

        self.reset()

    def reset(self):
        self.state = self.left_length
        return self.state

    def step(self, action):
        do_nothing = False
        if self.p_return_to_start and self.random_state.rand() < self.p_return_to_start:
            self.reset()
        elif action == 0:
            if self.p_error and self.random_state.random() < self.p_error:
                self.state += 1
            else:
                self.state -= 1
        elif action == 1:
            if self.p_error and self.random_state.random() < self.p_error:
                self.state -= 1
            else:
                self.state += 1
        else:
            do_nothing = True

        if do_nothing:
            reward = 0
        elif self.state == 0:
            reward = self.left_reward
        elif self.state == self.num_states - 1:
            reward = self.right_reward
        else:
            reward = self.on_chain_reward

        done = self.is_terminal()
        info = {}
        return self.observe(), reward, done, info

    def observe(self):
        return self.state

    def is_terminal(self):
        return self.state == 0 or self.state == self.num_states - 1

    def render(self, mode='human'):
        """Render the environment to the screen"""
        print(f"Current state: {self.state}")

    def close(self):
        """Clean up resources"""
        pass

    def get_max_reward(self):
        return max(self.left_reward, self.right_reward)

    def visualize_path(self, isFoundMax, initial_state, actions, rewards):
        """Visualize the path given a set of actions and rewards from a trajectory"""
        print(f"initial state is {initial_state}")
        sequence = ""
        for i, a in enumerate(actions):
            if a == 0:
                text_action = "l"
            elif a == 1:
                text_action = "r"
            else:
                text_action = "s"

            rew = rewards[i]
            if rew >= 0:
                sequence += text_action + "+" + str(rew) + " "
            else:
                sequence += text_action + str(rew) + " "

        print("success" if isFoundMax else "failed", sequence)

    def get_reward(self, use_instant_reward=True):
        if use_instant_reward:
            # Ground truth reward for deterministic LazyChain
            reward = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float32)

            reward[np.arange(1, self.num_states - 1, 1), 0, np.arange(0, self.num_states - 2, 1)] = self.on_chain_reward
            reward[1, 0, 0] = (self.num_states - 3) // 2

            reward[np.arange(1, self.num_states - 1, 1), 1, np.arange(2, self.num_states, 1)] = self.on_chain_reward
            reward[self.num_states - 2, 1, self.num_states - 1] = self.num_states - 2
        else:
            raise NotImplementedError("The expected reward is not supported in LazyChain")

        return reward
