import numpy as np


class Trial(object):
    """
    Class for running trial(s) for a given agent and env.

    Parameters
    ----------
    agent: Agent
    env: Env
    min_iterations: int
        The minimum number of iterations for a trial.
    min_episodes: int
        The minimum number of episodes for a trial.
    max_episode_iteration: int
        The maximum number of iterations for each episode.
    num_episodes_for_success: int
        The minimum number of consecutive successful episodes for early stopping.
    """
    def __init__(self, agent, env, env_name, min_iterations=5000, min_episodes=100, max_episode_iteration=1000, num_episodes_for_success=10):
        self.agent = agent
        self.env = env
        self.env_name = env_name
        self.min_iterations = min_iterations
        self.min_episodes = min_episodes
        self.max_episode_iteration = max_episode_iteration
        self.num_episodes_for_success = num_episodes_for_success

        self.array_rewards_by_episode = None
        self.array_iteration_by_episode = None
        self.array_rewards_by_iteration = None

        if env_name == "DeepSea" and not self.env._deterministic:
            self.diagonal_indexes_natnum = np.arange(self.env._size) * self.env._size + np.arange(self.env._size)
            self.diagonal_indexes = (np.arange(self.env._size), np.arange(self.env._size))
        elif env_name == "LazyChain" and not self.env._deterministic:
            self.right_indexes = np.arange(self.env.left_length, self.env.left_length + self.env.right_length, 1)

    def run(self):
        iteration = episode = 0
        rewards_by_iteration = []
        rewards_by_episode = []
        iteration_by_episode = []
        success_by_episode = []
        self.agent.reset() # initialize storage

        put_count = 0

        keep_logging_suboptimal = True
        metrics = {}

        if self.env_name == "Loop":
            loop_num_successes = 0
            metrics["loop_iteration_of_success"] = []

        while iteration < self.min_iterations:
            # Initialize the episode.
            initial_state = state = self.env.reset()
            reward = None       # the first state doesn't give any reward information

            cumulative_reward = 0
            episode_iteration = 0

            while episode_iteration < self.max_episode_iteration: # maximum steps per episode
                action = self.agent.interact(reward, state) # update model with reward and state, then take an action
                state, reward, done, info = self.env.step(action)

                # Log rewards.
                if iteration < self.min_iterations:
                    rewards_by_iteration.append(reward)
                cumulative_reward += reward

                iteration += 1
                episode_iteration += 1

                if self.env_name == "Loop" and reward == self.env.get_max_reward():
                    loop_num_successes += 1
                    metrics["loop_iteration_of_success"].append(iteration)

                if done:
                    break

            # incorporate the information from the terminal state
            self.agent.update_model(reward, state)

            rewards_by_episode.append(cumulative_reward)
            iteration_by_episode.append(iteration)

            if self.success_checker(cumulative_reward, rewards_by_iteration, episode_iteration, action, info):
                success_by_episode.append(1)
            else:
                success_by_episode.append(0)

            episode += 1

            if (
                episode >= self.num_episodes_for_success and
                np.all(success_by_episode[-self.num_episodes_for_success: ])
            ):
                print(f"Early stop due to success at iteration {iteration} with num. of episodes {episode}")
                break

            if self.env_name == "Loop" and loop_num_successes > 10: # Separately check for Loop
                print(f"Early stop due to optimality at iteration {iteration} with num. of episodes {episode}")
                break

        print(f"final cummulative rewards in {iteration} total steps: {rewards_by_episode}")

        if self.env_name == "Loop":
            if loop_num_successes > 10:
                success = True
            else:
                success = False
        elif iteration >= self.min_iterations or episode >= self.num_episodes_for_success:
            success = False
            if np.all(success_by_episode[-self.num_episodes_for_success: ]):
                success = True
        else:
            success = False

        if success:
            print("Success")
        else:
            print("Failed")

        return np.array(rewards_by_iteration), np.array(rewards_by_episode), np.array(iteration_by_episode), np.array(success_by_episode), metrics

    def success_checker(self, cumulative_reward, rewards_by_iteration, episode_iteration, action, info):
        if self.env_name == "LazyChain":
            if self.env._deterministic:
                return cumulative_reward == self.env.left_length
            else:
                policy_regret = np.mean(np.argmax(self.agent.value_table[self.right_indexes], axis=-1) == 1)
                return policy_regret == 1.0 or (policy_regret >= 0.95 and rewards_by_iteration[-1] == self.env.right_reward) # algorithmic policy matches to the optimal policy almost surely
        elif self.env_name == "Bipolar":
            deterministic_reached = self.env.is_deterministic_goal_optimal and (episode_iteration <= 8 and (action == 1 or action == 2))
            stochastic_reached = (not self.env.is_deterministic_goal_optimal) and (episode_iteration <= 8 and (action == 0 or action == 3))
            return deterministic_reached or stochastic_reached
        elif self.env_name == "Loop":
            return cumulative_reward == self.env.get_max_reward()
        elif self.env_name == "Chain":
            return np.sum(rewards_by_iteration) >= 3400 # achieved at least 3400 (drawn from and as a result of the published results) cumulative rewards over 1000 steps
        elif self.env_name == "DeepSea":
            if self.env._deterministic:
                return info['success'] and episode_iteration == self.env._size # reach to chest and get a treasure in the minimally possible number of steps
            else:
                diagonal_value_table = self.agent.value_table[self.diagonal_indexes_natnum]
                policy_regret = np.mean(np.argmax(diagonal_value_table, axis=-1) == self.env._action_mapping[self.diagonal_indexes])

                return policy_regret == 1.0 # algorithmic policy matches exactly to the optimal policy along diagonal
        else:
            raise NotImplementedError

    def run_multiple(self, num_trials):
        self.array_rewards_by_episode = num_trials * [[]]
        self.array_success_by_episode = num_trials * [[]]
        self.array_iteration_by_episode = num_trials * [[]]
        self.array_metrics = num_trials * [[]]
        self.array_rewards_by_iteration = num_trials * [[]]
        for i in range(num_trials):
            self.array_rewards_by_iteration[i], self.array_rewards_by_episode[i], self.array_iteration_by_episode[i], self.array_success_by_episode[i], self.array_metrics[i] = self.run()
        self.array_rewards_by_episode = np.array(self.array_rewards_by_episode)
        self.array_iteration_by_episode = np.array(self.array_iteration_by_episode)
        self.array_success_by_episode = np.array(self.array_success_by_episode)
        self.array_metrics = np.array(self.array_metrics)
        self.array_rewards_by_iteration = np.array(self.array_rewards_by_iteration)
