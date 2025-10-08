from typing import *
from .modelbasedagent import ModelBasedAgent
import numpy as np


class RMAXAgent(ModelBasedAgent):
    """Runs R-MAX only for an MDP, i.e., not a stochastic game, in order to simplify data structures."""
    def __init__(self, min_visit_count, use_jax: bool, use_max_reward=True, rng_key=None, **kwargs):
        super(RMAXAgent, self).__init__(**kwargs)
        self.min_visit_count = min_visit_count
        self.use_max_reward = use_max_reward

        self.rng_key = rng_key
        if use_jax:
            assert rng_key is not None, f"rng key is None when using jax."
        self.use_jax = use_jax

        if use_max_reward:
            self.Rmax = self.max_reward
        else:
            self.Rmax = 50 # arbitrarily set (!) will be updated through the course of learning

        self.reward = np.ones((self.num_states + 1, self.num_actions), dtype=np.float32) * self.Rmax
        self.reward_observations = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
        self.transition_observations = np.zeros((self.num_states + 1, self.num_actions, self.num_states + 1), dtype=np.float32)
        self.value_table = np.zeros((self.num_states + 1, self.num_actions), dtype=np.float32)

        # Rewrite the terminal indexes for envs having terminal states conflicting with the absorbing state
        if self.env_name == "LazyChain":
            self.terminal_indexes[-1] = -2
        elif self.env_name == "DeepSea":
            self.terminal_indexes[-1] = -2

    def reset(self):
        super(RMAXAgent, self).reset()
        self.reward.fill(self.Rmax)
        self.transition_observations.fill(0)
        self.value_table.fill(0)

    def update_model(self, reward, next_state):
        if reward is not None:
            n_sa = self.transition_observations[self.last_state, self.last_action].sum()

            # If unknown, gathering information
            if n_sa < self.min_visit_count:
                self.reward_observations[self.last_state, self.last_action] += reward
                self.transition_observations[self.last_state, self.last_action, next_state] += 1

            # Replace the guessed maximum reward with larger ones
            if not self.use_max_reward and self.Rmax < reward:
                self.reward[self.reward == self.Rmax] = reward
                self.Rmax = reward

            # Update the reward with sample mean (a generalization to stochastic environment) when associated (s, a) becomes known
            if n_sa + 1 == self.min_visit_count:
                self.reward[self.last_state, self.last_action] = self.reward_observations[self.last_state, self.last_action] / (n_sa + 1)
                self._compute_policy() # Compute the policy whenever a (s, a) is known

    def interact(self, reward, next_state):
        # update model
        self.update_model(reward, next_state)

        # Choose next action according to policy.
        if self.use_jax:
            next_action = self.jax_argmax_breaking_ties_randomly(self.value_table[next_state])
        else:
            next_action = self._argmax_breaking_ties_randomly(self.value_table[next_state])

        self.policy_step += 1
        self.last_state = next_state
        self.last_action = next_action

        return next_action

    def _compute_policy(self):
        """Compute an optimal T-step policy for the current state."""
        self.policy_step = 0

        transition_probs = np.zeros_like(self.transition_observations)
        n_sa = self.transition_observations.sum(axis=-1, keepdims=True)
        indice_unknown = n_sa.squeeze(-1) < self.min_visit_count # check unvisited (s, a) and get their indexes
        transition_probs[indice_unknown, -1] = 1 # absorbing state, which gives maximum rewards
        transition_probs[~indice_unknown] = self.transition_observations[~indice_unknown] / n_sa[~indice_unknown] # known state, maximum likelihood estimator

        if self.terminal_indexes is not None:
            transition_probs[self.terminal_indexes] = 0
            transition_probs[self.terminal_indexes, :, self.terminal_indexes] = 1

            self.reward[self.terminal_indexes] = 0

        if self.use_jax:
            self.jax_value_iteration(self.reward, transition_probs)
        else:
            self._value_iteration(self.reward, transition_probs)
