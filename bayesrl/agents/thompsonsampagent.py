from typing import *
from .modelbasedagent import ModelBasedAgent
import numpy as np


class ThompsonSampAgent(ModelBasedAgent):
    def __init__(self, dirichlet_param, reward_param, tau, precision, beta, use_jax: bool, use_normal_gamma_prior=False, transition_sampling=True, rng_key=None, instant_reward=True, env_reward=None, **kwargs):
        super(ThompsonSampAgent, self).__init__(**kwargs)
        self.dirichlet_param = dirichlet_param
        self.reward_param = reward_param

        if env_reward is not None:
            self.known_reward = True
            self.reward = env_reward
        else:
            self.known_reward = False

        self.instant_reward = instant_reward

        self.rng_key = rng_key
        if use_jax:
            assert rng_key is not None, f"rng key is None when using jax."
        self.use_jax = use_jax

        self.use_normal_gamma_prior = use_normal_gamma_prior
        self.transition_sampling = transition_sampling

        if not self.known_reward:
            if self.reward_param is not None: # assign the reward when it is first known
                self.reward = np.full((self.num_states, self.num_actions, self.num_states), self.reward_param, dtype=np.float32)
            else:               # use conjugate priors: [Normal with known precision, Normal-Gamma]
                if self.instant_reward:
                    self.reward = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float32)
                    self.reward_observations = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float32)
                else:
                    self.reward = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
                    self.reward_observations = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

                if self.use_normal_gamma_prior:
                    if self.instant_reward:
                        self.reward_squared_observations = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float32)
                    else:
                        self.reward_squared_observations = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

                if self.use_normal_gamma_prior:
                    self.mu = 0.0
                    self.lam = beta
                    self.alpha = 1.0
                    self.beta = beta
                else:
                    self.mu = 0
                    self.tau = tau
                    self.precision = precision

    def reset(self):
        super(ThompsonSampAgent, self).reset()
        self.transition_observations = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float32)
        self.value_table = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

    def update_model(self, reward, next_state):
        if reward is not None:
            # Update the reward associated with (s,a,s') if first time.
            if not self.known_reward:
                if self.reward_param is not None:
                    if self.reward[self.last_state, self.last_action, next_state] == self.reward_param:
                        self.reward[self.last_state, self.last_action, next_state] = reward
                else:
                    if self.instant_reward:
                        self.reward_observations[self.last_state, self.last_action, next_state] += reward
                        if self.use_normal_gamma_prior:
                            self.reward_squared_observations[self.last_state, self.last_action, next_state] += reward**2
                    else:
                        self.reward_observations[self.last_state, self.last_action] += reward
                        if self.use_normal_gamma_prior:
                            self.reward_squared_observations[self.last_state, self.last_action] += reward**2

            # Update set of states reached by playing a.
            self.transition_observations[self.last_state, self.last_action, next_state] += 1

    def interact(self, reward, next_state):
        # update model
        self.update_model(reward, next_state)

        # Update transition probabilities after every T steps
        if self.policy_step == self.T:
            self._compute_policy()

        # Choose next action according to policy.
        if self.use_jax:
            next_action = self.jax_argmax_breaking_ties_randomly(self.value_table[next_state])
        else:
            next_action = self._argmax_breaking_ties_randomly(self.value_table[next_state])

        self.policy_step += 1
        self.last_state = next_state
        self.last_action = next_action

        return self.last_action

    def _compute_policy(self):
        """Compute an optimal T-step policy for the current state."""
        self.policy_step = 0
        param = self.transition_observations + self.dirichlet_param

        if not self.known_reward and self.reward_param is None:
            if self.instant_reward:
                count = self.transition_observations
            else:
                count = np.sum(self.transition_observations, axis=-1)

            count_safe = np.where(count == 0, 1, count)

            if self.use_normal_gamma_prior:
                lam_new = self.lam + count
                alpha_new = self.alpha + count / 2
                # sample variance
                # s = (self.reward_squared_observations - self.reward_observations**2 / count_safe) / count_safe

                mean = self.reward_observations / count_safe
                mean_sq = self.reward_squared_observations / count_safe
                s = (mean_sq - mean**2).clip(min=0) # to prevent the accuracy issue due to dtype np.float32

                beta_new = self.beta + (count * s + ((self.lam * (self.reward_observations - count * self.mu)**2) / (count_safe * (self.lam + count)))) / 2

                reward_var = beta_new / (lam_new * alpha_new)
                reward_mean = (self.lam * self.mu + self.reward_observations) / (self.lam + count)
                assert reward_var.shape == reward_mean.shape, f"Normal-Gamma reward std and mean shape not match"
            else:
                reward_var = 1 / (self.tau + self.precision * count) # reward variance
                reward_mean = (self.tau * self.mu + self.precision * self.reward_observations) / (self.tau + self.precision * count)

            self.reward = reward_mean + np.random.normal(size=reward_var.shape) * np.sqrt(reward_var)

        if self.transition_sampling:
            gamma_samples = np.random.gamma(shape=param.clip(min=1e-3), scale=1.0) # Prevent sampling issue when dirichlet_param is too small
            transition_probs = gamma_samples / gamma_samples.sum(axis=-1, keepdims=True)
        else:
            transition_probs = self.dirichlet_mean(param)[0]

        # terminal states self looping
        if self.terminal_indexes is not None:
            transition_probs[self.terminal_indexes] = 0
            transition_probs[self.terminal_indexes, :, self.terminal_indexes] = 1

            self.reward[self.terminal_indexes] = 0
            
        self.value_iteration(self.reward, transition_probs)
