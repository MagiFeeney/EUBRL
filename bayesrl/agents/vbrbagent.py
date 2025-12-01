import jax
from typing import *
import jax.numpy as jnp
from .eubrlagent import EUBRLAgent
import numpy as np


class VBRBAgent(EUBRLAgent):
    def __init__(self, dirichlet_param, reward_param, tau, precision, beta, use_jax: bool, rng_key=None, transition_var_scale=1.0, reward_var_scale=1.0, use_normal_gamma_prior=False, instant_reward=True, env_reward=None, **kwargs):
        super().__init__(dirichlet_param, reward_param, tau, precision, beta, use_jax, rng_key, use_normal_gamma_prior, instant_reward, env_reward, **kwargs)

        self.transition_var_scale = transition_var_scale
        self.reward_var_scale = reward_var_scale

        self.reward_bonus = None

    def _compute_policy(self):
        """Compute an optimal T-step policy for the current state."""
        self.policy_step = 0

        param = self.transition_observations + self.dirichlet_param
        transition_probs = self.dirichlet_mean(param)[0]
        var = self.dirichlet_var(param)

        if not self.known_reward and self.reward_param is None:
            transition_var = np.sum(var, axis=-1)

            if self.instant_reward:
                count = self.transition_observations
            else:
                count = np.sum(self.transition_observations, axis=-1)

            count_safe = np.where(count == 0, 1, count)

            if self.use_normal_gamma_prior:
                lam_new = self.lam + count
                alpha_new = self.alpha + count / 2
                # sample variance
                s = (self.reward_squared_observations - self.reward_observations**2 / count_safe) / count_safe
                beta_new = self.beta + (count * s + ((self.lam * (self.reward_observations - count * self.mu)**2) / (count_safe * (self.lam + count)))) / 2

                reward_var = beta_new / (lam_new * alpha_new)
                self.reward = (self.lam * self.mu + self.reward_observations) / (self.lam + count)
            else:
                reward_var = 1 / (self.tau + self.precision * count) # reward variance
                self.reward = (self.tau * self.mu + self.precision * self.reward_observations) / (self.tau + self.precision * count)

            if self.instant_reward or self.reward_param is not None:
                reward_var = np.sum(reward_var * transition_probs, axis=-1) # E_s'[E_r(s, a, s')] = E_r(s, a)
            self.reward_bonus = np.sqrt(transition_var) * self.transition_var_scale + reward_var * self.reward_var_scale
        else:
            if self.instant_reward:
                self.reward_bonus = np.sqrt(np.sum(var, axis=-1, keepdims=True)) * self.transition_var_scale
            else:
                self.reward_bonus = np.sqrt(np.sum(var, axis=-1)) * self.transition_var_scale # S x A

        if self.instant_reward or self.reward_param is not None:
            self.reward_bonus = self.reward_bonus[..., np.newaxis] # S x A x 1, since the instant_reward is of S x A x S

        # terminal states self looping
        if self.terminal_indexes is not None:
            transition_probs[self.terminal_indexes] = 0
            transition_probs[self.terminal_indexes, :, self.terminal_indexes] = 1

            # no reward at terminal states
            self.reward[self.terminal_indexes] = 0
            self.reward_bonus[self.terminal_indexes] = 0

        rewards = self.reward + self.reward_bonus

        self.value_iteration(rewards, transition_probs)
