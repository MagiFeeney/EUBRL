import jax
from typing import *
import jax.numpy as jnp
from .eubrlagent import EUBRLAgent
import numpy as np


class BEBAgent(EUBRLAgent):
    def __init__(self, dirichlet_param, reward_param, tau, precision, beta, use_jax: bool, eu_type: str, rng_key=None, eu_scale=1.0, reward_eu_scale=1.0, transition_eu_scale=1.0, use_eubrl_reward=True, use_normal_gamma_prior=False, instant_reward=True, env_reward=None, num_dir_samples=50, use_alertness=False, alertness_scale=100, alertness_max_eu=1000, **kwargs):
        super().__init__(dirichlet_param, reward_param, tau, precision, beta, use_jax, rng_key, eu_scale, reward_eu_scale, transition_eu_scale, use_normal_gamma_prior, instant_reward, env_reward, **kwargs)

    def get_transition_probs(self, param):
        transition_probs, a_0 = self.dirichlet_mean(param)
        return transition_probs, np.squeeze(a_0)

    def get_epistemic_uncertainty(self, a_0):
        epistemic_uncertainty = self.eu_scale / (1 + a_0 + self.transition_observations.sum(-1))
        if self.instant_reward or self.reward_param is not None:
            epistemic_uncertainty = epistemic_uncertainty[..., np.newaxis] # S x A x 1, since the instant_reward is of S x A x S
        return epistemic_uncertainty

    def get_reward(self):
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
                s = (self.reward_squared_observations - self.reward_observations**2 / count_safe) / count_safe
                beta_new = self.beta + (count * s + ((self.lam * (self.reward_observations - count * self.mu)**2) / (count_safe * (self.lam + count)))) / 2

                self.reward = (self.lam * self.mu + self.reward_observations) / (self.lam + count)
            else:
                if self.use_alertness:
                    reward_var = (self.reward_squared_observations - self.reward_observations**2 / count_safe) / count_safe
                    reward_var_mask = (reward_var > 0)

                precision = np.ones_like(count) * self.precision

                if self.tau == 0:
                    count_mask = count == 0

                    if self.use_alertness:
                        nonzero_and_var_positive = reward_var_mask & ~count_mask
                        precision[nonzero_and_var_positive] = 1 / (reward_var[nonzero_and_var_positive] * self.alertness_scale)

                    self.reward[count_mask] = self.mu
                    self.reward[~count_mask] = self.reward_observations[~count_mask] / count[~count_mask]
                else:
                    if self.use_alertness:
                        precision[reward_var_mask] = 1 / (reward_var[reward_var_mask] * self.alertness_scale)

                    self.reward = (self.tau * self.mu + precision * self.reward_observations) / (self.tau + precision * count)

    def _compute_policy(self):
        """Compute an optimal T-step policy for the current state."""
        self.policy_step = 0

        param = self.transition_observations + self.dirichlet_param
        transition_probs, a_0 = self.get_transition_probs(param) # calculate the posterior mean

        self.get_reward() # calculate both the reward and epistemic uncertainty in reward
        self.epistemic_uncertainty = self.get_epistemic_uncertainty(a_0) # calculate the epistemic uncertainty in transition

        # terminal states self looping
        if self.terminal_indexes is not None:
            transition_probs[self.terminal_indexes] = 0
            transition_probs[self.terminal_indexes, :, self.terminal_indexes] = 1

            # no reward at terminal states
            self.reward[self.terminal_indexes] = 0
            self.epistemic_uncertainty[self.terminal_indexes] = 0

        rewards = self.reward + self.epistemic_uncertainty

        self.value_iteration(rewards, transition_probs)
