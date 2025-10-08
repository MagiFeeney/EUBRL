import jax
from typing import *
import jax.numpy as jnp
from .modelbasedagent import ModelBasedAgent
import numpy as np


class EUBRLAgent(ModelBasedAgent):
    def __init__(self, dirichlet_param, reward_param, tau, precision, beta, use_jax: bool, eu_type: str, rng_key=None, eu_scale=1.0, reward_eu_scale=1.0, transition_eu_scale=1.0, use_eubrl_reward=True, use_normal_gamma_prior=False, instant_reward=True, env_reward=None, num_dir_samples=50, use_alertness=False, alertness_scale=100, alertness_max_eu=1000, use_sqrt_last=False, use_independent_max=False, **kwargs):
        super(EUBRLAgent, self).__init__(**kwargs)
        self.dirichlet_param = dirichlet_param
        self.reward_param = reward_param
        self.rng_key = rng_key
        if use_jax:
            assert rng_key is not None, f"rng key is None when using jax."
        self.use_jax = use_jax
        self.eu_type = eu_type

        self.use_eubrl_reward = use_eubrl_reward
        self.use_normal_gamma_prior = use_normal_gamma_prior

        self.use_sqrt_last = use_sqrt_last
        self.use_independent_max = use_independent_max

        self.use_alertness = use_alertness
        self.alertness_scale = alertness_scale
        self.alertness_max_eu = alertness_max_eu

        self.eu_scale = eu_scale
        self.reward_eu_scale = reward_eu_scale
        self.transition_eu_scale = transition_eu_scale

        self.num_dir_samples = num_dir_samples

        if env_reward is not None:
            self.known_reward = True
            self.reward = env_reward
        else:
            self.known_reward = False

        self.instant_reward = instant_reward

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

                if self.use_normal_gamma_prior or self.use_alertness:
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
                    self.mu = 0.0
                    self.tau = tau
                    self.precision = precision

        self.epistemic_uncertainty = None
        if self.use_independent_max:
            if self.instant_reward:
                self.U_max = np.zeros((self.num_states, self.num_actions, 1), dtype=np.float32)
            else:
                self.U_max = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
        else:
            self.U_max = 0

        if self.terminal_indexes is not None:
            if use_jax:
                self.non_terminal_indexes = ~jnp.isin(jnp.arange(self.num_states), jnp.array(self.terminal_indexes))
            else:
                self.non_terminal_indexes = np.delete(np.arange(self.num_states), self.terminal_indexes)

    def reset(self):
        super(EUBRLAgent, self).reset()
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
                        if self.use_normal_gamma_prior or self.use_alertness:
                            self.reward_squared_observations[self.last_state, self.last_action, next_state] += reward**2
                    else:
                        self.reward_observations[self.last_state, self.last_action] += reward
                        if self.use_normal_gamma_prior or self.use_alertness:
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

    def get_transition_probs(self, param):
        transition_probs = self.dirichlet_mean(param)[0]
        return transition_probs

    def get_epistemic_uncertainty(self, reward_eu, transition_eu, transition_probs):
        if not self.use_sqrt_last:
            transition_eu = np.sqrt(transition_eu) # √E_T(s, a)
        transition_eu *= self.transition_eu_scale

        # if (self.known_reward and self.instant_reward) or self.reward_param is not None:
        if self.instant_reward or self.reward_param is not None:
            reward_eu = np.sum(reward_eu * transition_probs, axis=-1) # E_s'[E_r(s, a, s')] = E_r(s, a)

        if not self.use_sqrt_last:
            reward_eu = np.sqrt(reward_eu) # √E_r(s, a)
        reward_eu *= self.reward_eu_scale

        if self.eu_type == "Product":
            epistemic_uncertainty = \
                transition_eu * reward_eu + \
                transition_eu * (self.reward**2 + reward_eu) + \
                reward_eu * ((transition_probs**2).sum(axis=-1) + transition_eu)
        else:
            assert reward_eu.shape == transition_eu.shape, f"{reward_eu.shape} != {transition_eu.shape}"
            epistemic_uncertainty = transition_eu + reward_eu

        if self.instant_reward or self.reward_param is not None:
            epistemic_uncertainty = epistemic_uncertainty[..., np.newaxis] # S x A x 1, since the instant_reward is of S x A x S

        return epistemic_uncertainty

    def get_transition_eu(self, param):
        if self.eu_type == "One-hot":   # Var[[E[s], E[r]]^T]
            var = self.dirichlet_var(param)
            transition_eu = np.sum(var, axis=-1)
        elif self.eu_type == "Product":               # Var[E[s] x E[r]]
            var = self.dirichlet_var(param)
            transition_eu = np.sum(var, axis=-1)
        elif self.eu_type == "Information Gain":
            transition_eu = self.dirichlet_information_gain(param)
        else:
            raise NotImplementedError

        return transition_eu

    def get_reward_and_eu(self):
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

                reward_eu = beta_new / (lam_new * alpha_new)
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

                    reward_eu = np.zeros_like(count)
                    reward_eu[count_mask] = self.alertness_max_eu
                    reward_eu[~count_mask] = 1 / (precision[~count_mask] * count[~count_mask])

                    self.reward[count_mask] = self.mu
                    self.reward[~count_mask] = self.reward_observations[~count_mask] / count[~count_mask]
                else:
                    if self.use_alertness:
                        precision[reward_var_mask] = 1 / (reward_var[reward_var_mask] * self.alertness_scale)

                    reward_eu = 1 / (self.tau + precision * count) # reward variance
                    self.reward = (self.tau * self.mu + precision * self.reward_observations) / (self.tau + precision * count)

        else:
            reward_eu = 0

        return reward_eu

    def _compute_policy(self):
        """Compute an optimal T-step policy for the current state."""
        self.policy_step = 0

        param = self.transition_observations + self.dirichlet_param
        transition_probs = self.get_transition_probs(param) # calculate the posterior mean

        reward_eu = self.get_reward_and_eu() # calculate both the reward and epistemic uncertainty in reward
        transition_eu = self.get_transition_eu(param)

        epistemic_uncertainty = self.get_epistemic_uncertainty(reward_eu, transition_eu, transition_probs) # calculate the epistemic uncertainty in transition

        if self.use_sqrt_last:
            self.epistemic_uncertainty = np.sqrt(epistemic_uncertainty) * self.eu_scale
        else:
            self.epistemic_uncertainty = epistemic_uncertainty * self.eu_scale

        # terminal states self looping
        if self.terminal_indexes is not None:
            transition_probs[self.terminal_indexes] = 0
            transition_probs[self.terminal_indexes, :, self.terminal_indexes] = 1

            # no reward at terminal states
            self.reward[self.terminal_indexes] = 0
            self.epistemic_uncertainty[self.terminal_indexes] = 0

        if self.use_independent_max:
            self.U_max = np.maximum(self.U_max, self.epistemic_uncertainty)
            self.U_max = np.where(self.U_max == 0, 1, self.U_max)
        else:
            self.U_max = max(self.U_max, np.max(self.epistemic_uncertainty))

        if self.use_jax:
            # calculate rewards
            epistemic_uncertainty = jnp.array(self.epistemic_uncertainty)
            pU = epistemic_uncertainty / self.U_max

            if self.use_eubrl_reward:
                rewards = (1 - pU) * jnp.array(self.reward) + pU * epistemic_uncertainty
            else:
                rewards = jnp.array(self.reward) + epistemic_uncertainty

            self.jax_value_iteration(rewards, transition_probs)
        else:                   # fall back to numpy
            pU = self.epistemic_uncertainty / self.U_max
            if self.use_eubrl_reward:
                rewards = (1 - pU) * self.reward + pU * self.epistemic_uncertainty
            else:
                rewards = self.reward + self.epistemic_uncertainty

            self._value_iteration(rewards, transition_probs)
