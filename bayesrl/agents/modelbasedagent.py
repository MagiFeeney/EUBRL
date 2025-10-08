import jax
import jax.numpy as jnp
from .agent import Agent
import numpy as np
from scipy.special import digamma


class ModelBasedAgent(Agent):
    """Runs R-MAX only for an MDP, i.e., not a stochastic game, in order to simplify data structures."""
    def __init__(self, T, max_reward, env_name, **kwargs):
        super(ModelBasedAgent, self).__init__(**kwargs)
        self.T = T

        self.policy_step = self.T # To keep track of where in T-step policy the agent is in; initialized to recompute policy
        self.transition_observations = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.value_table = np.zeros((self.num_states, self.num_actions))
        self.max_reward = max_reward
        self.env_name = env_name

        # Initialize masks for terminal states
        if env_name == "Bipolar" or env_name == "GridWorld":
            up_row = np.arange(9)
            bottom_row = 8 * 9 + np.arange(9)
            left_col = np.arange(9) * 9
            right_col = np.arange(9) * 9 + 8

            wall_index = np.concatenate([up_row, bottom_row, left_col, right_col])
            goal_index = np.array([10, 70])

            self.terminal_indexes = np.concatenate([wall_index, goal_index])
        elif env_name == "LazyChain":
            self.terminal_indexes = np.array([0, -1])
        elif env_name == "Chain" or env_name == "Loop":
            self.terminal_indexes = None
        elif env_name == "DeepSea":
            self.terminal_indexes = np.array([-1])
        else:
            raise NotImplementedError

    def reset(self):
        super(ModelBasedAgent, self).reset()
        self.policy_step = self.T # To keep track of where in T-step policy the agent is in; initialized to recompute policy
        self.transition_observations.fill(0)
        self.value_table.fill(0)

    def _value_iteration(self, rewards, transition_probs):
        """
        Run value iteration, using procedure described in Sutton and Barto
        (2012). The end result is an updated value_table, from which one can
        deduce the policy for state s by taking the argmax (breaking ties
        randomly).
        """

        value_dim = transition_probs.shape[0]
        value = np.zeros(value_dim)
        # value = np.max(self.value_table, axis=-1)

        length = len(rewards.shape)

        k = 0
        while True:
            diff = 0
            for s in range(value_dim):
                old = value[s]
                if length == 3:
                    value[s] = \
                    np.max(
                        np.sum(
                            transition_probs[s] *
                            (rewards[s] + self.discount_factor*np.array([value,]*self.num_actions)),
                            axis=1
                        )
                    )
                elif length == 2:
                    value[s] = \
                    np.max(
                        rewards[s] +
                        np.sum(
                            transition_probs[s] * self.discount_factor*np.array([value,]*self.num_actions),
                            axis=1
                        )
                    )
                diff = max(diff, abs(old - value[s]))
            k += 1
            if diff < 1e-2:
                break
            if k > 1e6:
                raise Exception("Value iteration not converging. Stopped at 1e6 iterations.")

        for s in range(value_dim):
            if length == 3:
                self.value_table[s] = \
                    np.sum(
                        transition_probs[s] *
                        (rewards[s] + self.discount_factor*np.array([value,]*self.num_actions)),
                        axis=1
                    )
            elif length == 2:
                self.value_table[s] = \
                    rewards[s] + \
                    np.sum(
                        transition_probs[s] * self.discount_factor*np.array([value,]*self.num_actions),
                        axis=1
                    )


    def jax_value_iteration(self, rewards, transition_probs):
        """
        fast implementation of value iteration
        """

        transition_probs = jnp.array(transition_probs)
        value_dim = transition_probs.shape[0]
        value = jnp.zeros(value_dim)

        length = len(rewards.shape)

        def value_iteration_step(value):
            # Compute Q-values for all states and actions
            if length == 3:
                q_values = jnp.sum(transition_probs * (rewards + self.discount_factor * value[jnp.newaxis, :]), axis=2)
            elif length == 2:
                q_values = rewards + self.discount_factor * jnp.sum(transition_probs * value[jnp.newaxis, :], axis=2)
            # Update value function
            new_value = jnp.max(q_values, axis=1)
            diff = jnp.max(jnp.abs(new_value - value))
            return new_value, diff

        def cond_fn(state):
            _, diff, k = state
            return (diff > 1e-2) & (k < 1e6)

        def body_fn(state):
            value, _, k = state
            new_value, diff = value_iteration_step(value)
            return new_value, diff, k + 1

        # Iterate using JAX's while_loop
        value, _, k = jax.lax.while_loop(cond_fn, body_fn, (value, jnp.inf, 0))

        # Compute final value table for optimal policy
        if length == 3:
            self.value_table = jnp.sum(transition_probs * (rewards + self.discount_factor * value[jnp.newaxis, :]), axis=2)
        elif length == 2:
            self.value_table = rewards + self.discount_factor * jnp.sum(transition_probs * value[jnp.newaxis, :], axis=2)

    def _argmax_breaking_ties_randomly(self, x):
        """Taken from Ken."""
        max_value = np.max(x)
        indices_with_max_value = np.flatnonzero(x == max_value)
        return np.random.choice(indices_with_max_value)

    def jax_argmax_breaking_ties_randomly(self, x):
        """Taken from Ken."""
        max_value = jnp.max(x)
        indices_with_max_value = jnp.where(x == max_value)[0]
        # Generate a new random key for each random operation
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return jax.random.choice(subkey, indices_with_max_value)

    def dirichlet_mean(self, param):
        a_sum = np.sum(param, axis=-1, keepdims=True)
        return param / a_sum, a_sum

    def dirichlet_var(self, param):
        mean, s = self.dirichlet_mean(param)
        return (mean * (1. - mean)) / (s + 1.)

    def dirichlet_sample(self, param, num_dir_samples=()):
        """
        param: np.ndarray of shape (B, N)
        num_dir_samples: optional shape to repeat sampling (e.g., (K,) → (K, B, N))
        """
        param = np.asarray(param)  # shape: (B, N)
        if num_dir_samples:
            param = np.broadcast_to(param, num_dir_samples + param.shape)
            flat_param = param.reshape(-1, param.shape[-1])
        else:
            flat_param = param

        # Check per-row if all elements are >= 1e-3
        gamma_mask = np.all(flat_param >= 1e-3, axis=1)
        out = np.zeros_like(flat_param)

        # Gamma sampling for rows where all alpha_i >= 1e-3
        if np.any(gamma_mask):
            gamma_rows = flat_param[gamma_mask]
            gamma_samples = np.random.gamma(shape=gamma_rows, scale=1.0)
            gamma_samples /= gamma_samples.sum(axis=1, keepdims=True)
            out[gamma_mask] = gamma_samples
            print("normal alpha", np.any(np.isnan(out[~gamma_mask])), np.any(np.isinf(out[~gamma_mask])))

        # One-hot (categorical) sampling where any alpha_i < 1e-3
        if np.any(~gamma_mask):
            alpha_rows = flat_param[~gamma_mask]
            probs = alpha_rows / alpha_rows.sum(axis=1, keepdims=True)
            # Avoid division by zero
            probs = np.where(np.isnan(probs), 1.0 / param.shape[-1], probs)

            idx = np.array([
                np.random.choice(param.shape[-1], p=p) for p in probs
            ])
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(len(probs)), idx] = 1.0
            out[~gamma_mask] = one_hot
            print("small alpha", np.any(np.isnan(out[~gamma_mask])), np.any(np.isinf(out[~gamma_mask])))

        # Reshape back to include sampling dims if needed
        return out.reshape(param.shape)

    def dirichlet_information_gain(self, param):
        """
        Most direct computation using the derived closed form:

        E[KL(P || P̄)] = sum_i (α_i/Σα_j) * [ψ(α_i + 1) - ψ(Σα_j + 1) - log(α_i/Σα_j)]
        """
        posterior_mean, param_sum = self.dirichlet_mean(param) # S x A x S
        MI = np.sum(posterior_mean * (digamma(param + 1) - digamma(param_sum + 1) - np.log(posterior_mean + 1e-16)), axis=-1) # S x A

        return MI
