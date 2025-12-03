import numpy as np
import jax
import jax.numpy as jnp


def _value_iteration(rewards, transition_probs, gamma):
    """
    Run value iteration, using procedure described in Sutton and Barto
    (2012). The end result is an updated value_table, from which one can
    deduce the policy for state s by taking the argmax (breaking ties
    randomly).
    """

    P = np.array(transition_probs)  # Shape: (S, A, S)
    R = np.array(rewards)           # Shape: (S, A) or (S, A, S)
    S = P.shape[0]

    if R.ndim == 3:
        expected_rewards = np.einsum('ijk,ijk->ij', P, R)
    else:
        expected_rewards = R

    value = np.zeros(S)
    k = 0
    while True:
        next_v_expected = np.tensordot(P, value, axes=([2], [0])) # 2nd dim of P dots 1st dim of value
        q_values = expected_rewards + gamma * next_v_expected
        new_value = np.max(q_values, axis=1)
        diff = np.max(np.abs(new_value - value))
        value = new_value

        k += 1
        if diff < 1e-2:
            break
        if k > 1e6:
            raise Exception("Value iteration not converging. Stopped at 1e6 iterations.")

    # Compute final Q value table for extracting optimal policy
    next_v_expected = np.tensordot(P, value, axes=([2], [0])) # 2nd dim of P dots 1st dim of value
    final_q = expected_rewards + gamma * next_v_expected
    return final_q


@jax.jit
def jax_value_iteration(rewards, transition_probs, gamma):
    """
    Fast implementation of value iteration
    """
    P = jnp.array(transition_probs, dtype=jnp.float32)
    R = jnp.array(rewards, dtype=jnp.float32)
    S = P.shape[0]

    # Pre-compute expected rewards
    if R.ndim == 3:
        expected_rewards = jnp.einsum('ijk,ijk->ij', P, R)
    else:
        expected_rewards = R

    V = jnp.zeros(S, dtype=jnp.float32)

    def cond_fn(state):
        _, diff, iter_count = state
        return (diff > 1e-2) & (iter_count < 1e6)

    def body_fn(state):
        v, _, iter_count = state
        next_v_expected = jnp.einsum('ijk,k->ij', P, v) # einsum is faster than tensordot and dot in jax
        new_v = jnp.max(expected_rewards + gamma * next_v_expected, axis=1)
        diff = jnp.max(jnp.abs(new_v - v))
        return new_v, diff, iter_count + 1

    # Iterate using JAX's while_loop
    final_v, _, _ = jax.lax.while_loop(cond_fn, body_fn, (V, jnp.inf, 0))

    # Compute final Q value table for extracting optimal policy
    next_v_expected = jnp.einsum('ijk,k->ij', P, final_v)
    final_q = expected_rewards + gamma * next_v_expected
    return final_q


def _argmax_breaking_ties_randomly(x):
    """Taken from Ken."""
    max_value = np.max(x)
    indices_with_max_value = np.flatnonzero(x == max_value)
    return np.random.choice(indices_with_max_value)


@jax.jit
def jax_argmax_breaking_ties_randomly(key, x):
    """
    JAX JIT compatible version with a non-variable length approach.
    """    
    max_val = jnp.max(x)
    mask = x == max_val                            # Get the mask of the maximum
    cumsum = jnp.cumsum(mask)                      # Get the cumulative sum of the mask
    num_ties = cumsum[-1]                          # Get the number of ties
    idx = jax.random.randint(key, (), 0, num_ties) # Sample one index from [0, 1, ..., num_ties - 1]
    return jnp.searchsorted(cumsum, idx + 1)       # Find the first occurance of the value idx + 1 and return its position
