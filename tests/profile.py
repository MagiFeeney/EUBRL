from bayesrl.environments import LazyChain
import timeit
import numpy as np


def check_value_equal():
    discount_factor = 0.995
    chain_sizes = [50]

    for cs in chain_sizes:
        task = LazyChain(cs, cs - 1, cs, 2*cs - 1, -1, 0, 0)
        q1, v1 = task.solve_optimal_value_function_numpy(discount_factor)
        q2, v2 = task.solve_optimal_value_function_jax(discount_factor)
        max_err_q, mean_err_q = np.max(np.abs(q1-q2)), np.mean(np.abs(q1-q2))
        max_err_v, mean_err_v = np.max(np.abs(v1-v2)), np.mean(np.abs(v1-v2))
        print(f"N = {cs}... \nmax_err_q {max_err_q} | max_err_v {max_err_v}\nmean_err_q {mean_err_q} | mean_err_v {mean_err_v}\n")
        print(q1)
        print(q2)

        print("\n", v1)
        print(v2)


def profile_with_timeit():
    discount_factor = 0.995
    chain_sizes = [50, 100, 200, 500, 1000]

    print("Profiling NumPy Implementation...")
    for cs in chain_sizes:
        task = LazyChain(cs, cs - 1, cs, 2*cs - 1, -1, 0, 0)
        elapsed_time = timeit.timeit(
            lambda: task.solve_optimal_value_function_numpy(discount_factor),
            number=3,  # Run 3 times and take the average
        )
        print(f"NumPy: cs={cs}, avg_time={elapsed_time / 3:.4f} seconds")

    print("\nProfiling JAX Implementation...")
    for cs in chain_sizes:
        task = LazyChain(cs, cs - 1, cs, 2*cs - 1, -1, 0)
        elapsed_time = timeit.timeit(
            lambda: task.solve_optimal_value_function_jax(discount_factor),
            number=3,  # Run 3 times and take the average
        )
        print(f"JAX: cs={cs}, avg_time={elapsed_time / 3:.4f} seconds")


check_value_equal()
profile_with_timeit()
