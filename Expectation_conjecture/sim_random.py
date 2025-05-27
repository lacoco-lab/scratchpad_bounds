from itertools import product, combinations
from fractions import Fraction
import random
import pandas as pd
import time

def custom_xor(*args):
    count_of_ones = sum(1 for a in args if a == 1)
    return 1 if count_of_ones % 2 == 1 else -1

def custom_and(*args):
    return min(args)

def compute_C_T0(num_vars, x, y, T):
    return custom_xor(*(custom_and(x[k], y[T - k]) for k in range(T + 1) if T - k < num_vars))

def compute_C_ji(j, x, y, order, T):
    if order == 0:
        return custom_xor(*(custom_and(x[k], y[j - k]) for k in range(j + 1) if j - k >= 0))
    elif j < (1 << (order)):
        return 0
    else:
        return custom_xor(*(custom_and(*([x[k], y[j - k]] + [item for o in comb for item in (x[o], y[j - o])]))
                             for comb in combinations(range(j + 1), order) for k in range(j + 1) if j - k >= 0))

def generate_random_subsets(num_vars, T, max_order=4, force_include_T=False):
    """
    Generates random subsets U, V, W. If force_include_T is True, ensures T is included in both V and W.
    """
    indices = list(range(T + 1))
    U = {i: random.sample([j for j in indices[:-1] if j >= (1 << i)], 
            k=min(len([j for j in indices[:-1] if j >= (1 << i)]), random.randint(0, max(1, len(indices) // (i+1))))) 
         for i in range(max_order)}

    V = random.sample(indices, k=min(len(indices), random.randint(1, max(1, len(indices) // 2)))) if indices else []
    W = random.sample(indices, k=min(len(indices), random.randint(1, max(1, len(indices) // 2)))) if indices else []

    if force_include_T:
        if T not in V:
            V.append(T)
        if T not in W:
            W.append(T)
    else:
        if T in V:
            V.remove(T)
        if T in W:
            W.remove(T)

    return U, V, W

def calculate_expectation(num_vars, T, U, V, W):
    variables = [False, True]
    combinations = product(variables, repeat=2 * num_vars)
    total = 0
    count = 0

    for combination in combinations:
        x = [1 if combination[i] else -1 for i in range(num_vars)]
        y = [1 if combination[num_vars + i] else -1 for i in range(num_vars)]

        c_T0 = compute_C_T0(num_vars, x, y, T)
        sum_U = custom_xor(*(custom_xor(*(compute_C_ji(j, x, y, order, T) for j in U[order])) for order in U))
        sum_V = custom_xor(*(x[j] for j in V)) if V else 0
        sum_W = custom_xor(*(y[j] for j in W)) if W else 0

        result = custom_xor(c_T0, sum_U, sum_V, sum_W)
        total += result
        count += 1

    return Fraction(total, count)


def calculate_multiple_expectations(num_vars, num_trials, max_order=4, specific_T=None):
    results = []
    start_time = time.time() 

    for i in range(num_trials):
        # Ensure 1/4 of trials include T in both V and W
        force_include_T = (i % 4 == 0)

        T = specific_T if specific_T is not None else random.randint(1, num_vars - 1)
        U, V, W = generate_random_subsets(num_vars, T, max_order, force_include_T)
        expectation = calculate_expectation(num_vars, T, U, V, W)
        results.append((T, U, V, W, expectation))

        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / (i + 1)) * num_trials
        remaining_time = estimated_total_time - elapsed_time
        progress_percent = ((i + 1) / num_trials) * 100

        print(f"\rProgress: {progress_percent:.2f}% | Completed {i+1}/{num_trials} | "
              f"Elapsed Time: {elapsed_time:.2f}s | Estimated Time Left: {remaining_time:.2f}s", 
              end="", flush=True)

    print("\n")  
    return results

num_trials = 100 
max_order = 3 

specific_T = 9  # Desired T
print(f"\nRunning for specific T={specific_T}...")
results_for_specific_T = calculate_multiple_expectations(specific_T + 1, num_trials, max_order, specific_T)

df_specific = pd.DataFrame(results_for_specific_T, columns=["T", "U", "V", "W", "Expectation"])
df_specific.to_csv(f"random_combinations_results_T{specific_T}.csv", index=False)
print(f"\nResults for T={specific_T} saved to 'random_combinations_results_T{specific_T}.csv'")
