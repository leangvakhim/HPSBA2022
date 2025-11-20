import numpy as np
import matplotlib.pyplot as plt
from hpsba import hpsba
import benchmark

# funcs_to_test = ['F1', 'F8', 'F10', 'F16', 'F21']
funcs_to_test = ['F1']

population_size = 30
max_iter = 500
t = 30
result_list = []

results = {}

# print(f"{'Function':<15} | {'Best Fitness':<15} | {'Mean (Curve)':<15}")
# print("-" * 50)

for fname in funcs_to_test:
    for _ in range(t):
        details = benchmark.get_function_details(fname)
        if not details:
            print(f"Function {fname} not found.")
            continue

        func = details['func']
        lb = details['lb']
        ub = details['ub']
        dim = details['dim']
        name = details['name']

        bounds = (lb, ub)
        optimizer = hpsba(func, bounds, dim, population_size, max_iter)
        best_pos, best_score, curve = optimizer.run()
        result_list.append(best_score)

        results[fname] = curve

    # print(f"{fname:<15} | {best_score:.4e}      | {np.mean(curve):.4e}")

mean_value = np.mean(result_list)
std_value = np.std(result_list)
min_value = np.min(result_list)
max_value = np.max(result_list)
print(f"Function name: {name}")
print(f"Mean: {mean_value:.4e}")
print(f"Standard Deviation: {std_value:.4e}")
print(f"Minimum Value: {min_value:.4e}")
print(f"Maximum Value: {max_value:.4e}")
# plt.figure(figsize=(10, 6))

# for fname, curve in results.items():
#     plt.semilogy(curve, label=fname)

# plt.title('HPSBA Convergence on Selected Benchmarks')
# plt.xlabel('Iteration')
# plt.ylabel('Fitness (Log Scale)')
# plt.legend()
# plt.grid(True, which="both", ls="-")
# plt.show()