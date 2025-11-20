import numpy as np

def get_function_details(func_name):

    dim = 30

    benchmarks = {
        'F1': {'func': F1, 'lb': -100, 'ub': 100, 'dim': dim, 'name': 'Sphere'},
        'F2': {'func': F2, 'lb': -10, 'ub': 10, 'dim': dim, 'name': 'Schwefel 2.22'},
        'F3': {'func': F3, 'lb': -10, 'ub': 10, 'dim': dim, 'name': 'Schwefel 1.2'},
        'F4': {'func': F4, 'lb': -10, 'ub': 10, 'dim': dim, 'name': 'Schwefel 2.21'},
        'F5': {'func': F5, 'lb': -10, 'ub': 10, 'dim': dim, 'name': 'Step'},
        'F6': {'func': F6, 'lb': -1.28, 'ub': 1.28, 'dim': dim, 'name': 'Quartic'},
        'F7': {'func': F7, 'lb': -10, 'ub': 10, 'dim': dim, 'name': 'Exponential'},
        'F8': {'func': F8, 'lb': -1, 'ub': 1, 'dim': dim, 'name': 'Sum Power'},
        'F9': {'func': F9, 'lb': -10, 'ub': 10, 'dim': dim, 'name': 'Sum Square'},
        'F10': {'func': F10, 'lb': -10, 'ub': 10, 'dim': dim, 'name': 'Rosenbrock'},
        'F11': {'func': F11, 'lb': -5.12, 'ub': 5.12, 'dim': dim, 'name': 'Zakharov'},
        'F12': {'func': F12, 'lb': -5, 'ub': 5, 'dim': dim, 'name': 'Trid'},
        'F13': {'func': F13, 'lb': -100, 'ub': 100, 'dim': dim, 'name': 'Elliptic'},
        'F14': {'func': F14, 'lb': -100, 'ub': 100, 'dim': dim, 'name': 'Cigar'},
        'F15': {'func': F15, 'lb': -10, 'ub': 10, 'dim': dim, 'name': 'Tablet'},
        'F16': {'func': F16, 'lb': -5.12, 'ub': 5.12, 'dim': dim, 'name': 'Rastrigin'},
        'F17': {'func': F17, 'lb': -5.12, 'ub': 5.12, 'dim': dim, 'name': 'NCRastrigin'},
        'F18': {'func': F18, 'lb': -20, 'ub': 20, 'dim': dim, 'name': 'Ackley'},
        'F19': {'func': F19, 'lb': -600, 'ub': 600, 'dim': dim, 'name': 'Griewank'},
        'F20': {'func': F20, 'lb': -10, 'ub': 10, 'dim': dim, 'name': 'Alpine'},
        'F21': {'func': F21, 'lb': -10, 'ub': 10, 'dim': dim, 'name': 'Penalized 1'},
        'F22': {'func': F22, 'lb': -5, 'ub': 5, 'dim': dim, 'name': 'Penalized 2'},
        'F23': {'func': F23, 'lb': -2, 'ub': 2, 'dim': dim, 'name': 'Levy'},
        'F24': {'func': F24, 'lb': -1, 'ub': 1, 'dim': dim, 'name': 'Weierstrass'},
        'F25': {'func': F25, 'lb': -20, 'ub': 20, 'dim': dim, 'name': 'Solomon'},
        'F26': {'func': F26, 'lb': -5, 'ub': 5, 'dim': dim, 'name': 'Bohachevsky'},
    }
    return benchmarks.get(func_name)

def F1(x): # Sphere
    return np.sum(x**2)

def F2(x): # Schwefel 2.22
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x): # Schwefel 1.2
    dim = len(x)
    result = 0
    for i in range(dim):
        result += np.sum(x[:i+1])**2
    return result

def F4(x): # Schwefel 2.21
    return np.max(np.abs(x))

def F5(x): # Step
    return np.sum((x + 0.5)**2)

def F6(x): # Quartic
    dim = len(x)
    # i ranges from 1 to dim
    i = np.arange(1, dim + 1)
    return np.sum(i * x**4) + np.random.random()

def F7(x): # Exponential
    return np.exp(0.5 * np.sum(x))

def F8(x): # Sum Power
    dim = len(x)
    i = np.arange(1, dim + 1)
    return np.sum(np.abs(x)**(i + 1))

def F9(x): # Sum Square
    dim = len(x)
    i = np.arange(1, dim + 1)
    return np.sum(i * x**2)

def F10(x): # Rosenbrock
    dim = len(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def F11(x): # Zakharov
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, len(x)+1) * x)
    return sum1 + sum2**2 + sum2**4

def F12(x): # Trid
    dim = len(x)
    term1 = (x[0] - 1)**2
    term2 = 0
    for i in range(1, dim):
        coeff = i + 1
        term2 += coeff * (2 * x[i]**2 - x[i-1])**2
    return term1 + term2

def F13(x): # Elliptic
    dim = len(x)
    i = np.arange(1, dim + 1)
    condition = (i - 1) / (dim - 1)
    return np.sum((10**6)**condition * x**2)

def F14(x): # Cigar
    return x[0]**2 + 10**6 * np.sum(x[1:]**2)

def F15(x): # Tablet
    return 10**6 * x[0]**2 + np.sum(x[1:]**6)
    # return 10**6 * x[0]**2 + np.sum(x[1:]**2)

# --- Multimodal Functions ---

def F16(x): # Rastrigin
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def F17(x): # NCRastrigin (Non-Continuous)
    y = np.where(np.abs(x) < 0.5, x, np.round(2 * x) / 2)
    return np.sum(y**2 - 10 * np.cos(2 * np.pi * y) + 10)

def F18(x): # Ackley
    dim = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / dim)) - np.exp(sum2 / dim) + 20 + np.e

def F19(x): # Griewank
    dim = len(x)
    i = np.arange(1, dim + 1)
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(i)))
    return sum_sq - prod_cos + 1

def F20(x): # Alpine
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

# Helper for Penalized Functions
def u(x, a, k, m):
    result = np.zeros_like(x)
    mask1 = x > a
    mask2 = x < -a
    result[mask1] = k * (x[mask1] - a)**m
    result[mask2] = k * (-x[mask2] - a)**m
    return result

def F21(x): # Penalized 1
    dim = len(x)
    y = 1 + (x + 1) / 4
    term1 = 10 * np.sin(np.pi * y[0])**2
    term2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
    term3 = (y[-1] - 1)**2
    penalty = np.sum(u(x, 10, 100, 4))
    return (np.pi / dim) * (term1 + term2 + term3) + penalty

def F22(x): # Penalized 2
    dim = len(x)
    term1 = np.sin(3 * np.pi * x[0])**2
    term2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
    term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    penalty = np.sum(u(x, 5, 100, 4))
    return 0.1 * (term1 + term2 + term3) + penalty

def F23(x): # Levy
    dim = len(x)
    w = 1 + (x - 1) / 4
    term1 = np.sin(3 * np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(3 * np.pi * w[1:])**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def F24(x): # Weierstrass
    a = 0.5
    b = 3
    kmax = 20
    dim = len(x)

    result = 0
    for i in range(dim):
        for k in range(kmax + 1):
            result += a**k * np.cos(2 * np.pi * b**k * (x[i] + 0.5))

    offset = 0
    for k in range(kmax + 1):
        offset += a**k * np.cos(2 * np.pi * b**k * 0.5)

    return result - dim * offset

def F25(x): # Solomon
    norm_sq = np.sum(x**2)
    return 1 - np.cos(2 * np.pi * np.sqrt(norm_sq)) + 0.1 * np.sqrt(norm_sq)

def F26(x): # Bohachevsky
    result = 0
    dim = len(x)
    for i in range(dim - 1):
        result += (x[i]**2 + 2 * x[i+1]**2 -
                   0.3 * np.cos(3 * np.pi * x[i]) -
                   0.4 * np.cos(4 * np.pi * x[i+1]) + 0.7)
    return result