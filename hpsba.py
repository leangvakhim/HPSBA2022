import numpy as np
from tqdm import tqdm
import random

class hpsba:
    def __init__(self, func, bounds, dim, pop_size=30, max_iter=500):
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter

        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9 # omega_u in eq 17
        self.w_min = 0.2 # omega_l in eq 17
        self.a = 0.1
        self.sp = 0.6
        self.c_initial = 0.35 # initial chaotic

        self.x = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        self.v = np.zeros((pop_size, dim))

        self.fitness = np.zeros(pop_size)
        self.scent = np.zeros(pop_size) # F_i

        self.p_best_x = np.copy(self.x)
        self.p_best_f = np.full(pop_size, float('inf'))

        self.g_best_x = np.zeros(dim)
        self.g_best_f = float('inf')

    def calculate_fitness(self):
        for i in range(self.pop_size):
            val = float(self.func(self.x[i]))
            self.fitness[i] = val

            # update personal best (p_best)
            if val < self.p_best_f[i]:
                self.p_best_f[i] = val
                self.p_best_x[i] = self.x[i].copy()

            # update global best (g_best)
            if val < self.g_best_f:
                self.g_best_f = val
                self.g_best_x = self.x[i].copy()

    def run(self):
        self.calculate_fitness()

        convergence_curve = []
        c = self.c_initial
        for t in tqdm(range(1, self.max_iter + 1), desc="HPSBA Progress: "):
            # eq 17
            w = self.w_max - (self.w_max - self.w_min) * (t / self.max_iter)

            # eq 3
            for i in range(self.pop_size):
                self.scent[i] = c * (abs(self.fitness[i]) ** self.a)

            x_new = np.copy(self.x)

            for i in range(self.pop_size):
                r1 = np.random.random()
                r2 = np.random.random()

                # eq 13
                self.v[i] = (w * self.v[i] +
                             self.c1 * r1 * (self.p_best_x[i] - self.x[i]) +
                             self.c2 * r2 * (self.g_best_x - self.x[i]))

                # eq 14
                x_temp = self.x[i] + self.v[i]

                # eq 15
                r = np.random.random()
                f_i = abs(self.scent[i])

                if self.sp > r:
                    # global search
                    step = (r**2) * (self.g_best_x - x_temp) * f_i
                    x_new[i] = (w * x_temp) + step
                else:
                    # local search
                    j = np.random.randint(0, self.pop_size)
                    k = np.random.randint(0, self.pop_size)
                    step = (r**2) * (self.x[k] - self.x[j]) * f_i
                    x_new[i] = (w * x_temp) + step

                x_new[i] = np.clip(x_new[i], self.bounds[0], self.bounds[1])

            # greedy selection
            for i in range(self.pop_size):
                new_fitness = float(self.func(x_new[i]))

                if new_fitness < self.fitness[i]:
                    self.x[i] = x_new[i]
                    self.fitness[i] = new_fitness

                    if new_fitness < self.g_best_f:
                        self.g_best_f = new_fitness
                        self.g_best_x = x_new[i].copy()

            c = 4 * c * (1 - c)

            convergence_curve.append(self.g_best_f)

        return convergence_curve, self.g_best_x, self.g_best_f



