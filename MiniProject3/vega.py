import math
import random
from operator import mod

from matplotlib import pyplot as plt

from utils import crossover, mutation, selection, gen_init_pop, reproduction
import numpy as np


def fitness(pop):
    fitness_values = np.zeros((pop.shape[0], 2))  # 2 values for each choromosome/solution
    for i,t in enumerate(pop):
        obj1 = t * math.sin(10 * math.pi * t)
        obj2 = 2.5 * t * math.cos(3*math.pi*t)

        fitness_values[i, 0] = obj1
        fitness_values[i, 1] = obj2

    return fitness_values

def pareto_front_finding(fitness_values, pop_index):
    pop_size = fitness_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool)    # all True initially
    for i in range(pop_size):
        for j in range(pop_size):
            if all(fitness_values[j] <= fitness_values[i]) and any(fitness_values[j] < fitness_values[i]):
                pareto_front[i] = 0 # i is not in pareto front becouse j dominates i
                break

def vector_evaluated_genetic_algorithm(f, N, iter_max, C, M, R):
    population = gen_init_pop(N)
    K = len(f(population)[1])
    # N = len(population)
    Ns = int(N / K)
    random.shuffle(population)
    for i in range(iter_max):
        ys = f(population)
        parents = []
        for k in range(K):
            # start = k*Ns
            # end = (k+1)*Ns
            fit= [y[k] for y in ys]
            subpop = [selection(fit, population) for _ in range(Ns)]
            parents += subpop
        # p = np.random.permutation(2 * N)
        # p_ind = parents[(p[i] - 1) % N][(p[i] - 1) / N]
        # selected_parents = [[p_ind(i), p_ind(i + 1)] for i in range(1, 2, 2 * N)]
        offsprings = reproduction(parents, C, M, N, R)
        population = parents + offsprings
        population = np.array(population, dtype=float)
    return population


pop = vector_evaluated_genetic_algorithm(fitness, 150, iter_max=150, C=0.2, M=0.2, R=80)
# Pareto front visualization
fitness_values = fitness(pop)
index = np.arange(pop.shape[0]).astype(int)
pareto_front_index = pareto_front_finding(fitness_values, index)
pop = pop[pareto_front_index, :]
print("_________________")
print("Optimal solutions:")
print("       x1               x2                 x3")
print(pop) # show optimal solutions
fitness_values = fitness_values[pareto_front_index]
print("______________")
print("Fitness values:")
print("  objective 1    objective 2")
print(fitness_values)
plt.scatter(fitness_values[:, 0],fitness_values[:, 1], label='Pareto optimal front')
plt.legend(loc='best')
plt.xlabel('Objective function F1')
plt.ylabel('Objective function F2')
plt.grid(b=1)
plt.show()