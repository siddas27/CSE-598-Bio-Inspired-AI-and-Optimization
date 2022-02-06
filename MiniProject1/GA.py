# Using any language you would like, implement “from scratch” a simple genetic algo-
# rithm (GA) that encodes the decision variable x as a string of a sign variable and 4 base-10 digits (e.g., -0.1234
# is (-1,1,2,3,4)). You are free to choose whatever GA parameters would like (e.g., crossover operator, muta-
# tion operator, population size, number of parents, crossover probability, mutation probability, how to handle
# constraints), but your algorithm must have both crossover and mutation.
# • Describe your GA-implementation choices (5 points)
# • Generate two plots:
# – Best, worst, and average fitness for each successive generation of the GA (5 points)
# – Best individual for each successive generation generation of the GA (5 points)
# • Plot the function f and evaluate the quality of the solution found by the GA (5 points)

import numpy as np

# Vector of place values, long form
p = [1.0, 0.1, 0.01, 0.001, 0.0001]
p = np.array(p, dtype=float)

# Define decision variable x
xs = -1
xv = np.array([0, 1, 2, 3, 4], dtype=float)

# Method to generate two random parents
xs1 = np.power(-1, np.random.randint(0, high=2, size=1))
xs2 = np.power(-1, np.random.randint(0, high=2, size=1))
xv1 = np.random.randint(1, high=10, size=5)
xv2 = np.random.randint(1, high=10, size=5)
ft = np.round(xv1.dot(p.T), 4)
print(ft)

M = 6
N = 5


def objective():
    pass


def get_fitness(X):
    F = np.round(X.dot(p.T), 4)
    Fn = np.linalg.norm(F)
    return Fn


def gen_init_pop(M, N):
    XV = np.random.randint(1, high=10, size=(M, N))
    XS = (-1) ** (np.random.randn(M, 1) < 0.5)
    print(XV.shape, XS.shape)
    X = np.concatenate((XS, XV), axis=1)
    return X


def selection(R, init_pop):
    pass


def crossover(parent1, parent2, cross_prob):
    os1 = parent1.copy()
    os2 = parent2.copy()
    if np.random.rand() < cross_prob:
        cross_pt = np.random.randint(1, len(parent1))
        os1 = os1[:cross_pt] + os2[cross_pt:]
        os2 = os2[:cross_pt] + os1[cross_pt:]
    return [os1, os2]


def mutation(offspring, mut_prob):
    for i in range(len(offspring)):
        if np.random.rand() < mut_prob:
            if i==0:
                offspring[0] = (-1) ** (np.random.randn(M, 1) < 0.5)
            else:
                offspring[i] = np.random.randint(0,9)
    return offspring


def reproduction(selected_parents, cross_prob=None, mut_prob=None):
    offsprings = []
    for i in range(0, M, 2):
        parent1, parent2 = selected_parents[i], selected_parents[i + 1]
        for os in crossover(parent1, parent2, cross_prob):
            offspring = mutation(os, mut_prob)
            offsprings.append(offspring)
    return offsprings


def terminal_criteria():
    pass


def genetic_algo(M, N, max_gen=None):
    init_pop = gen_init_pop(M, N)
    scores = [objective(p) for p in init_pop]
    selected_parents = selection(R, init_pop)
    for generation in range(max_gen):
        offspring = reproduction(selected_parents)

        if terminal_criteria():
            break


s = gen_init_pop(100, 4)
print(s)
