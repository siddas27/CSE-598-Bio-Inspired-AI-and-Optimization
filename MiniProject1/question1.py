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
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, normalize

# Vector of place values, long form
p = [0.1, 0.01, 0.001, 0.0001]
p = np.array(p, dtype=float)

def array2real(x):
    return np.round(x[0] * x[1:].dot(p.T), 4)


def handle_constrains(x):
    if array2real(x) < -0.5:
        x[1] -= 5
    return x


def gen_init_pop(M, N):
    XV = np.random.randint(1, high=10, size=(M, N))
    XS = (-1) ** (np.random.randn(M, 1) < 0.5)
    X = np.concatenate((XS, XV), axis=1)
    return X


def objective(x):
    return x * math.sin(10 * math.pi * x) + 1


def get_fitness(X):
    x = np.round(X[:, 0] * X[:, 1:].dot(p.T), 4)
    vect_obj = np.vectorize(objective)
    F = vect_obj(x)
    return F


def selection(fitness, pop):
    # scaler = MinMaxScaler()
    fitness = np.array([(x - min(fitness)) / (max(fitness) - min(fitness)) for x in fitness])
    # normalized_fitness = normalize(scaled_X, norm='l1', axis=0, copy=True)
    pop_fitness = np.sum(fitness)
    individual_probabilities = [fitness[i] / pop_fitness for i in range(len(fitness))]
    a = np.arange(len(pop))
    s = np.random.choice(a, p=individual_probabilities)
    return pop[s]


def crossover(parent1, parent2, cross_prob):
    os1 = parent1.copy()
    os2 = parent2.copy()
    if np.random.rand() < cross_prob:
        cross_pt = np.random.randint(1, len(parent1))
        os1 = np.concatenate((os1[:cross_pt], os2[cross_pt:]), axis=0)
        os2 = np.concatenate((os2[:cross_pt], os1[cross_pt:]), axis=0)
    return [os1, os2]


def mutation(offspring, mut_prob):
    for i in range(len(offspring)):
        if np.random.rand() < mut_prob:
            if i == 0:
                offspring[0] = (-1) ** (np.random.randn() < 0.5)
            else:
                offspring[i] = np.random.randint(0, 9)
    return offspring


def reproduction(selected_parents, cross_prob=None, mut_prob=None, M=None, R=None):
    offsprings = []
    for i in range(len(selected_parents) - 1):
        if len(offsprings) >= M - R:
            break
        parent1, parent2 = selected_parents[i], selected_parents[i + 1]
        for os in crossover(parent1, parent2, cross_prob):
            offspring = mutation(os, mut_prob)
            offsprings.append(offspring)
    return offsprings


def terminal_criteria():
    pass


def genetic_algo(M, N, R, max_gen=None, cross_prob=None, mut_prob=None):
    pop = gen_init_pop(M, N)
    pop = np.apply_along_axis(handle_constrains,1,pop)
    best, worst, avg, best_indiv = [], [], [], []
    for generation in range(max_gen):
        # get fitness
        fitness = get_fitness(pop)
        # compute and store best, worst, avg, best_indiv
        best_fit_idx = np.argmax(fitness)
        best.append(fitness[best_fit_idx])
        worst.append(min(fitness))
        avg.append(np.mean(fitness))
        best_indiv.append(array2real(pop[best_fit_idx]))
        # selection
        selected_parents = [selection(fitness, pop) for _ in range(R)]
        # reproduction
        offsprings = reproduction(selected_parents, cross_prob, mut_prob, M, R)
        # new population
        total_pop = selected_parents + offsprings
        pop = np.array(total_pop)
        pop = np.apply_along_axis(handle_constrains, 1, pop)
        # if terminal_criteria():
        #     break

    # dictionary of lists
    dict = {'best': best, 'worst': worst, 'avg': avg, 'best_ind': best_indiv}
    df = pd.DataFrame(dict)
    df1=df.drop(['best_ind'], axis=1)
    # fig1 = plt.figure()
    df1.plot.line()
    plt.savefig('plot1.png')
    fig2 = plt.figure()
    df2 = df['best_ind']
    df2.plot.line()
    fig2.savefig('plot2.png')
    # saving the dataframe
    df.to_csv(f'data_{M}_{N}_{R}_{max_gen}_{cross_prob}_{mut_prob}.csv')


genetic_algo(7, 4, 3, 10, 0.5, 0.5)
