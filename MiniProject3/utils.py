# question 2
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def handle_constrains(x):
    if x < -0.5:
        x = -0.5
    elif x > 1.0:
        x = 1.0
    return x


def gen_init_pop(M):
    X = np.random.uniform(low=0.0, high=1.0, size=(M))
    return X


def objective(x):
    return x * math.sin(10 * math.pi * x) + 1


def get_fitness(X):
    vect_obj = np.vectorize(objective)
    F = vect_obj(X)
    return F


def selection(fitness, pop):
    # scaler = MinMaxScaler()
    fitness = np.array([(x - min(fitness)) / (max(fitness) - min(fitness)) for x in fitness])
    # normalized_fitness = normalize(scaled_X, norm='l1', axis=0, copy=True)
    pop_fitness = np.sum(fitness)
    individual_probabilities = np.array([fitness[i] / pop_fitness for i in range(len(fitness))])
    a = np.arange(len(pop))
    s = np.random.choice(a, p=individual_probabilities)
    return pop[s]


def crossover(parent1, parent2, cross_prob):
    os1 = parent1.copy()
    os2 = parent2.copy()
    if np.random.rand() < cross_prob:
        cross_pt = np.random.randint(1, len(str(parent1)))

        os1 = float(str(os1)[:cross_pt]+ str(os2)[cross_pt:])
        os2 = float(str(os2)[:cross_pt]+ str(os1)[cross_pt:])
    return [os1, os2]


def mutation(offspring, mut_prob):
    if np.random.rand() < mut_prob:
        offspring = np.random.uniform(low=0.0, high=1.0, size=(1,))
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
    pop = gen_init_pop(M)
    # pop = np.apply_along_axis(handle_constrains,0,pop)
    best, worst, avg, best_indiv = [], [], [], []
    for generation in range(max_gen):
        # get fitness
        fitness = get_fitness(pop)
        # compute and store best, worst, avg, best_indiv
        best_fit_idx = np.argmax(fitness)
        best.append(fitness[best_fit_idx])
        worst.append(min(fitness))
        avg.append(np.mean(fitness))
        best_indiv.append(pop[best_fit_idx])
        # selection
        selected_parents = [selection(fitness, pop) for _ in range(R)]
        # reproduction
        offsprings = reproduction(selected_parents, cross_prob, mut_prob, M, R)
        # new population
        total_pop = selected_parents + offsprings
        pop = np.array(total_pop, dtype=float)
        hc = np.vectorize(handle_constrains)
        pop = hc(pop)
        # if terminal_criteria():
        #     break

    # dictionary of lists
    dict = {'best': best, 'worst': worst, 'avg': avg, 'best_ind': best_indiv}
    df = pd.DataFrame(dict)
    df1 = df.drop(['best_ind'], axis=1)
    # fig1 = plt.figure()
    fig1 = plt.figure()
    plt.xlabel('generation')
    plt.ylabel('fitness')
    df1.plot.line()
    fig1.savefig('plot1.png')
    fig2 = plt.figure()
    plt.xlabel('generation')
    plt.ylabel('decision-variable')
    df2 = df['best_ind']
    df2.plot.line()
    fig2.savefig('q2plot2.png')
    # saving the dataframe
    df.to_csv(f'data_{M}_{N}_{R}_{max_gen}_{cross_prob}_{mut_prob}.csv')


# genetic_algo(300, 4, 100, 100, 0, 0.5)
