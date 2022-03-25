import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def handle_constrains(x):
    if x < -0.5:
        x = -0.5
    elif x > 1.0:
        x = 1.0
    return x


def gen_init_pop(M):
    X = np.random.uniform(low=-0.5, high=1.0, size=(M))
    return X


def objective(x):
    return x * math.sin(10 * math.pi * x) + 1


def plot_obj(nx, ny, filename):
    plt.close()
    x = np.linspace(-0.5, 1, 1000, endpoint=True, dtype=float)
    vect_obj = np.vectorize(objective)
    y = vect_obj(x)
    plt.xlabel('x in [-0.5,1]')
    plt.ylabel('f(x)')
    plt.plot(x, y)
    plt.scatter(nx, ny, c='g')
    plt.savefig(filename + '.png')


def get_fitness(X):
    vect_obj = np.vectorize(objective)
    F = vect_obj(X)
    return F


def selection(fitness, pop):
    fitness = np.array([(x - min(fitness)) / (max(fitness) - min(fitness)) for x in fitness])
    pop_fitness = np.sum(fitness)
    individual_probabilities = np.array([fitness[i] / pop_fitness for i in range(len(fitness))])
    a = np.arange(len(pop))

    s = np.random.choice(a, p=individual_probabilities)
    return pop[s]


def crossover(parent1, parent2, cross_prob):
    if np.random.rand() < cross_prob:
        os1 = "%f" % parent1.copy()
        os2 = "%f" % parent2.copy()

        cross_pt = np.random.randint(1, min(len(os1), len(os2)))

        temp = os1
        os1 = os1[:cross_pt] + os2[cross_pt:]
        os2 = os2[:cross_pt] + temp[cross_pt:]
        os1 = os1.replace("..", ".")
        os2 = os2.replace("..", ".")
        return [float(os1), float(os2)]

    return [parent1, parent2]


def mutation(offspring, mut_prob):
    if np.random.rand() < mut_prob:
        offspring = np.random.uniform(low=0.0, high=1.0, size=(1,))
    return offspring


def reproduction(selected_parents, cross_prob=None, mut_prob=None, M=None, R=None):
    offsprings = []
    for i in range(len(selected_parents) - 1):
        parent1, parent2 = selected_parents[i], selected_parents[i + 1]
        for os in crossover(parent1, parent2, cross_prob):
            offspring = mutation(os, mut_prob)
            offsprings.append(offspring)
        # if len(offsprings) > M - R:
        #     break
    return offsprings[:29]


def terminal_criteria():
    pass


def sharing(distance, sigma, alpha):
    res = 0
    if distance < sigma:
        res += 1 - (distance / sigma) ** alpha
    return res


def shared_fitness(pop, sigma, alpha):
    f = get_fitness(pop)
    # pop = pop.reshape(-1, 1)
    # nz = np.zeros_like(pop)
    Fs = []
    for i, fi in enumerate(f):
        # a= pop[i]
        dists = list(map(lambda b: np.sqrt(np.sum((b - pop[i]) ** 2, axis=0)), pop))
        tmp = [sharing(d, sigma, alpha) for d in dists]
        den = sum(tmp)
        Fsi = fi / den
        Fs.append(Fsi)
    return np.array(Fs)


def cluster_fitness(pop, n_clusters, dmax, alpha):
    f = get_fitness(pop)
    f = f.reshape(-1, 1)
    # pop = pop.reshape(-1,1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(f)
    unique_elements, counts_elements = np.unique(kmeans.labels_, return_counts=True)
    ncs = {k: v for k, v in zip(unique_elements, counts_elements)}
    F = []
    for i, fi in enumerate(f):
        cluster = kmeans.labels_[i]
        distance = abs(fi - kmeans.cluster_centers_[cluster][0])
        nc = ncs[cluster]
        den = nc * (1 - (distance / (2 * dmax)) ** alpha)
        Fi = fi / den
        F.append(Fi)

    return np.array(F)


def genetic_algo(M, R, max_gen=None, cross_prob=None, mut_prob=None, niching="without_niching"):
    pop = gen_init_pop(M)
    k = 8
    for generation in range(max_gen):
        # get fitness
        f = get_fitness(pop)
        if niching == "sharing":
            fitness = shared_fitness(pop, 0.1, 1)
        elif niching == "clustering":
            fitness = cluster_fitness(pop, n_clusters=k, dmax=0.1, alpha=1)
            fitness = fitness.reshape(50, )
        else:
            fitness = f
        bestk = f
        bestk_ind = pop
        selected_parents = [selection(fitness, pop) for _ in range(R + 1)]
        # reproduction
        offsprings = reproduction(selected_parents, cross_prob, mut_prob, M, R)
        # new population
        total_pop = selected_parents + offsprings
        pop = np.array(total_pop, dtype=object)
        hc = np.vectorize(handle_constrains)
        pop = hc(pop)
        # if terminal_criteria():
        #     break

    plot_obj(bestk_ind,bestk, filename=niching)


if __name__ == "__main__":
    for n in ["sharing", "clustering", "without_niching"]:
        genetic_algo(50, 20, max_gen=400, cross_prob=0.0, mut_prob=0.0, niching=n)
