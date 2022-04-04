import math

import numpy as np
from numpy.random import rand, seed
from matplotlib import pyplot as plt
from decimal import Decimal, localcontext


def objective(x):
    return x * math.sin(10 * math.pi * x) + 1  # step[0] ** 2.0


def annealing_schedule(ch, temp, alpha, i):
    with localcontext() as ctx:
        ctx.prec = 100

        if ch == "slow":
            t = temp * Decimal(Decimal(alpha) ** i)
        else:
            t = temp - alpha * i
    return t


def sa(objective, iterations, temperature, beta, a_schedule):
    # Choose a random microstate
    x = np.random.uniform(low=0.0, high=0.03)
    y = objective(x)
    x_best, y_best = x, y
    y_updates = []
    funct_evals = []
    for i in range(iterations):
        # Choose a nearby microstate
        x1 = x_best + np.random.uniform(low=0.0, high=0.03)
        x1 = max(min(x1, 1.0), -0.5)
        y1 = objective(x1)
        del_y = y1 - y

        if del_y < 0:
            x, y = x1, y1
            y_updates.append(y1)
            funct_evals.append(i + 1)
        difference = y1 - y_best
        alpha = y1 / y_best

        if i % beta == 0:
            if i != 0:
                t = annealing_schedule(a_schedule, temperature, alpha, i)
            else:
                t = temperature
        # Metropolis  Acceptance Probability
        metropolis = np.exp(Decimal(-difference) / Decimal(t))
        if difference < 0 or rand() < metropolis:
            x_best, y_best = x1, y1
    return [funct_evals, y_updates]


if __name__ == "__main__":

    # initial temperature
    temperature = 12
    # define the total no. of iterations
    iterations = 1200
    beta = 20
    for a_s in ["fast", "slow"]:
        # perform the simulated annealing
        f_evals, y_updates = sa(objective, iterations, temperature, beta, a_s)
        # plotting the values
        if a_s == "fast":
            plt.plot(f_evals, y_updates, 'ro-', label=a_s)
        else:
            plt.plot(f_evals, y_updates, 'bo-', label=a_s)
    plt.legend()
    plt.xlabel('function evaluations')
    plt.ylabel('Loss')
    plt.savefig("sa4.png")
    plt.show()
