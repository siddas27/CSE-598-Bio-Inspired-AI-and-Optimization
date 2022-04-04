import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def f(x, y):
    return (x - 3.14) ** 2 + (y - 2.72) ** 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73)


def update(a, up, ug, objs):
    # Update params
    global V, X, pbest, pbest_obj, gbest, gbest_obj
    rp = np.random.uniform(low=0.0, high=up)
    rg = np.random.uniform(low=0.0, high=ug)

    V += a * (rp * (pbest - X) + rg * (gbest.reshape(-1, 1) - X))
    X += V
    obj = f(X[0], X[1])

    pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
    pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()
    objs.append(gbest_obj)


for j in range(5):
    a = 0.1
    up, ug = 0.5, 0.5 + j * 0.3
    objs = []

    # Compute and plot the function in 3D within [0,5]x[0,5]
    x, y = np.array(np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100)))
    z = f(x, y)

    # Find the global minimum
    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]

    # Hyper-parameter of the algorithm

    # Create particles
    n_particles = 20
    np.random.seed(100)
    X = np.random.rand(2, n_particles) * 5
    V = np.random.randn(2, n_particles) * 0.1

    # Initialize data
    pbest = X
    pbest_obj = f(X[0], X[1])
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()
    for i in range(20):
        update(a, up, ug, objs)

    print("PSO found best solution at f({})={}".format(gbest, gbest_obj))
    print("Global optimal at f({})={}".format([x_min, y_min], f(x_min, y_min)))

    plt.plot(objs, label=str(j))

plt.legend()
plt.xlabel('global best')
plt.ylabel('# iterations')
plt.show()
# plt.close()
