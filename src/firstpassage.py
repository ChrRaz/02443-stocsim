import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def simulate(rng, N, mu, sigma, lam, y_dist):
    phi1 = math.sqrt(mu ** 2 / sigma ** 4 + 2 * lam / sigma ** 2) - mu / sigma ** 2
    phi2 = math.sqrt(mu ** 2 / sigma ** 4 + 2 * lam / sigma ** 2) + mu / sigma ** 2

    t = np.zeros(N)
    p = np.zeros(N)
    v = np.zeros(N)
    w = np.zeros(N)
    a = np.zeros(N)
    y = np.zeros(N)
    m = np.zeros(N)

    for i in range(1, N):
        # Advance time
        dt = rng.exponential(1 / lam)
        t[i] = t[i - 1] + dt

        # Step 1
        v[i] = rng.exponential(1 / phi1)
        w[i] = rng.exponential(1 / phi2)

        # Step 2
        y[i] = y_dist()

        # Step 3
        p[i] = a[i - 1] + (v[i] - w[i])
        a[i] = p[i] + y[i]
        m[i] = max(m[i - 1], a[i - 1] + v[i], a[i])

    return t, p, a, m


if __name__ == '__main__':
    rng = np.random.default_rng(3245)

    n = 1000
    N = 1000
    M = np.zeros((n, N))
    a10 = np.zeros(N)
    a100 = np.zeros(N)
    a500 = np.zeros(N)
    a1000 = np.zeros(N)
    a2000 = np.zeros(N)
    a100000 = np.zeros(N)

    for i in range(n):
        _, _, _, m = simulate(rng, N, -110, 100, 1, lambda: rng.exponential(100))
        M[i, :] = m
        if i % 10 == 0:
            print(i)

    for j in range(N):
        a10[j] = np.mean(M[:, j] >= 10)
        a100[j] = np.mean(M[:, j] >= 100)
        a500[j] = np.mean(M[:, j] >= 500)
        a1000[j] = np.mean(M[:, j] >= 1000)
        a2000[j] = np.mean(M[:, j] >= 2000)
        a100000[j] = np.mean(M[:, j] >= 100000)

    # %% Display simulation
    plt.figure(1)
    plt.plot(range(N), a10)
    plt.plot(range(N), a100)
    plt.plot(range(N), a500)
    plt.plot(range(N), a1000)
    plt.plot(range(N), a2000)
    plt.plot(range(N), a100000)
    plt.legend(['$P(sup_{0\\leq t \\leq\\tilde{t}} X_t \\geq 10)$',
                '$P(sup_{0\\leq t \\leq\\tilde{t}} X_t \\geq 100)$',
                '$P(sup_{0\\leq t \\leq\\tilde{t}} X_t \\geq 500)$',
                '$P(sup_{0\\leq t \\leq\\tilde{t}} X_t \\geq 1000)$',
                '$P(sup_{0\\leq t \\leq\\tilde{t}} X_t \\geq 2000)$',
                '$P(sup_{0\\leq t \\leq\\tilde{t}} X_t \\geq 100000)$'],
               loc=4)
    plt.title(
        'First passage probabilities for $X_t = -110\cdot t + 100\cdot B_t + \sum_{i=1}^{N_t} Y_i ,     \lambda = 1, \mathbb{E}(Y_i) = 100$')
    plt.xlabel('$N_t$')
    plt.show()

    # %%
    rng = np.random.default_rng(2714)

    n = 1000
    N = 1000
    M = np.zeros((n, N))
    a10 = np.zeros(N)
    a1000 = np.zeros(N)
    a100000 = np.zeros(N)

    # %%
    for i in range(n):
        _, _, _, m = simulate(rng, N, -110, 100, 1, lambda: rng.exponential(100))
        M[i, :] = m
        if i % 10 == 0:
            print(i)

    for j in range(N):
        a10[j] = np.mean(M[:, j] >= 10)
        a1000[j] = np.mean(M[:, j] >= 1000)
        a100000[j] = np.mean(M[:, j] >= 100000)

    dt10 = a10[-1] - a10[-101]
    dt1000 = a1000[-1] - a1000[-101]
    dt100000 = a100000[-1] - a100000[-101]

    print(f'P(X>10) = {a10[-1]}, dP/di = {dt10}')
    print(f'P(X>1000) = {a1000[-1]}, dP/di = {dt1000}')
    print(f'P(X>100000) = {a100000[-1]}, dP/di = {dt100000}')

    # %%
    for i in range(n):
        _, _, _, m = simulate(rng, N, -2200, 2000, 1, lambda: rng.exponential(2000))
        M[i, :] = m
        if i % 10 == 0:
            print(i)

    for j in range(N):
        a10[j] = np.mean(M[:, j] >= 10)
        a1000[j] = np.mean(M[:, j] >= 1000)
        a100000[j] = np.mean(M[:, j] >= 100000)

    dt10 = a10[-1] - a10[-101]
    dt1000 = a1000[-1] - a1000[-101]
    dt100000 = a100000[-1] - a100000[-101]

    print(f'P(X>10) = {a10[-1]}, dP/di = {dt10}')
    print(f'P(X>1000) = {a1000[-1]}, dP/di = {dt1000}')
    print(f'P(X>100000) = {a100000[-1]}, dP/di = {dt100000}')
    # %%
    for i in range(n):
        _, _, _, m = simulate(rng, N, -2500, 10000, 1, lambda: 0)
        M[i, :] = m
        if i % 10 == 0:
            print(i)

    for j in range(N):
        a10[j] = np.mean(M[:, j] >= 10)
        a1000[j] = np.mean(M[:, j] >= 1000)
        a100000[j] = np.mean(M[:, j] >= 100000)

    dt10 = a10[-1] - a10[-101]
    dt1000 = a1000[-1] - a1000[-101]
    dt100000 = a100000[-1] - a100000[-101]

    print(f'P(X>10) = {a10[-1]}, dP/di = {dt10}')
    print(f'P(X>1000) = {a1000[-1]}, dP/di = {dt1000}')
    print(f'P(X>100000) = {a100000[-1]}, dP/di = {dt100000}')
    # #%%
    for i in range(n):
        _, _, _, m = simulate(rng, N, -2200, 2000, 1, lambda: 100 * rng.pareto(1.05))
        M[i, :] = m
        if i % 10 == 0:
            print(i)

    for j in range(N):
        a10[j] = np.mean(M[:, j] >= 10)
        a1000[j] = np.mean(M[:, j] >= 1000)
        a100000[j] = np.mean(M[:, j] >= 100000)

    dt10 = a10[-1] - a10[-101]
    dt1000 = a1000[-1] - a1000[-101]
    dt100000 = a100000[-1] - a100000[-101]

    print(f'P(X>10) = {a10[-1]}, dP/di = {dt10}')
    print(f'P(X>1000) = {a1000[-1]}, dP/di = {dt1000}')
    print(f'P(X>100000) = {a100000[-1]}, dP/di = {dt100000}')
    # %%
    n = 1000
    N = 10000
    M = np.zeros((n, N))
    a10 = np.zeros(N)
    a1000 = np.zeros(N)
    a100000 = np.zeros(N)

    for i in range(n):
        _, _, _, m = simulate(rng, N, -2000, 2000, 1, lambda: rng.exponential(2000))
        M[i, :] = m
        if i % 10 == 0:
            print(i)

    for j in range(N):
        a10[j] = np.mean(M[:, j] >= 10)
        a1000[j] = np.mean(M[:, j] >= 1000)
        a100000[j] = np.mean(M[:, j] >= 100000)

    dt10 = a10[-1] - a10[-101]
    dt1000 = a1000[-1] - a1000[-101]
    dt100000 = a100000[-1] - a100000[-101]

    print(f'P(X>10) = {a10[-1]}, dP/di = {dt10}')
    print(f'P(X>1000) = {a1000[-1]}, dP/di = {dt1000}')
    print(f'P(X>100000) = {a100000[-1]}, dP/di = {dt100000}')
