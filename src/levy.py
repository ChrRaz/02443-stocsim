#!/usr/bin/python3
import math

import matplotlib.pyplot as plt
import numpy as np


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

    # %% Demo a single simulation
    t, p, a, m = simulate(rng, 200, 0, 1, 1, lambda: rng.exponential(0.1))

    # %% Display simulation
    plt.figure(1, clear=True)
    plt.plot(t, a, drawstyle='steps-post')
    plt.plot(t, p, 'o')
    plt.plot(t, m, '--', drawstyle='steps-pre')
    plt.legend(['$A_i$', '$P_i$', '$M_i$'])
    plt.xlabel('$T_i$')

    # %% Simulate 100 realisations
    n = 100
    N = 1000
    T = np.zeros((n, N))
    P = np.zeros((n, N))
    A = np.zeros((n, N))
    M = np.zeros((n, N))

    for i in range(n):
        t, p, a, m = simulate(rng, N, 0, 1, 1, lambda: rng.normal(0, 1))
        T[i, :] = t
        P[i, :] = p
        A[i, :] = a
        M[i, :] = m

    # %% Display simulation
    plt.figure(2, clear=True)
    plt.hist(P[:, -1])
    plt.title(f'$P_{{{N}}}$')
    plt.figure(3, clear=True)
    plt.hist(A[:, -1])
    plt.title(f'$A_{{{N}}}$')
    plt.figure(4, clear=True)
    plt.hist(M[:, -1])
    plt.title(f'$M_{{{N}}}$')
