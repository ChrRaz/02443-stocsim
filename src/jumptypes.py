#!/usr/bin/python3
import math

import matplotlib.pyplot as plt
import numpy as np


def hyperexp(rng, p, l1, l2):
    u = rng.uniform()
    if u < p:
        return rng.exponential(1 / l1)
    else:
        return rng.exponential(1 / l2)


def simulate(rng1, rng2, N, mu, sigma, lam, y_dist):
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
        dt = rng1.exponential(1 / lam)
        t[i] = t[i - 1] + dt

        # Step 1
        v[i] = rng1.exponential(1 / phi1)
        w[i] = rng1.exponential(1 / phi2)

        # Step 2
        y[i] = y_dist(rng2)

        # Step 3
        p[i] = a[i - 1] + (v[i] - w[i])
        a[i] = p[i] + y[i]
        m[i] = max(m[i - 1], a[i - 1] + v[i], a[i])

    return t, p, a, m


if __name__ == '__main__':
    N, ntrials = 400, 20
    mu, sig, lam = -1, 1, 1

    jumps = [
        ('exp', 'Exp(1)', lambda rng: rng.exponential()),
        ('erl2', 'Erlang_2(2)', lambda rng: rng.gamma(2, 1 / 2)),
        ('erl3', 'Erlang_3(3)', lambda rng: rng.gamma(3, 1 / 3)),
        ('hyper', '0.8 * Exp(0.8333) + 0.2 * Exp(5)', lambda rng: hyperexp(rng, 0.8, 0.8333, 5)),
        ('pareto1', 'Pareto(1.05)', lambda rng: (rng.pareto(1.05) + 1) / (1.05 / 0.05)),
        ('pareto2', 'Pareto(2.05)', lambda rng: (rng.pareto(2.05) + 1) / (2.05 / 1.05)),
        ('unif', 'U(0,2)', lambda rng: rng.uniform(0, 2)),
    ]

    seed = 547225744

    # %% Base
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1)

    plt.figure(1, clear=True)

    T = np.zeros((ntrials, N))
    A = np.zeros((ntrials, N))
    M = np.zeros((ntrials, N))

    for j in range(ntrials):
        t, p, a, m = simulate(rng1, rng2, N, 0, sig, lam, lambda rng: 0)
        T[j, :] = t
        A[j, :] = a
        M[j, :] = m

    plt.plot(T.T, A.T)
    plt.xlabel('$T_i$')
    plt.title('$\\mu = 0, Y_i = 0$')
    plt.savefig(f'../figs/ex4/base.eps')

    # %% Comparisons
    for i, (name, title, y) in enumerate(jumps):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed + 1)

        plt.figure(i + 2, clear=True)

        T = np.zeros((ntrials, N))
        A = np.zeros((ntrials, N))
        M = np.zeros((ntrials, N))

        for j in range(ntrials):
            t, p, a, m = simulate(rng1, rng2, N, mu, sig, lam, y)
            T[j, :] = t
            A[j, :] = a
            M[j, :] = m

        plt.plot(T.T, A.T)
        plt.xlabel('$T_i$')
        plt.title(title)
        plt.savefig(f'../figs/ex4/{name}.eps')
