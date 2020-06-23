#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

import levy


def hyperexp(rng, p, l1, l2):
    u = rng.uniform()
    if u < p:
        return rng.exponential(1 / l1)
    else:
        return rng.exponential(1 / l2)


if __name__ == '__main__':
    N, ntrials = 400, 20
    mu, sig, lam = -1, 1, 1

    jumps = [
        (333, 'exp', 'Exp(1)', lambda: rng.exponential()),
        (334, 'erl2', 'Erlang_2(1/2)', lambda: rng.gamma(2, 1 / 2)),
        (353, 'erl3', 'Erlang_3(1/3)', lambda: rng.gamma(3, 1 / 3)),
        (433, 'hyper', '0.8 * Exp(0.8333) + 0.2 * Exp(5)', lambda: hyperexp(rng, 0.8, 0.8333, 5)),
        (383, 'pareto', 'Pareto(2.05)', lambda: rng.pareto(2.05) / (2.05 / 1.05)),
        (343, 'y0', 'Constant 0', lambda: 0),
    ]

    for i, (seed, name, title, y) in enumerate(jumps):
        rng = np.random.default_rng(seed)

        plt.figure(i + 1, clear=True)

        T = np.zeros((ntrials, N))
        A = np.zeros((ntrials, N))
        M = np.zeros((ntrials, N))

        for j in range(ntrials):
            t, p, a, m = levy.simulate(rng, N, mu, sig, lam, y)
            T[j, :] = t
            A[j, :] = a
            M[j, :] = m

        plt.plot(T.T, A.T)
        plt.xlabel('$T_i$')
        plt.title(title)
        plt.savefig(f'../figs/ex4/{name}.eps')
