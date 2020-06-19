#!/usr/bin/python3
import numpy as np

from levy import simulate

if __name__ == '__main__':
    rng = np.random.default_rng()

    nupcrossed = 0
    ntrials = 500
    for i in range(ntrials):
        N = 1000
        j, t, p, a, m = simulate(rng, N, 0, 1, 1, lambda: rng.exponential(0.1), 100)
        # print(j)
        if j != N-1:
            nupcrossed += 1

    print(f'{nupcrossed/ntrials = }')
