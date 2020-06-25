import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def simulate(rng, N, mu, sigma, lam, y_dist, limit=math.inf):
    phi1 = math.sqrt(mu ** 2 / sigma ** 4 + 2 * lam / sigma ** 2) - mu / sigma ** 2
    phi2 = math.sqrt(mu ** 2 / sigma ** 4 + 2 * lam / sigma ** 2) + mu / sigma ** 2

    t = np.zeros(N)
    p = np.zeros(N)
    v = np.zeros(N)
    w = np.zeros(N)
    a = np.zeros(N)
    y = np.zeros(N)
    m = np.zeros(N)

    i = 0
    for i in range(1, N):
        # Advance time
        dt = rng.exponential(1 / lam)
        t[i] = t[i - 1] + dt

        # Step 1
        v[i] = rng.exponential(1 / phi1)
        w[i] = rng.exponential(1 / phi2)

        # Step 2
        y[i] = y_dist(rng)

        # Step 3
        p[i] = a[i - 1] + (v[i] - w[i])
        a[i] = p[i] + y[i]
        m[i] = max(m[i - 1], a[i - 1] + v[i], a[i])

        if m[i] > limit:
            break

    return i, t, p, a, m, v, w, y


def printSimulate(N, mu, sigma, lam, y_dist, n, limit=math.inf, seed=3017, start_string=""):
    rng = np.random.default_rng(seed)
    T = np.zeros((n, N))
    P = np.zeros((n, N))
    A = np.zeros((n, N))
    M = np.zeros((n, N))
    W = np.zeros((n, N))
    V = np.zeros((n, N))
    Y = np.zeros((n, N))

    for i in range(n):
        _, t, p, a, m, v, w, y = simulate(rng, N, mu, sigma, lam, y_dist, limit)
        T[i, :] = t
        P[i, :] = p
        A[i, :] = a
        M[i, :] = m
        W[i, :] = w
        V[i, :] = v
        Y[i, :] = y
    # plt.figure()
    # for i in range(n):
    #     plt.plot(T[i, :], V[i, :])
    # plt.title(f"v of $\sigma$ = {sigma}")
    # plt.figure()
    # for i in range(n):
    #     plt.plot(T[i, :], W[i, :])
    # plt.title(f"w of $\sigma$ = {sigma}")
    # plt.figure()
    # for i in range(n):
    #     plt.plot(T[i, :], Y[i, :])
    # plt.title(f"Y of $\sigma$ = {sigma}")
    plt.figure()
    for i in range(n):
        plt.plot(T[i, :], A[i, :])
    plt.title(f"$\\mu = {mu}, \\sigma = {sigma}, \\lambda = {lam} $")
    plt.savefig(f"{start_string}_{mu}_{sigma}_{lam}.eps")


if __name__ == "__main__":

    # %% Demo a single simulation
    rng = np.random.default_rng(3017)
    _, t, p, a, m, _, _, _ = simulate(
        rng, 1000, 1, 1, 1, lambda rng: rng.exponential(0.1)
    )

    # %% Display simulation
    # plt.figure(1, clear=True)
    # plt.plot(t, a, drawstyle="steps-post")
    # plt.plot(t, p, "o")
    # plt.plot(t, m, "--", drawstyle="steps-pre")
    # plt.legend(["$A_i$", "$P_i$", "$M_i$"])
    # plt.xlabel("$T_i$")

    # %% Simulate 100 realisations
    rng = np.random.default_rng(3017)
    n = 200
    N = 1000
    T = np.zeros((n, N))
    P = np.zeros((n, N))
    A = np.zeros((n, N))
    M = np.zeros((n, N))

    for i in range(n):
        # _, t, p, a, m = simulate(rng, N, 0, 1, 1, lambda rng: rng.normal(0, 1))
        _, t, p, a, m, _, _, _ = simulate(
            rng, N, 0, 1, 1, lambda rng: rng.exponential(0.1)
        )
        T[i, :] = t
        P[i, :] = p
        A[i, :] = a
        M[i, :] = m
    Tl = T[:, -1]
    Pl = P[:, -1]
    Al = A[:, -1]
    Ml = M[:, -1]

    # %% Display simulation
    plt.figure(1, clear=True)
    plt.hist(Pl)
    print(st.kstest(Pl, "norm"))
    plt.title(
        f"$P_{{{N}}}, p = {st.kstest((Pl-np.mean(Pl))/np.std(Pl), 'norm')[1]:.4}$"
    )
    plt.figure(2, clear=True)
    plt.hist(Tl)
    plt.title(
        f"$T_{{{N}}}, p = {st.kstest((Tl-np.mean(Tl))/np.std(Tl), 'norm')[1]:.4}$"
    )
    plt.figure(3, clear=True)
    plt.hist(Al)
    plt.title(
        f"$A_{{{N}}}, p = {st.kstest((Al-np.mean(Al))/np.std(Al), 'norm')[1]:.4}$"
    )
    plt.figure(4, clear=True)
    plt.hist(Ml)
    plt.title(
        f"$M_{{{N}}}, p = {st.kstest((Ml-np.mean(Ml))/np.std(Ml), 'norm')[1]:.4}$"
    )
    f = lambda rng: rng.exponential(0.1)
    f0 = lambda rng: rng.exponential(1) * 0
    n = 20
    N = 1000
    ex3 = "secondproject/ex3"
    printSimulate(N, 0, 1, 1, f, n, start_string=ex3)
    printSimulate(N, -1, 1, 1, f, n, start_string=ex3)
    printSimulate(N, 1, 1, 1, f, n, start_string=ex3)
    printSimulate(N, 0, 0.1, 1, f, n, start_string=ex3)
    printSimulate(N, 0, 10, 1, f, n, start_string=ex3)
    printSimulate(N, 0, 1, 0.1, f, n, start_string=ex3)
    printSimulate(N, 0, 1, 10, f, n, start_string=ex3)
    printSimulate(N, -0.1, 1, 1, f, n, start_string=ex3)
    printSimulate(N, -1, np.sqrt(10), 10, f, n, start_string=ex3)
    printSimulate(N, -1, 1, 1, f0, n, start_string=ex3 + "y0")
    plt.show()
