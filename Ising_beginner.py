# 2D Ising model (beginner version)
# Idea: a grid of tiny magnets. Each one points up (+1) or down (-1).
# Neighbours prefer to point the same way. Heat (temperature T) makes them flip randomly.
# Low T  -> most point the same way (magnetized).
# High T -> random, mixed (not magnetized).

import random
import math
import numpy as np
import matplotlib.pyplot as plt

N = 10  # grid is N x N

def make_grid():
    # start with every spin randomly +1 or -1
    grid = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            grid[i][j] = random.choice([1, -1])
    return grid

def neighbour_sum(grid, i, j):
    # add up the 4 neighbours (up, down, left, right)
    # "% N" wraps around the edges, so the grid has no boundary
    up = grid[(i - 1) % N][j]
    down = grid[(i + 1) % N][j]
    left = grid[i][(j - 1) % N]
    right = grid[i][(j + 1) % N]
    return up + down + left + right

def one_sweep(grid, T):
    # try to flip N*N randomly chosen spins
    for step in range(N * N):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)

        # energy change if we flip this spin
        dE = 2 * grid[i][j] * neighbour_sum(grid, i, j)

        # Metropolis rule:
        # always flip if it lowers energy,
        # otherwise flip with probability exp(-dE / T)
        if dE <= 0:
            grid[i][j] = -grid[i][j]
        elif random.random() < math.exp(-dE / T):
            grid[i][j] = -grid[i][j]

def magnetization(grid):
    # average spin, ignoring direction (0 = mixed, 1 = all aligned)
    return abs(grid.sum()) / (N * N)

# ---- main program ----
temperatures = [1.0, 1.5, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0]
results = []

for T in temperatures:
    grid = make_grid()
    for sweep in range(200):          # let it settle
        one_sweep(grid, T)
    total = 0
    for sweep in range(200):          # now measure
        one_sweep(grid, T)
        total = total + magnetization(grid)
    results.append(total / 200)
    print("T =", T, " magnetization =", round(total / 200, 2))

plt.plot(temperatures, results, "o-")
plt.axvline(2.269, color="red", linestyle="--", label="exact Tc = 2.269")
plt.xlabel("Temperature T")
plt.ylabel("Magnetization")
plt.legend()
plt.savefig("ising_beginner.png")
plt.show()
