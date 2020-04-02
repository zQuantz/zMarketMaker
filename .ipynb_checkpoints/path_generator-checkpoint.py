import pandas as pd
import numpy as np
import sys, os

df = pd.read_csv("data/cooccurrence_matrix.csv", index_col=0)

ticks = df.index.values
dist = df.values
sums = dist.sum(axis=1)
dist = df.values

NUM_PATHS = 10_000
LENGTH = 500
paths = []

for j in range(NUM_PATHS):
    print("Simulating Path", j)
    path = [0]
    for i in range(LENGTH):
        idx = path[-1] + 5
        ps = dist[idx, :] / sums[idx]
        path.append(np.random.choice(ticks, p=ps))
    paths.append(path)
    
paths = np.array(paths)
np.save("data/sample_paths.npy", paths)