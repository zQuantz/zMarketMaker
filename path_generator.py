from const import TICKS, TICK_LIMIT
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, os

###################################################################################################

coocc = pd.read_csv("data/cooccurrence_matrix.csv", index_col=0)
ticks = coocc.columns.astype(int).tolist()

coocc = coocc.values
coocc = (coocc.T / coocc.sum(axis=1)).T

NUM_PATHS = 10_000
LENGTH = 500
paths = []

###################################################################################################

for j in range(NUM_PATHS):
    print("Simulating Path", j)
    path = [0]
    for i in range(LENGTH):
        idx = path[-1] + TICK_LIMIT
        ps = coocc[idx]
        path.append(np.random.choice(TICKS, p=ps))
    paths.append(path)
    
paths = np.array(paths)
np.save("paths/sample_paths.npy", paths)

###################################################################################################

try:
	os.mkdir(f"paths/plots")
except Exception as e:
	print(e)

### Instance Selector
for length in [50, 100, 200]:

	try:
		os.mkdir(f"paths/{length}")
	except Exception as e:
		print(e)

	paths_ = paths[:, :length]

	## Trending Paths
	trends = np.sum(paths_, axis=1)
	idc = np.argsort(trends)

	uptrend = paths_[idc[-1]]
	downtrend = paths_[idc[0]]
	no_change = paths_[idc[int(len(idc)/2)]]

	## Volatile Paths
	volatility = np.std(paths_, axis=1)
	idc = np.argsort(volatility)

	high_vol = paths_[idc[-1]]
	low_vol = paths_[idc[0]]
	avg_vol = paths_[idc[int(len(idc)/2)]]

	np.save(f"paths/{length}/uptrend.npy", uptrend)
	np.save(f"paths/{length}/downtrend.npy", downtrend)
	np.save(f"paths/{length}/no_change.npy", no_change)

	np.save(f"paths/{length}/high_vol.npy", high_vol)
	np.save(f"paths/{length}/low_vol.npy", low_vol)
	np.save(f"paths/{length}/avg_vol.npy", avg_vol)

	f, ax = plt.subplots(2, 3, figsize=(8,8), sharex=True, sharey=True)
	ax[0,0].plot(np.cumsum(uptrend)) ; ax[0,0].set_title("Uptrend")
	ax[0,1].plot(np.cumsum(downtrend)) ; ax[0,1].set_title("Downtrend")
	ax[0,2].plot(np.cumsum(no_change)) ; ax[0,2].set_title("No Change")
	ax[1,0].plot(np.cumsum(high_vol)) ; ax[1,0].set_title("High Volatility")
	ax[1,1].plot(np.cumsum(low_vol)) ; ax[1,1].set_title("Low Volatility")
	ax[1,2].plot(np.cumsum(avg_vol)) ; ax[1,2].set_title("Average Volatility")

	for row in ax:
		for ax_ in row:
			ax_.set_ylabel("Price")
			break
	f.suptitle(f"Instance Cummulative Tick Change by Time - {length} Periods")
	f.savefig(f"paths/plots/{length}.png")

###################################################################################################
