from const import TICK_LIMIT
import pandas as pd
import numpy as np
import sys, os

import matplotlib.pyplot as plt

###################################################################################################

TICK_SIZE = 0.125
TICK_REDIST = [2, 3, 4]
TICK_REDIST_P = [0.8703717876, 0.0793026889, 0.0503255235]

###################################################################################################

## Load data
df = pd.read_csv("data/ibm.csv")
print("\nBefore")
print(df.head())

df['delta'] = df.price.diff()
df['mid'] = (df.ask + df.bid) / 2
df['mid_delta'] = df.mid.diff()
df = df.iloc[1:]

## Represent 
df['ticks'] = (df.delta / TICK_SIZE).astype(int)
df['tick_smooth'] = df.ticks

## Smooth out the return series by only allowing for price jumps in the range of (-5, 5) ticks.
to_smooth = df.ticks[abs(df.ticks) > TICK_LIMIT]
new_jumps = np.random.choice(TICK_REDIST, size = len(to_smooth), p=TICK_REDIST_P)
jump_signs = np.sign(to_smooth)

## Replace them in the data
df.loc[to_smooth.index, 'tick_smooth'] = new_jumps * jump_signs

## Scale down the volume by 10
df['volume_scaled'] = (df.volume / 10).astype(int)

###################################################################################################

## Save the transformed data
df = df[['tick_smooth', 'volume_scaled']]
df.columns = ['change', 'volume']
df.to_csv("data/ibm_t.csv", index=False)
print("\nAfter")
print(df.head())

if False:
    plt.figure(figsize=(20, 14))
    plt.title("Cumsum of Smoothed Tick Changes")
    plt.plot(df.change.cumsum())
    plt.show()