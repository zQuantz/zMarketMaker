import pandas as pd
import numpy as np
import sys, os

import matplotlib.pyplot as plt

TICK_SIZE = 0.125

## Load data
df = pd.read_csv("data/ibm.csv")
print(df.head())

df['delta'] = df.price.diff()
df['mid'] = (df.ask + df.bid) / 2
df['mid_delta'] = df.mid.diff()
df = df.iloc[1:]

## Represent 
df['ticks'] = (df.delta / TICK_SIZE).astype(int)
df['tick_smooth'] = df.ticks

## Smooth out the return series by only allowing for price jumps in the range of (-5, 5) ticks.
to_smooth = df.ticks[abs(df.ticks) > 5]
new_jumps = np.random.choice([3, 4, 5], size = len(to_smooth),
                              p=[0.6703717876, 0.1793026889, 0.1503255235])
jump_signs = np.sign(to_smooth)

## Replace them in the data
df.loc[to_smooth.index, 'tick_smooth'] = new_jumps * jump_signs

## Scale down the volume by 10
df['volume_scaled'] = (df.volume / 10).astype(int)

## Save the transformed data
df = df[['tick_smooth', 'volume_scaled']]
df.columns = ['change', 'volume']
df.to_csv("data/ibm_t.csv", index=False)

if False:
    plt.figure(figsize=(20, 14))
    plt.title("Cumsum of Smoothed Tick Changes")
    plt.plot(df.change.cumsum())
    plt.show()