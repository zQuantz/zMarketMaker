from const import TICK_LIMIT
import pandas as pd

df = pd.read_csv("data/ibm_t.csv")

cov = pd.DataFrame()
cov['v1'] = df.change
cov['v2'] = df.change.shift()
cov = cov.iloc[1:, :].astype(int)

vals = [i-TICK_LIMIT for i in range(0, 2 * TICK_LIMIT + 1)]
coocc = pd.DataFrame(columns = vals, index = vals)
coocc.loc[:, :] = 0

cov['comp'] = '(' + cov.v1.astype(str) + ',' + cov.v2.astype(str) + ')'
cov = cov.comp.value_counts()
    
for idx, val in zip(cov.index, cov.values):
    idx = eval(idx)
    coocc.loc[idx[0], idx[1]] = val
    
print(coocc)
coocc.to_csv("data/cooccurrence_matrix.csv")