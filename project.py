# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:47:53 2018

@author: User
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

DJI = pd.read_csv('C:/Users/User/Desktop/FINA4380/Project/CSV/INDU Index.csv',index_col=0)

#training ~ 70%; test ~ 30%
traindays = round(len(DJI)*0.7)

#find cointegrated pairs
n = DJI.shape[1]
keys = DJI.keys()
pairs = []
for i in range(n):
    for j in range(i+1, n):
        S1 = DJI[keys[i]].iloc[:traindays]
        S2 = DJI[keys[j]].iloc[:traindays]
        if np.count_nonzero(~np.isnan(S1)) and np.count_nonzero(~np.isnan(S2)) == traindays:
            S1 = sm.add_constant(S1)
            results = sm.OLS(S2, S1, missing = 'drop').fit()
            S1 = S1[S1.keys()[1]]
            b = results.params[1]
            Z = S2 - b * S1
            pvalue = adfuller(Z.dropna())[1]
            if pvalue < 0.01:
                pairs.append((keys[i], keys[j]))
print(pairs)

Z = S2 - b * S1
train = Z[:traindays]
test = Z[traindays:]

Z_mavg5 = train.rolling(window = 5, center = False).mean()
Z_mavg60 = train.rolling(window = 60, center = False).mean()
std_60 = train.rolling(window = 60, center = False).std()

zscore_60_5 = (Z_mavg5 - Z_mavg60) / std_60
zscore_60_5.plot()

#if zscore < -1, long S2 & short S1
