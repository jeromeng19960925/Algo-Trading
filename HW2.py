# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:01:39 2018

@author: User
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Q1
df = pd.read_csv('C:/Users/User/Desktop/FINA4380/HW2/FINA 4380 HW2 dataset.csv',index_col=0)

#Q2
df.index = pd.to_datetime(df.index)
df_m = df.resample('M').last()
df_sr = df_m/df_m.shift()


#Q3
df_sr2 = df_sr.drop(df_sr.index[0])
geomeanlist = []
for i in range(0,11):
    ticker = df_sr2.iloc[:,i:i+1]
    geomean = ticker.prod()**(1.0/len(ticker)) - 1
    geomeanlist.append(geomean.values)
sr_cov = (df_sr2 - 1).cov()
I = np.ones((1, 11))
J = np.ones((11, 1))
A = np.matmul(np.matmul(np.transpose(geomeanlist), np.linalg.inv(sr_cov)), geomeanlist)
B = np.matmul(np.matmul(I, np.linalg.inv(sr_cov)), geomeanlist)
C = np.matmul(np.matmul(I, np.linalg.inv(sr_cov)), np.transpose(I))

#Q4
lambdalist = []
gammalist = []
stdlist = []
weightingslist = []
erlist = []
for j in range(5, 105, 5):
    lambdaa = ((j/1000)*C - B) / (A*C - B**2)
    gamma = (A - B*(j/1000)) / (A*C - B**2)
    lambdalist.append(lambdaa)
    gammalist.append(gamma) 
    weightings = np.matmul(np.linalg.inv(sr_cov), geomeanlist)*lambdaa + np.matmul(np.linalg.inv(sr_cov), J)*gamma
    std = np.asscalar(np.matmul(np.matmul(np.transpose(weightings), sr_cov), weightings)**0.5)
    weightingslist.append(weightings)
    stdlist.append(std)
    erlist.append(j/1000)

#Q5
rp_weightings_list = []
rp_std_list = []
rf_weightings_list = []
for j in range(5, 105, 5):
    delta = (j/1000 - 0.005) / (np.matmul(np.matmul(np.transpose(geomeanlist - J*0.005), np.linalg.inv(sr_cov)), geomeanlist - J*0.005))
    rp_weightings = delta*np.matmul(np.linalg.inv(sr_cov), geomeanlist - J*0.005)
    rp_weightings_list.append(rp_weightings)
    rp_std = np.asscalar(np.matmul(np.matmul(np.transpose(rp_weightings), sr_cov), rp_weightings)**0.5)
    rp_std_list.append(rp_std)
plt.plot(stdlist, erlist, c = 'red')
plt.plot(rp_std_list, erlist, c = 'blue')
distance = np.linspace(min(stdlist),max(stdlist))
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.xticks(fontsize = '10')
plt.yticks(fontsize = '10')

#Q6
df2 = pd.read_csv('C:/Users/User/Desktop/FINA4380/HW2/FINA 4380 HW2 dataset 2.csv',index_col=0)
df2_gm_list = []
for l in range(0,11):
    targetprice  = np.asscalar(df2.iloc[l:l+1,0:1].values)
    lastprice = np.asscalar(df.iloc[0:1,l:l+1].values)
    df2_gm = pd.Series((targetprice / lastprice)**(1 / 12) - 1)
    df2_gm_list.append(df2_gm.values)
D = np.matmul(np.matmul(np.transpose(df2_gm_list), np.linalg.inv(sr_cov)), df2_gm_list)
E = np.matmul(np.matmul(I, np.linalg.inv(sr_cov)), df2_gm_list)
F = np.matmul(np.matmul(I, np.linalg.inv(sr_cov)), np.transpose(I))
lambdalist2 = []
gammalist2 = []
stdlist2 = []
weightingslist2 = []
erlist2 = []
for j in range(5, 105, 5):
    lambdaa = ((j/1000)*F - E) / (D*F - E**2)
    gamma = (D - E*(j/1000)) / (D*F - E**2)
    lambdalist2.append(lambdaa)
    gammalist2.append(gamma) 
    weightings = np.matmul(np.linalg.inv(sr_cov), df2_gm_list)*lambdaa + np.matmul(np.linalg.inv(sr_cov), J)*gamma
    std = np.asscalar(np.matmul(np.matmul(np.transpose(weightings), sr_cov), weightings)**0.5)
    weightingslist2.append(weightings)
    stdlist2.append(std)
    erlist2.append(j/1000)
rp_weightings_list2 = []
rp_std_list2 = []
rf_weightings_list2 = []
for j in range(5, 105, 5):
    delta = (j/1000 - 0.005) / (np.matmul(np.matmul(np.transpose(df2_gm_list - J*0.005), np.linalg.inv(sr_cov)), df2_gm_list - J*0.005))
    rp_weightings = delta*np.matmul(np.linalg.inv(sr_cov), df2_gm_list - J*0.005)
    rp_weightings_list2.append(rp_weightings)
    rp_std = np.asscalar((np.matmul(np.matmul(np.transpose(rp_weightings), sr_cov), rp_weightings)**0.5))
    rp_std_list2.append(rp_std)
plt.plot(stdlist2, erlist2, c = 'red')
plt.plot(rp_std_list2, erlist2, c = 'blue')
distance = np.linspace(min(stdlist2),max(stdlist2))
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.xticks(fontsize = '10')
plt.yticks(fontsize = '10')