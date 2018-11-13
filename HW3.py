# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:25:20 2018

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm

#Q1&2
Indices = pd.read_csv('C:/Users/User/Desktop/FINA4380/HW3/Indices.csv',index_col=0)
Factors = pd.read_csv('C:/Users/User/Desktop/FINA4380/HW3/Factors.csv',index_col=0)
Indices.index = pd.to_datetime(Indices.index)
Factors.index = pd.to_datetime(Factors.index)

#Q3
Factors_ret = np.log(Factors/Factors.shift())
Factors_ret = Factors_ret.drop(Factors_ret.index[0])
pca = PCA(n_components = 3)
pca.fit(Factors_ret)
PCA(copy=True, iterated_power='auto', n_components=3, random_state=None, svd_solver='auto', tol=0.0, whiten=False)

#Q4
Indices_ret = np.log(Indices/Indices.shift())
Indices_ret = Indices_ret.drop(Indices_ret.index[0])
F1 = np.matmul(Factors_ret, pca.components_[:1,:].T)
F2 = np.matmul(Factors_ret, pca.components_[1:2,:].T)
F3 = np.matmul(Factors_ret, pca.components_[2:3,:].T)
X = sm.add_constant(np.column_stack((np.column_stack((F1, F2)), F3)))
parameters = sm.OLS(Indices_ret,X).fit().params
pd.DataFrame(parameters).to_csv('C:/Users/User/Desktop/FINA4380/beta.csv', header = Indices_ret.columns.values)

#Q5&6
FF3 = pd.read_csv('C:/Users/User/Desktop/FINA4380/HW3/Japan_3_Factors_Daily.csv',index_col=0)
TOPIX30 = pd.read_csv('C:/Users/User/Desktop/FINA4380/HW3/Topix30.csv',index_col=0)

#Q7
TOPIX30_ret = np.log((TOPIX30/TOPIX30.shift()))
TOPIX30_ret = TOPIX30_ret.drop(TOPIX30_ret.index[0])

pvalues_list = []
for i in range(0,30):
    pvalues = sm.OLS(TOPIX30_ret.iloc[:,i:i+1].values,sm.add_constant(FF3.iloc[:,:1].values)).fit().pvalues
    pvalues_list.append(np.array(pvalues))
pd.DataFrame(pvalues_list).to_csv('C:/Users/User/Desktop/FINA4380/pvalues_list.csv')

pvalues2_list = []
for i in range(0,30):
    pvalues2 = sm.OLS(TOPIX30_ret.iloc[:,i:i+1].values,sm.add_constant(np.column_stack((np.column_stack((FF3.iloc[:,:1].values, FF3.iloc[:,1:2])), FF3.iloc[:,2:3])))).fit().pvalues
    pvalues2_list.append(np.array(pvalues2))
pd.DataFrame(pvalues2_list).to_csv('C:/Users/User/Desktop/FINA4380/pvalues2_list.csv')
