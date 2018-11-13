# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:54:12 2018

@author: User
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize 

Stocks = pd.read_csv('C:/Users/User/Desktop/FINA4380/HW4/Stocks.csv',index_col=0)
Stocks.index = pd.to_datetime(Stocks.index)
Stocks_m = Stocks.resample('M').last()
Stocks_sr = np.log(Stocks_m/Stocks_m.shift())
Stocks_sr = Stocks_sr.drop(Stocks_sr.index[0])
    
#Q1 
def risk_parity_function(weights, cov, stocks_sd_list):
    combined_sd = np.dot(weights, stocks_sd_list)
    port_sd = np.dot(np.dot(weights, cov), weights)**(1/2)
    return(- combined_sd / port_sd)
    
def risk_parity(data, long=1):
    cov = data.cov()
    stocks_sd_list = []
    for i in range(0,11):
        stocks_sd = np.std(data.iloc[:,i:i+1])
        stocks_sd_list.append(np.array(stocks_sd))
    n = cov.shape[0]
    weights = np.ones(n) / n
    cons = ({'type': 'eq', 'fun': lambda x:1-sum(x)})
    bnds = [(0,0.5) for i in weights]
    if long == 1:
        res = minimize(risk_parity_function, x0 = weights, args = (cov, stocks_sd_list), method = 'SLSQP', constraints = cons, bounds = bnds, tol = 1e-30)
    else:
        res = minimize(risk_parity_function, x0 = weights, args = (cov, stocks_sd_list), method = 'SLSQP', constraints = cons, tol = 1e-30)
    return res.x

print(risk_parity(Stocks_sr))

#Q2
def risk_parity_function2(weights, cov):
    return(np.dot(np.dot(weights, cov), weights))
    
def risk_parity2(data, long=1):
    cov = data.cov()
    n = cov.shape[0]
    weights = np.ones(n) / n
    cons = ({'type': 'eq', 'fun': lambda x:1-sum(x)})
    bnds = [(0,0.5) for i in weights]
    if long == 1:
        res = minimize(risk_parity_function2, x0 = weights, args = (cov), method = 'SLSQP', constraints = cons, bounds = bnds, tol = 1e-30)
    else:
        res = minimize(risk_parity_function2, x0 = weights, args = (cov), method = 'SLSQP', constraints = cons, tol = 1e-30)
    return res.x

print(risk_parity2(Stocks_sr))

#Q3
def risk_parity_function3(weights, cov):
    stocks_avgret_list = []
    for i in range(0,11):
        stocks_avgret = np.mean(Stocks_sr.iloc[:,i:i+1])
        stocks_avgret_list.append(np.array(stocks_avgret))
    return(- (np.dot(weights, stocks_avgret_list) - 0.005) / np.dot(np.dot(weights, cov), weights)**(1/2))
    
def risk_parity3(data, long=1):
    cov = data.cov()
    n = cov.shape[0]
    weights = np.ones(n) / n
    cons = ({'type': 'eq', 'fun': lambda x:1-sum(x)})
    bnds = [(0,0.5) for i in weights]
    if long == 1:
        res = minimize(risk_parity_function3, x0 = weights, args = (cov), method = 'SLSQP', constraints = cons, bounds = bnds, tol = 1e-30)
    else:
        res = minimize(risk_parity_function3, x0 = weights, args = (cov), method = 'SLSQP', constraints = cons, tol = 1e-30)
    return res.x

print(risk_parity3(Stocks_sr))

df = np.column_stack((np.column_stack((risk_parity(Stocks_sr), risk_parity2(Stocks_sr))), risk_parity3(Stocks_sr)))

pd.DataFrame(df).to_csv('C:/Users/User/Desktop/FINA4380/HW4/df.csv')


















