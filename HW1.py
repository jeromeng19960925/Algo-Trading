from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm, normaltest

#Q1&2
df = pd.read_csv('C:/Users/User/Desktop/FINA4380/HW1/4380 hw1 data.csv',index_col=0)

#Q3
df.index = pd.to_datetime(df.index)
df_w = df.resample('W-FRI').last()
df_m = df.resample('M').last()
df_lr = np.log(df/df.shift())
df_wlr = np.log(df_w/df_w.shift())
df_mlr = np.log(df_m/df_m.shift())

#Q4
HSI_cov = df_wlr.cov()
HSI_cov.to_csv('C:/Users/User/Desktop/FINA4380/HW1/HSI_cov.csv', index=True)

#Q5
df_lr2 = df_lr.drop(df_lr.index[0])
mu = np.average(df_lr2['700 HK Equity'])
sig = np.std(df_lr2['700 HK Equity'])
plt.hist(df_lr2['700 HK Equity'], normed=True, bins=100)
distance = np.linspace(min(df_lr2['700 HK Equity']),max(df_lr2['700 HK Equity']))
plt.hold(True)
plt.plot(distance, norm.pdf(distance,mu,sig))
plt.xlabel('log return')
plt.ylabel('density')
plt.title('700 HK')
plt.xticks(fontsize = '10')
plt.yticks(fontsize = '10')

#Q6
df_wlr.index = pd.to_datetime(df_wlr.index)
df_wlr2 = df_wlr.drop(df_wlr.index[0])
fig = plt.figure(figsize = (15, 10))
x = normaltest(df_wlr2).pvalue
x = x[~np.isnan(x)]
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.subplot(321)
plt.hist(x, normed=True, bins=100)
distance = np.linspace(min(x),max(x))
plt.hold(True)
plt.xlabel('p-value', fontsize = '10', fontweight="bold")
plt.ylabel('density', fontsize = '10', fontweight="bold")
plt.title('HSI Constituents Normaltest P-value', fontsize = '10', fontweight="bold")
plt.xticks(fontsize = '10', fontweight="bold")
plt.yticks(fontsize = '10', fontweight="bold")
plt.text(-0.2, -10, 'Since most pvalues are very small, most constituents log return are not normal', fontsize = '10')

#Q7
from statsmodels.stats.diagnostic import acorr_ljungbox
z = []
tickers = list(df_wlr2.columns)
for i in tickers:
    tickersi = df_wlr2[i]
    n = acorr_ljungbox(tickersi, lags = 5)
    z.append(n[1])
z = np.transpose(z)
z1 = z[:1,:]
z2 = z[1:2,:]
z3 = z[2:3,:]
z4 = z[3:4,:]
z5 = z[4:5,:]
z6 = z1[~np.isnan(z1)]
z7 = z2[~np.isnan(z2)]
z8 = z3[~np.isnan(z3)]
z9 = z4[~np.isnan(z4)]
z10 = z5[~np.isnan(z5)]
plt.subplot(322)
plt.hist(z6, normed=True, bins=100)
distance = np.linspace(min(z6),max(z6))
plt.hold(True)
plt.xlabel('p-value', fontsize = '10', fontweight="bold")
plt.ylabel('density', fontsize = '10', fontweight="bold")
plt.title('HSI Constituents Autocorrelation Test P-value (lag=1)', fontsize = '10', fontweight="bold")
plt.xticks(fontsize = '10', fontweight="bold")
plt.yticks(fontsize = '10', fontweight="bold")
plt.text(-0.2, -2, 'Since most pvalues are very small, most constituents log return are correlated', fontsize = '10')
plt.subplot(323)
plt.hist(z7, normed=True, bins=100)
distance = np.linspace(min(z7),max(z7))
plt.hold(True)
plt.xlabel('p-value', fontsize = '10', fontweight="bold")
plt.ylabel('density', fontsize = '10', fontweight="bold")
plt.title('HSI Constituents Autocorrelation Test P-value (lag=2)', fontsize = '10', fontweight="bold")
plt.xticks(fontsize = '10', fontweight="bold")
plt.yticks(fontsize = '10', fontweight="bold")
plt.text(-0.2, -2, 'Since most pvalues are very small, most constituents log return are correlated', fontsize = '10')
plt.subplot(324)
plt.hist(z8, normed=True, bins=100)
distance = np.linspace(min(z8),max(z8))
plt.hold(True)
plt.xlabel('p-value', fontsize = '10', fontweight="bold")
plt.ylabel('density', fontsize = '10', fontweight="bold")
plt.title('HSI Constituents Autocorrelation Test P-value (lag=3)', fontsize = '10', fontweight="bold")
plt.xticks(fontsize = '10', fontweight="bold")
plt.yticks(fontsize = '10', fontweight="bold")
plt.text(-0.2, -2, 'Since most pvalues are very small, most constituents log return are correlated', fontsize = '10')
plt.subplot(325)
plt.hist(z9, normed=True, bins=100)
distance = np.linspace(min(z9),max(z9))
plt.hold(True)
plt.xlabel('p-value', fontsize = '10', fontweight="bold")
plt.ylabel('density', fontsize = '10', fontweight="bold")
plt.title('HSI Constituents Autocorrelation Test P-value (lag=4)', fontsize = '10', fontweight="bold")
plt.xticks(fontsize = '10', fontweight="bold")
plt.yticks(fontsize = '10', fontweight="bold")
plt.text(-0.2, -1.5, 'Since most pvalues are very small, most constituents log return are correlated', fontsize = '10')
plt.subplot(326)
plt.hist(z10, normed=True, bins=100)
distance = np.linspace(min(z10),max(z10))
plt.hold(True)
plt.xlabel('p-value', fontsize = '10', fontweight="bold")
plt.ylabel('density', fontsize = '10', fontweight="bold")
plt.title('HSI Constituents Autocorrelation Test P-value (lag=5)', fontsize = '10', fontweight="bold")
plt.xticks(fontsize = '10', fontweight="bold")
plt.yticks(fontsize = '10', fontweight="bold")
plt.text(-0.2, -2, 'Since most pvalues are very small, most constituents log return are correlated', fontsize = '10')


#Q8
df_lr2.index = pd.to_datetime(df_lr2.index)
year = np.unique(df_lr2.index.year) 
month = np.unique(df_lr2.index.month)
n = np.array(range(50))
ticker = list(df_lr2)
v = {}
for yi in year:
    for mi in month:
        try:
            temp = df_lr2.ix[(df_lr2.index.year == yi) & (df_lr2.index.month == mi)]
            if len(temp)==0:
                break
            else:
                v[yi * 100 + mi] = {}
                for index in n:
                    idx = ticker[index]
                    temp2 = temp.iloc[:, index]
                    v[yi * 100 + mi][idx] = []
                    vol = np.std(temp2)
                    v[yi * 100 + mi][idx].append(vol)
        except:
            continue
pd.DataFrame(v).to_csv('C:/Users/User/Desktop/FINA4380/HW1/2.csv')
