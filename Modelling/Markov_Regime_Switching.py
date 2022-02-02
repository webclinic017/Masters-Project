import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pandas as pd
import pyodbc
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

conn3 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectWarehouse;Trusted_Connection=yes")
stmt_1h = "SELECT [Datetime],[ONE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_1d = "SELECT [Datetime],[ONE_DAYKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_DAYDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_15m = "SELECT [Datetime],[FIFTEEN_MINUTEKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[FIFTEEN_MINUTEDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_12h = "SELECT [Datetime],[TWELVE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[TWELVE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"


df_1d = pd.read_sql(stmt_1d,conn3)
df_1hour = pd.read_sql(stmt_1h,conn3)
#df_15min = pd.read_sql(stmt_15m,conn3)
df_12hour = pd.read_sql(stmt_12h,conn3)

dat_mod = df_1hour[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
dat_mod2 = df_1d[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
#dat_mod3 = df_15min[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
dat_mod4 = df_12hour[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

########################################################################################
#dat_mod3 = (dat_mod3.dropna().set_index("Datetime"))
#dat_mod2 = (dat_mod2.dropna().set_index("Datetime"))
#dat_mod = (dat_mod.dropna().set_index("Datetime"))

#dat_mark = dat_mod[["Close"]]
#.loc['2015-01-01':'2021-08-31']
#dat_mark.index = pd.DatetimeIndex(dat_mark.index)

#dat_mark.iloc[:-1].head(100)

#dat_mark.plot(title="Gold Price",figsize=(12, 3))


# mod_fedfunds = sm.tsa.MarkovRegression(dat_mark, k_regimes=2)
# res_fedfunds = mod_fedfunds.fit()

# res_fedfunds.summary()


# res_fedfunds.smoothed_marginal_probabilities[0].plot(title="Probability of being in the Low regime", figsize=(12, 5))
# res_fedfunds.smoothed_marginal_probabilities[1].plot(title="Probability of being in the high regime", figsize=(12, 5))

# regime_prob = res_fedfunds.smoothed_marginal_probabilities

############
#mod_fedfunds1 = sm.tsa.MarkovRegression(dat_mark.iloc[1:], k_regimes=2, exog = dat_mark.iloc[:-1])
#res_fedfunds1 = mod_fedfunds1.fit()

#res_fedfunds1.summary()

#res_fedfunds1.smoothed_marginal_probabilities[0].plot(title="Probability of being in the Low regime", figsize=(12, 5))
#res_fedfunds1.smoothed_marginal_probabilities[1].plot(title="Probability of being in the high regime", figsize=(12, 5))

##########################################################################################
dat_mod4 = (dat_mod4.dropna().set_index("Datetime"))
dat_mark_ret = dat_mod4[["Close"]].pct_change()
#dat_mark_ret.index = pd.DatetimeIndex(dat_mark_ret.index)
dat_mark_ret.plot(title='Excess returns')
plt.xlabel('Datetime')
plt.ylabel('Returns')
plt.savefig('Gold_DailyReturns_Markov');

#Adfuller Test
adfuller(dat_mark_ret.dropna())

#Fit the model
mod_kns = sm.tsa.MarkovRegression(dat_mark_ret.dropna(), k_regimes=3, trend='nc', switching_variance=True)
res_kns = mod_kns.fit()
res_kns.summary()

#Plotting Fitted Probability from markov model  
res_kns.smoothed_marginal_probabilities[0].loc['2020-08-01':'2021-05-31'].plot(title="Probability of being in the Low regime")
plt.xlabel('Datetime')
plt.ylabel('Probability')
plt.savefig('mark_low_regime');

res_kns.smoothed_marginal_probabilities[1].loc['2020-08-01':'2021-05-31'].plot(title="Probability of being in the high regime")
plt.xlabel('Datetime')
plt.ylabel('Probability')
plt.savefig('mark_high_regime');

res_kns.smoothed_marginal_probabilities















