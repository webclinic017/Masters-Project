#Loading Data From local database into memory

import pandas as pd
import pyodbc

#Connections
conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectPK_one;Trusted_Connection=yes")
conn2 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectPartition_one;Trusted_Connection=yes")
conn3 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectWarehouse;Trusted_Connection=yes")

#SQL Statement for script matching
stmt = "SELECT [dbo].[Gold_table].[Datetime_1min],[Ticker_1min],[ONE_MINUTEKEY],[FIVE_MINUTEKEY],[FIFTEEN_MINUTEKEY],[THIRTY_MINUTEKEY],[ONE_HOURKEY],[TWO_HOURKEY],[FOUR_HOURKEY],[FIVE_HOURKEY],[TEN_HOURKEY],[EIGHT_HOURKEY],[TWELVE_HOURKEY],[ONE_DAYKEY],[ONE_WEEKKEY],[ONE_MONTHKEY]FROM [ProjectPK_one].[dbo].[Gold_table]order by Datetime_1min ASC"

#Forward fill of key values across all time frames
df = pd.read_sql(stmt,conn)
df.fillna(method='ffill', inplace=True)
dd = df.tail(1000)

#SQL Statements for pulling  Historical data for Gold from the database
stmt_1m = "SELECT [Datetime],[ONE_MINUTEKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM [ProjectPartition_one].[dbo].[ONE_MINUTEDim]WHERE Ticker = 'G'ORDER BY [dbo].[ONE_MINUTEDim].[Datetime] ASC"
stmt_5m = "SELECT [Datetime_5min],[FIVE_MINUTEKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[FIVE_MINUTEDim] WHERE Ticker = 'G'ORDER BY [Datetime_5min] ASC"
stmt_15m = "SELECT [Datetime],[FIFTEEN_MINUTEKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[FIFTEEN_MINUTEDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_30m = "SELECT [Datetime],[THIRTY_MINUTEKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[THIRTY_MINUTEDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_1h = "SELECT [Datetime],[ONE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_2h = "SELECT [Datetime],[TWO_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[TWO_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_4h = "SELECT [Datetime],[FOUR_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[FOUR_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_5h = "SELECT [Datetime],[FIVE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[FIVE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_10h = "SELECT [Datetime],[TEN_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[TEN_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_12h = "SELECT [Datetime],[TWELVE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[TWELVE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_1d = "SELECT [Datetime],[ONE_DAYKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_DAYDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_1w = "SELECT [Datetime],[ONE_WEEKKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_WEEKDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_1mon = "SELECT [Datetime],[ONE_MONTHKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_MONTHDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"

#Passing data into pandas dataframe
df_1min = pd.read_sql(stmt_1m,conn2)
df_5min = pd.read_sql(stmt_5m,conn3)
df_15min = pd.read_sql(stmt_15m,conn3)
df_30min = pd.read_sql(stmt_30m,conn3)
df_1hour = pd.read_sql(stmt_1h,conn3)
df_2hour = pd.read_sql(stmt_2h,conn3)
df_4hour = pd.read_sql(stmt_4h,conn3)
df_5hour = pd.read_sql(stmt_5h,conn3)
df_10hour = pd.read_sql(stmt_10h,conn3)
df_12hour = pd.read_sql(stmt_12h,conn3)
df_1d = pd.read_sql(stmt_1d,conn3)
df_1w = pd.read_sql(stmt_1w,conn3)
df_1mon = pd.read_sql(stmt_1mon,conn3)

###############################################################################################################
#Practicals for necessary calcutions and algorithms 

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math

#Passing one hour data 
dat_mod = df_1hour[["Datetime", "Open", "High", "Low", "Close", "Volume"]]


########################################################################################
#Implementing Markov Regime Switching to Gold Data
dat_mod = (dat_mod.dropna().set_index("Datetime"))
dat_mark = dat_mod[["Close"]]
#.loc['2015-01-01':'2021-08-31']
#dat_mark.index = pd.DatetimeIndex(dat_mark.index)

#dat_mark.iloc[:-1].head(100)


#Data Plot
dat_mark.plot(title="Gold Price",figsize=(12, 3))

#Fitting Markov Model
mod_fedfunds = sm.tsa.MarkovRegression(dat_mark, k_regimes=2)
res_fedfunds = mod_fedfunds.fit()


#Summary of Markov Model 
res_fedfunds.summary()

#Probability of Regimes
res_fedfunds.smoothed_marginal_probabilities[0].plot(title="Probability of being in the Low regime", figsize=(12, 5))
res_fedfunds.smoothed_marginal_probabilities[1].plot(title="Probability of being in the high regime", figsize=(12, 5))

############
# mod_fedfunds1 = sm.tsa.MarkovRegression(dat_mark.iloc[1:], k_regimes=2, exog = dat_mark.iloc[:-1])
# res_fedfunds1 = mod_fedfunds1.fit()

# res_fedfunds1.summary()

# res_fedfunds1.smoothed_marginal_probabilities[0].plot(title="Probability of being in the Low regime", figsize=(12, 5))
# res_fedfunds1.smoothed_marginal_probabilities[1].plot(title="Probability of being in the high regime", figsize=(12, 5))



####################################################################
#Volatility Estimator calculations

#Passing Data into another variable
price_data = dat_mod

#Garman Klass volatility estimator calculation and Function
def garman_klass(price_data, window=30, trading_periods=252, clean=True):

    log_hl = (price_data['High'] / price_data['Low']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

    rs = 0.5 * log_hl**2 - (2*math.log(2)-1) * log_co**2
    
    def f(v):
        return (trading_periods * v.mean())**0.5
    
    result = rs.rolling(window=window, center=False).apply(func=f)
    
    if clean:
        return result.dropna()
    else:
        return result
    
#Parkison Volatility Estimator Calculation and Function
def parkison(price_data, window=30, trading_periods=252, clean=True):

    rs = (1.0 / (4.0 * math.log(2.0))) * ((price_data['High'] / price_data['Low']).apply(np.log))**2.0

    def f(v):
        return (trading_periods * v.mean())**0.5
    
    result = rs.rolling(
        window=window,
        center=False
    ).apply(func=f)
    
    if clean:
        return result.dropna()
    else:
        return result

#Volatility Calculations and plots
vol1 = garman_klass(dat_mod, trading_periods=40)
vol2 = parkison(dat_mod)
vol1.loc['2020-08-01':'2021-05-31'].plot()
plt.title('Garman Klass Volatility Estimator')
plt.xlabel('Datetime')
plt.ylabel('Volatilty')
plt.savefig('Garman_Klass');


vol2.loc['2020-08-01':'2021-05-31'].plot()
plt.title('Parkison Volatility Estimator')
plt.xlabel('Datetime')
plt.ylabel('Volatility')
plt.savefig('Parkison');

vol1.loc['2020-08-01':'2021-05-31'].plot()
vol2.loc['2020-08-01':'2021-05-31'].plot()
plt.title('Garman Klass and Parkison Volatility Estimator')
plt.xlabel('Datetime')
plt.ylabel('Volatility')
plt.savefig('Garman_Klass_Parkison');


###################################################################################
#Mean Reversion Data
dat_mean = df_1hour[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
dat_mean= (dat_mean.dropna().set_index("Datetime"))

dat_mr = dat_mean[["Close"]]


#Function for mean reversion algorithm 
def meanreversion(ma= 100):
    dat_mr['returns'] = np.log(dat_mr["Close"]).diff()
    dat_mr['ma'] = dat_mr['Close'].rolling(ma).mean()
    dat_mr['ratio'] = dat_mr['Close'] / dat_mr['ma']

    dat_mr['ratio'].describe()

    percentiles = [1,5, 10, 50, 90, 95,99]
    p = np.percentile(dat_mr['ratio'].dropna(), percentiles)


    short = p[-1]
    long = p[0]
    dat_mr['position'] = np.where(dat_mr.ratio > short, -1, np.nan)
    dat_mr['position'] = np.where(dat_mr.ratio < long, 1, dat_mr['position'])
    dat_mr['position'] = dat_mr['position'].ffill()


    dat_mr['strat_return'] = dat_mr['returns'] * dat_mr['position'].shift()
    m = dat_mr['strat_return'].mean()
    s =dat_mr['strat_return'].std()
    
    #return print(np.exp(dat_mr['returns'].dropna()).cumprod()[-1] -1,np.exp(dat_mr['strat_return'].dropna()).cumprod()[-1] - 1)
    return np.exp(dat_mr['strat_return'].dropna()).cumprod()[-1] - 1 
    #return m/s


meanreversion(100)

#An attempt to a passic maximum-minimum optimisation technique
#Range of mean reversion lenght for optimisation
opt_max = pd.DataFrame(np.arange(10,350,5), columns=["SMA"])

#Applying the mean reversion algorithm over varrying lenghts 
opt_max['Return'] = opt_max['SMA'].apply(meanreversion)
plt.plot(opt_max["SMA"],opt_max["Return"], linestyle="-" )
plt.title('Returns v Lengths')
plt.xlabel('Length')
plt.ylabel('Returns')
plt.savefig('Return_Optimisation_Mean');


# opt_max['Sharpe_Ratio'] = opt_max['SMA'].apply(meanreversion)
# plt.plot(opt_max["SMA"],opt_max["Sharpe_Ratio"], linestyle="-" )

#Using the the lenght with the maximum return to plot the strategy/slgorithm
ma = opt_max.iloc[opt_max.Return.argmax(),0]  
#ma = opt_max.iloc[opt_max.Sharpe_Ratio.argmax(),0]  
dat_mr['returns'] = np.log(dat_mr["Close"]).diff()
dat_mr['ma'] = dat_mr['Close'].rolling(ma).mean()
dat_mr['ratio'] = dat_mr['Close'] / dat_mr['ma']

dat_mr['ratio'].describe()

percentiles = [1,5, 10, 50, 90, 95,99]
p = np.percentile(dat_mr['ratio'].dropna(), percentiles)


short = p[-1]
long = p[0]
dat_mr['position'] = np.where(dat_mr.ratio > short, -1, np.nan)
dat_mr['position'] = np.where(dat_mr.ratio < long, 1, dat_mr['position'])
dat_mr['position'] = dat_mr['position'].ffill()


dat_mr['strat_return'] = dat_mr['returns'] * dat_mr['position'].shift() 
 
print("Gold Returns:" ,np.exp(dat_mr['returns'].dropna()).cumprod()[-1] -1)
print("Strategy Returns:", np.exp(dat_mr['strat_return'].dropna()).cumprod()[-1] - 1)

#m = dat_mr['returns'].mean()
#s = dat_mr['returns'].std()
#s_r = m/s 
 

#Strategy vs Gold Returns Plot
plt.plot(np.exp(dat_mr['returns'].dropna()).cumprod(), label='Buy/Hold')
plt.plot(np.exp(dat_mr['strat_return'].dropna()).cumprod(), label='Strategy')
plt.legend()
plt.title('Strategy vs Gold Returns')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.savefig('Strat_v_Gold');

#Illustration of Mean Reversion Calculations
dat_mr['ratio'].dropna().plot(legend = True)
plt.axhline(p[0], c= (.5,.5,.5), ls='--')
plt.axhline(p[2], c= (.5,.5,.5), ls='--')
plt.axhline(p[-1], c= (.5,.5,.5), ls='--')
plt.title('Mean Reversion Ratio')
plt.savefig('Mean_Reversion_Ratio');

###################################################################################
























