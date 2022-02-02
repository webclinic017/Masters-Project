##############################################
#Mean Reversion Modification with the use of a Volume Weighted Avergae Price

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pandas as pd
import pyodbc


conn3 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectWarehouse;Trusted_Connection=yes")
stmt_1h = "SELECT [Datetime],[ONE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_1d = "SELECT [Datetime],[ONE_DAYKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_DAYDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_15m = "SELECT [Datetime],[FIFTEEN_MINUTEKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[FIFTEEN_MINUTEDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_12h = "SELECT [Datetime],[TWELVE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[TWELVE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"


df_1d = pd.read_sql(stmt_1d,conn3)
#df_1hour = pd.read_sql(stmt_1h,conn3)
#df_15min = pd.read_sql(stmt_15m,conn3)
#df_12hour = pd.read_sql(stmt_12h,conn3)

#dat_mod = df_1hour[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
#dat_mod2 = df_1d[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
dat_mean = df_1d[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
dat_mr= (dat_mean.dropna().set_index("Datetime"))



def meanreversion(ma= 100):
    dat_mr['returns'] = np.log(dat_mr["Close"]).diff()
    #dat_mr['ma'] = dat_mr['Close'].rolling(ma).mean()
    dat_mr['ma'] = ((((dat_mr.High + dat_mr.Close + dat_mr.Low)/3)*dat_mr.Volume).rolling(ma).mean())/(dat_mr.Volume.rolling(ma).mean())
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
    #return np.exp(dat_mr['strat_return'].dropna()).cumprod()[-1] - 1, 
    return m/s


######################################################################################################
ma = 100
dat_mr['returns'] = np.log(dat_mr["Close"]).diff()
#dat_mr['ma'] = dat_mr['Close'].rolling(ma).mean()
dat_mr['ma'] = ((((dat_mr.High + dat_mr.Close + dat_mr.Low)/3)*dat_mr.Volume).rolling(ma).mean())/(dat_mr.Volume.rolling(ma).mean())
dat_mr['ratio'] = dat_mr['Close'] / dat_mr['ma']

# dat_mr['ratio'].describe()

percentiles = [1,5, 10, 50, 90, 95,99]
p = np.percentile(dat_mr['ratio'].dropna(), percentiles)


short = p[-1]
long = p[0]

dat_mr['position'] = [1 if long > i else -1 if short < i else 0 for i in dat_mr['ratio']]



















meanreversion(100)

opt_max = pd.DataFrame(np.arange(10,350,5), columns=["SMA"])


#opt_max['Return'] = opt_max['SMA'].apply(meanreversion)
#plt.plot(opt_max["SMA"],opt_max["Return"], linestyle="-" )

opt_max['Sharpe_Ratio'] = opt_max['SMA'].apply(meanreversion)
plt.plot(opt_max["SMA"],opt_max["Sharpe_Ratio"], linestyle="-" );


#ma = opt_max.iloc[opt_max.Return.argmax(),0]  
ma = opt_max.iloc[opt_max.Sharpe_Ratio.argmax(),0]  
dat_mr['returns'] = np.log(dat_mr["Close"]).diff()
#dat_mr['ma'] = dat_mr['Close'].rolling(ma).mean()
dat_mr['ma'] = ((((dat_mr.High + dat_mr.Close + dat_mr.Low)/3)*dat_mr.Volume).rolling(ma).mean())/(dat_mr.Volume.rolling(ma).mean())
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
 
print(np.exp(dat_mr['returns'].dropna()).cumprod()[-1] -1)
print(np.exp(dat_mr['strat_return'].dropna()).cumprod()[-1] - 1)

#m = dat_mr['returns'].mean()
#s = dat_mr['returns'].std()
#s_r = m/s 
 

  
plt.plot(np.exp(dat_mr['returns'].dropna()).cumprod(), label='Buy/Hold')
plt.plot(np.exp(dat_mr['strat_return'].dropna()).cumprod(), label='Strategy')
plt.legend();




dat_mr['ratio'].dropna().plot(legend = True,figsize=(12, 5))
plt.axhline(p[0], c= (.5,.5,.5), ls='--')
plt.axhline(p[2], c= (.5,.5,.5), ls='--')
plt.axhline(p[-1], c= (.5,.5,.5), ls='--');


