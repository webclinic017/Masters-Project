#Time Series Momentum Algorithm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyodbc


conn3 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectWarehouse;Trusted_Connection=yes")
stmt_1h = "SELECT [Datetime],[ONE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_1d = "SELECT [Datetime],[ONE_DAYKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_DAYDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_15m = "SELECT [Datetime],[FIFTEEN_MINUTEKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[FIFTEEN_MINUTEDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"


df_1d = pd.read_sql(stmt_1d,conn3)
#df_1hour = pd.read_sql(stmt_1h,conn3)
#df_15min = pd.read_sql(stmt_15m,conn3)
dat_mean = df_1d[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
dat_mom= (dat_mean.dropna().set_index("Datetime"))
dat_mom.index = pd.DatetimeIndex(dat_mom.index)

def tsmom(df, window = 1):
    df = dat_mom.copy()
    df['ret']= np.log(df.Close.pct_change()+1)
    df['prior_n']= df.ret.rolling(window).sum()
    df.dropna(inplace=True)
    df['position']= [1 if i > 0 else -1 for i in df.prior_n]
    df['strat'] = df.position.shift(1)*df.ret
    
    m = df['strat'].mean()
    s = df['strat'].std()
    
    #return np.exp(df[['ret', 'strat']].cumsum()).plot(figsize=(12,5))
    # return m/s
    return df['position']

mom_out = tsmom(dat_mom, window = 20)
mom_out.loc['2018-12-01':'2021-01-31'].plot()
plt.title('Time Series Momentum Signal')
plt.xlabel('Datetime')
plt.ylabel('Signal')
plt.savefig('TSMOM_signal');


#opt_mom = pd.DataFrame(np.arange(10,80,10), columns=["Length"])


#opt_max['Return'] = opt_max['SMA'].apply(meanreversion)
#plt.plot(opt_max["SMA"],opt_max["Return"], linestyle="-" )
#
#opt_mom['Sharpe_Ratio'] = opt_mom['Length'].apply(tsmom)
#plt.plot(opt_mom["Length"],opt_mom["Sharpe_Ratio"], linestyle="-" )



    
    