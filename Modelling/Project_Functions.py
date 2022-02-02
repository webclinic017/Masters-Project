#Bespoke Functions for signal generation for various algorithms 
#in addition to function utilising Dataase Keys for matching of signals

import numpy as np
import pandas as pd
import math
import pandas as pd
import pyodbc
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time

stmt = "SELECT [dbo].[Gold_table].[Datetime_1min],[Ticker_1min],[ONE_MINUTEKEY],[FIVE_MINUTEKEY],[FIFTEEN_MINUTEKEY],[THIRTY_MINUTEKEY],[ONE_HOURKEY],[TWO_HOURKEY],[FOUR_HOURKEY],[FIVE_HOURKEY],[TEN_HOURKEY],[EIGHT_HOURKEY],[TWELVE_HOURKEY],[ONE_DAYKEY],[ONE_WEEKKEY],[ONE_MONTHKEY]FROM [ProjectPK_one].[dbo].[Gold_table]order by Datetime_1min ASC"
conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectPK_one;Trusted_Connection=yes")
conn3 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectWarehouse;Trusted_Connection=yes")
stmt_1h = "SELECT [Datetime],[ONE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
stmt_12h = "SELECT [Datetime],[TWELVE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[TWELVE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"


df_1hour = pd.read_sql(stmt_1h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
#.set_axis(['Datetime', 'Key','Ticker','Open','Low', 'High','Close','Volume'], axis=1).dropna().set_index("Datetime")
df_12hour = pd.read_sql(stmt_12h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
#.set_axis(['Datetime', 'Key','Ticker','Open','Low', 'High','Close','Volume'], axis=1).dropna().set_index("Datetime")

# t = time.perf_counter()
df = pd.read_sql(stmt,conn).rename(columns={"Datetime_1min" : "Datetime"}).set_index("Datetime")
df.fillna(method='ffill', inplace=True)
# # end_t = time.perf_counter()
# # print(f"Time Elapsed: {end_t - t:,.2f}")

# # x = df.tail(100)

# # #dat_mean = df_1hour[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
# # #dat_mean=df_1hour
# # #df_1hour= (df_1hour.dropna().set_index("Datetime"))

# # price_data = df_1hour


#Function for Mean Reversion Signal 
def meanreversion(price_data, ma=100, n=0):
    
    # price_data = price_data.copy()
    price_data['returns'] = np.log(price_data["Close"]).diff()
    #price_data['ma'] = price_data['Close'].rolling(ma).mean()
    price_data['ma']=((((price_data.High + price_data.Close + price_data.Low)/3)*price_data.Volume).rolling(ma).mean())/(price_data.Volume.rolling(ma).mean())
    price_data['ratio'] = price_data['Close'] / price_data['ma']

    percentiles = [1,2,3,4,5,10,15]
    p = np.percentile(price_data['ratio'].dropna(), percentiles)

    long = p[n]
    
    price_data['Mean_Signal'] = np.where(price_data.ratio < long, 1, 0)
    price_data['Mean_Signal'] = price_data['Mean_Signal'].shift(1)
    
    
    return price_data[[price_data.columns[0],'Mean_Signal']]


#YangZhang Volatility Estimator
def yangzhang(price_data, window=30, trading_periods=252, clean=True):
    
    # price_data = price_data.copy()
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    result = ((open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods))

    if clean:
        return result.dropna().shift()
    else:
        return 

#Volatility Corssing Signal Function Using YangZhang Volality Estimators
def volcrossing(price_data, short = 20, long = 30 ):
    
    # price_data = price_data.copy()
    short = 20
    long = 30
    price_data["short_vol"] = yangzhang(price_data, window=short)
    price_data["long_vol"] = yangzhang(price_data, window=long)
    price_data = price_data.dropna()
    
    price_data['Vol_Signal']= np.where(price_data['short_vol'] > price_data['long_vol'],1,0)
    price_data['Vol_Signal'] = price_data['Vol_Signal'].shift(1)
    
    return price_data[[price_data.columns[0],'Vol_Signal']]


def tsmom(price_data, window = 1):
    
    # price_data = price_data.copy()
    price_data['ret']= np.log(price_data.Close.pct_change()+1)
    price_data['prior_n']= price_data.ret.rolling(window).sum()
    price_data.dropna(inplace=True)
    price_data['TSMOM_Signal']= [1 if i > 0 else 0 for i in price_data.prior_n]
    price_data['TSMOM_Signal'] = price_data['TSMOM_Signal'].shift(1).dropna()
   
    return price_data[[price_data.columns[0],'TSMOM_Signal']]

#Signal Generation Using a markov regime switching model
def markov(price_data, p = 0.2):
    
    # price_data = price_data.copy()
    price_data["Change"] = price_data[["Close"]].pct_change()
    mod_parameter = sm.tsa.MarkovRegression(price_data.Change.dropna(), k_regimes=2, trend='c', switching_variance=True)
    mod = mod_parameter.fit()
    price_data["Prob"] = mod.smoothed_marginal_probabilities[0]
    #mark_data["Markov_Signal"] = [1 if i > p else 0 for i in mark_data.Prob]
    price_data["Markov_Signal"] = np.where(price_data["Prob"] > p , 1,0)
    price_data["Markov_Signal"] = price_data["Markov_Signal"].shift(1).dropna()
    
    return price_data[[price_data.columns[0],"Markov_Signal"]]

a = df
b = meanreversion(df_1hour, ma=22,n=1)
g= yangzhang(df_1hour, window=22, trading_periods=2016, clean=True)
g.plot(figsize=(12, 5))
c= volcrossing(df_1hour, short = 150, long = 250)
d=tsmom(df_12hour, window = 6)

e=markov(df_12hour, p = 0.7)

# x.plot(title="Probability of being in the Low regime", figsize=(12, 5))

# y = markov(my_dict[2], regimes = 2 , p = 0.8)


###################################

abc = [0,1,2,3,4]
# # a = []
# # b = []
# # c = []
# # d = []
# # e = []
my_dict = {str(abc[0]): a, str(abc[1]):b , str(abc[2]): c, str(abc[3]): d, str(abc[4]): e}


#Merger function to output multi time frame signal output
def data_merge(**my_dict):
    ref_data = my_dict['0']
    ref_data = ref_data.merge(my_dict['1'], on = my_dict['1'].columns[0], how='left')
    ref_data = ref_data.merge(my_dict['2'], on = my_dict['2'].columns[0], how='left')
    ref_data = ref_data.merge(my_dict['3'], on = my_dict['3'].columns[0], how='left')
    ref_data = ref_data.merge(my_dict['4'], on = my_dict['4'].columns[0], how='left')
    
    ref_data = ref_data.set_index("Datetime")
    ref_data = ref_data.iloc[:,15:]
    ref_data = my_dict['1'].merge(ref_data, left_index=True, right_index=True, how='inner').iloc[:,2:]
    ref_data["Entry"] = ref_data[ref_data.columns[0]] + ref_data[ref_data.columns[1]] + ref_data[ref_data.columns[2]] + ref_data[ref_data.columns[3]]
    ref_data["Entry"] = np.where(ref_data["Entry"] == 4 , 1,0)
    
    return ref_data["Entry"].dropna()

y = data_merge(**my_dict)
# y.Entry.describe()










    
    
    
    






