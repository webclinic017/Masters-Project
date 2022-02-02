#All Volatility Estimation Calculations 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pandas as pd
import pyodbc


conn3 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectWarehouse;Trusted_Connection=yes")
stmt_1h = "SELECT [Datetime],[ONE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"

df_1hour = pd.read_sql(stmt_1h,conn3)
dat_mod = df_1hour[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
dat_mod = (dat_mod.dropna().set_index("Datetime"))

price_data = dat_mod


#Garman Klass
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
    
#Parkison
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

#YangZhang
def yangzhang(price_data, window=30, trading_periods=252, clean=True):

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
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)

    if clean:
        return result.dropna()
    else:
        return 

#Rogerssatchell
def rogerssatchell(price_data, window=30, trading_periods=252, clean=True):
    
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

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

#HodgesTompkins
def hodgestompkins(price_data, window=30, trading_periods=252, clean=True):
    
    log_return = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)

    vol = log_return.rolling(
        window=window,
        center=False
    ).std() * math.sqrt(trading_periods)

    h = window
    n = (log_return.count() - h) + 1

    adj_factor = 1.0 / (1.0 - (h / n) + ((h**2 - 1) / (3 * n**2)))

    result = vol * adj_factor

    if clean:
        return result.dropna()
    else:
        return result
   
    
   
vol1 = garman_klass(dat_mod, window=30)
vol2 = parkison(dat_mod, window=40)
vol3 = yangzhang(dat_mod, window=50)
vol4 = rogerssatchell(dat_mod, window=60)
vol5 = hodgestompkins(dat_mod, window=70)


# vol1.loc['2020-12-01':'2021-01-31'].plot(figsize=(12, 5))

#Volatility Estimators with varying lengths (Leading and Lagging indicators)
vol2.loc['2020-12-01':'2021-01-31'].plot()
vol3.loc['2020-12-01':'2021-01-31'].plot()
plt.title('Leading and Lagging Volatility Estimator')
plt.xlabel('Datetime')
plt.ylabel('Volatility')
plt.savefig('Vol_lag_lead');

#vol4.loc['2020-12-01':'2021-05-31'].plot(figsize=(12, 5))
# vol5.loc['2020-12-01':'2021-01-31'].plot(figsize=(12, 5))



###############################################################################
#Volatility Crossing Algorithm
short = 20
long = 30

price_data = pd.DataFrame()
price_data["short_vol"] = yangzhang(dat_mod, window=short)
price_data["long_vol"] = yangzhang(dat_mod, window=long)
price_data = price_data.dropna()

price_data['signal']= np.where(price_data['short_vol'] > price_data['long_vol'],1,0)
price_data['signal']= np.where(price_data['short_vol'] < price_data['long_vol'],-1, price_data['signal'])
price_data.dropna(inplace=True)
price_data.head(10)


price_data['signal'].loc['2020-12-01':'2021-01-31'].plot()
plt.title('Volatility Crossing Signal')
plt.xlabel('Datetime')
plt.ylabel('Signal')
plt.savefig('Vol_signal');

# price_data['return'] = np.log(price_data['Close']).diff()
# price_data['system_return'] = price_data['signal']*price_data['return']
# price_data['entry']  =price_data.signal.diff()



#plt.rcParams['figure.figsize'] = 12, 6
#plt.grid(True, alpha = .3)
#plt.plot(price_data.iloc[-252:]['Close'], label = 'GLD')
#plt.plot(price_data.iloc[-252:]['short_vol'], label = 'Short_vol')
#plt.plot(price_data.iloc[-252:]['long_vol'], label = 'long_vol')
#plt.plot(price_data[-252:].loc[price_data.entry == 2].index, price_data[-252:]['short_vol'][price_data.entry == 2],'^',color = 'g', markersize = 12)
#plt.plot(price_data[-252:].loc[price_data.entry == -2].index, price_data[-252:]['long_vol'][price_data.entry == -2], 'v',color = 'r', markersize = 12)
#plt.legend(loc=2);


# plt.plot(np.exp(price_data['return']).cumprod(), label='Buy/Hold') 
# plt.plot(np.exp(price_data['system_return']).cumprod(), label='System')
# plt.legend(loc=2)
# plt.grid(True, alpha=.3)


# np.exp(price_data['return']).cumprod()[-1]-1
# np.exp(price_data['system_return']).cumprod()[-1]-1







