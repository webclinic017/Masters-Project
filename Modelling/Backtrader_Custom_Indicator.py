#Developing custom Class objects to replicate the behaviour of initial signal functions
#Necessary for utilisation with the cerebro framework
import numpy as np
import pandas as pd
import math
import pandas as pd
import pyodbc
import statsmodels.api as sm
import backtrader as bt 
import backtrader.indicator as btind

def meanreversion(price_data, ma=100, n=0):
    
    
    price_data['returns'] = np.log(price_data["Close"]).diff()
    #price_data['ma'] = price_data['Close'].rolling(ma).mean()
    price_data['ma']=((((price_data.High + price_data.Close + price_data.Low)/3)*price_data.Volume).rolling(ma).mean())/(price_data.Volume.rolling(ma).mean())
    price_data['ratio'] = price_data['Close'] / price_data['ma']

    percentiles = [1,2,3,4,5,10,15]
    p = np.percentile(price_data['ratio'].dropna(), percentiles)

    long = p[n]
    
    price_data['Mean_Signal'] = np.where(price_data.ratio < long, 1, 0)
    price_data['Mean_Signal'] = price_data['Mean_Signal'].shift(1)
    
    
    return price_data['Mean_Signal']

class MeanReversion(bt.Indicator):
    lines = ( 'signal',)

    params = (
        ('ma', 100),  
        ('n', 0),  
    ) 
    def __init__(self):
    #     self.addminperiod(self.params.ma)
        
        
    # def next(self):
        self.returns = np.diff(np.log(self.data.close))
        #m_a = ((((self.data.high + self.data.close + self.data.low)/3)*self.data.volume).rolling(self.params.ma).mean())/(self.data.volume.rolling(self.params.ma).mean())
        #self.avgdat = ((self.data.high + self.data.close + self.data.low)/3)*self.data.volume

        def moving_avg(x, n):
            cumsum = np.cumsum(np.insert(x, 0, 0)) 
            return (cumsum[n:] - cumsum[:-n]) / float(n)
        
        self.m_a = moving_avg(self.data.close, self.params.ma)
        
        #self.m_a = moving_avg(self.avgdat, self.params.ma)
        
        
        #self.m_a = bt.ind.SMA(self.avgdat, period=self.params.ma)
        # m_a = m_a.rolling(self.params.ma)
        # m_a = m_a.mean()
        
        self.v_a = moving_avg(self.data.volume, self.params.ma )
        #self.v_a = bt.ind.SMA(self.data.volume, period=self.params.ma) 
        # v_a = self.data.volume.rolling(self.params.ma)
        # v_a = v_a.mean()
        
        dist = len(self.returns) - len(self.m_a)
    
        #self.b_a =self.m_a/ self.v_a
        ratio = self.returns[dist:] /self.m_a
        #key = self.data.key
   
        # data_frame = pd.DataFrame()
        # data_frame["ratio"] = ratio
        #data_frame["key"] = key
        # data_frame = data_frame.dropna()
            
        percentiles = [1,2,3,4,5,10,15]
        p = np.percentile(ratio, percentiles)

        long = p[self.params.n]
        
        #Mean_Signal = np.where(ratio < long, 1, 0)
        Mean_Signal = pd.Series([1 if long > i else 0 for i in ratio])
        self.l.signal = Mean_Signal
        #self.lines.mean_id = data_frame.key.shift(1)
       
        
def yangzhang(price_data, window=30, trading_periods=252, clean=True):
    
    price_data = price_data.copy()
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

        
class YangZhang(bt.indicator):
    lines = ('vol',)
       
    params = (
        ('window', 30),  
        ('trading_periods', 252), 
        )
    def __init__(self):
        log_ho = (self.data.high / self.data.open).apply(np.log)
        log_lo = (self.data.low / self.data.open).apply(np.log)
        log_co = (self.data.close / self.data.open).apply(np.log)
       
        log_oc = (self.data.open / self.data.close.shift(1)).apply(np.log)
        log_oc_sq = log_oc**2
        
        log_cc = (self.data.close / self.data.close.shift(1)).apply(np.log)
        log_cc_sq = log_cc**2
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        close_vol = log_cc_sq.rolling(
            window=self.params.window,
            center=False
        ).sum() * (1.0 / (self.params.window - 1.0))
        open_vol = log_oc_sq.rolling(
            window=self.params.window,
            center=False
        ).sum() * (1.0 / (self.params.window - 1.0))
        window_rs = rs.rolling(
            window=self.params.window,
            center=False
        ).sum() * (1.0 / (self.params.window - 1.0))

        k = 0.34 / (1.34 + (self.params.window + 1) / (self.params.window - 1))
        result = ((open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(self.params.trading_periods))
        
        self.lines.vol = result

def volcrossing(price_data, short = 20, long = 30 ):
    
    price_data = price_data.copy()
    short = 20
    long = 30
    price_data["short_vol"] = yangzhang(price_data, window=short)
    price_data["long_vol"] = yangzhang(price_data, window=long)
    price_data = price_data.dropna()
    
    price_data['Vol_Signal']= np.where(price_data['short_vol'] > price_data['long_vol'],1,0)
    price_data['Vol_Signal'] = price_data['Vol_Signal'].shift(1)
    
    return price_data[[price_data.columns[0],'Vol_Signal']]


class VolCrossing (bt.indicator):
    lines = ('vol_id', 'signal',)

    params = (
        ('short', 20),  
        ('long', 30),  
        )
    def __init__(self):
        dat = pd.DataFrame()
        dat["short_vol"] = YangZhang(self.data, window=self.params.short)
        dat["long_vol"] = YangZhang(self.data, window=self.params.long)
        dat["key"] = self.data.key
        dat = dat.dropna()
        
        dat['vol_signal']= np.where(dat['short_vol'] > dat['long_vol'],1,0)
        dat['vol_signal'] = dat['vol_signal'].shift(1)
        
        self.lines.signal = dat.vol_signal
        self.lines.vol_id = dat.key

def tsmom(price_data, window = 1):
    
    price_data = price_data.copy()
    price_data['ret']= np.log(price_data.Close.pct_change()+1)
    price_data['prior_n']= price_data.ret.rolling(window).sum()
    price_data.dropna(inplace=True)
    price_data['TSMOM_Signal']= [1 if i > 0 else 0 for i in price_data.prior_n]
    price_data['TSMOM_Signal'] = price_data['TSMOM_Signal'].shift(1).dropna()
   
    return price_data[[price_data.columns[0],'TSMOM_Signal']]

class TSMOM(bt.indicator):
    lines = ('mom_id', 'signal',)

    params = (
        ('window', 2),   
        )
    
    def __init__(self):
        dat = pd.DataFrame()
        dat['ret']= np.log(self.data.close.pct_change()+1)
        dat['prior_n']= dat.ret.rolling(self.params.window).sum()
        dat["key"] = self.data.key
        dat.dropna(inplace=True)
        dat['TSMOM_Signal']= [1 if i > 0 else 0 for i in dat.prior_n]
        dat['TSMOM_Signal'] = dat['TSMOM_Signal'].shift(1).dropna()
        
        
        self.lines.signal = dat.TSMOM_Signal
        self.lines.mom_id = dat.key
        
def markov(price_data, p = 0.2):
    
    price_data = price_data.copy()
    price_data["Change"] = price_data[["Close"]].pct_change()
    mod_parameter = sm.tsa.MarkovRegression(price_data.Change.dropna(), k_regimes=2, trend='c', switching_variance=True)
    mod = mod_parameter.fit()
    price_data["Prob"] = mod.smoothed_marginal_probabilities[0]
    #mark_data["Markov_Signal"] = [1 if i > p else 0 for i in mark_data.Prob]
    price_data["Markov_Signal"] = np.where(price_data["Prob"] > p , 1,0)
    price_data["Markov_Signal"] = price_data["Markov_Signal"].shift(1).dropna()
    
    return price_data[[price_data.columns[0],"Markov_Signal"]]


class Markov(bt.indicator):
      lines = ('mark_id', 'signal',)

      params = (
          ('p', 0.5),   
          )
      
      def __init__(self):
          dat = pd.DataFrame()
          dat["key"] = self.data.key
          dat["Change"] = self.data.close.pct_change()
          mod_parameter = sm.tsa.MarkovRegression(dat.Change.dropna(), k_regimes=2, trend='c', switching_variance=True)
          mod = mod_parameter.fit()
          dat["Prob"] = mod.smoothed_marginal_probabilities[0]
          #mark_data["Markov_Signal"] = [1 if i > p else 0 for i in mark_data.Prob]
          dat["Markov_Signal"] = np.where(dat["Prob"] > self.params.p , 1,0)
          dat["Markov_Signal"] = dat["Markov_Signal"].shift(1).dropna()
          
          self.lines.signal = dat.Markov_Signal
          self.lines.mark_id = dat.key  
        
        
        
        
        
        
        
        
        
        
        
        
        
        