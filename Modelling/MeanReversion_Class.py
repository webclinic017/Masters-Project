#Custom Mean Reversion Indicator
import numpy as np
import pandas as pd
import math
import pandas as pd
import pyodbc
import statsmodels.api as sm
import backtrader as bt 
import backtrader.indicator as btind

# def meanreversion(price_data, ma=100, n=0):
    
    
#     price_data['returns'] = np.log(price_data["Close"]).diff()
#     #price_data['ma'] = price_data['Close'].rolling(ma).mean()
#     price_data['ma']=((((price_data.High + price_data.Close + price_data.Low)/3)*price_data.Volume).rolling(ma).mean())/(price_data.Volume.rolling(ma).mean())
#     price_data['ratio'] = price_data['Close'] / price_data['ma']

#     percentiles = [1,2,3,4,5,10,15]
#     p = np.percentile(price_data['ratio'].dropna(), percentiles)

#     long = p[n]
    
#     price_data['Mean_Signal'] = np.where(price_data.ratio < long, 1, 0)
#     price_data['Mean_Signal'] = price_data['Mean_Signal'].shift(1)
    
    
#     return price_data['Mean_Signal']

class MeanReversion(bt.Indicator):
    lines = ('signal',)

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
        self.signal = pd.Series([1 if long > i else 0 for i in ratio])
        self.l.signal = self.signal
        #self.lines.mean_id = data_frame.key.shift(1)