#Multi TimeFrame Strategy Using Project Functions
import numpy as np
import pandas as pd
import math
import pandas as pd
import pyodbc
import statsmodels.api as sm
import matplotlib.pyplot as plt

import Project_Functions as pf
# from Project_Functions import tsmom
# from Project_Functions import yangzhang
# from project_functions import volcrossing
# from Project_Functions import meanreversion
# from Project_Functions import markov
# from Project_Functions import data_merge  

import backtrader as bt 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange
import time
import random
from deap import base  
from deap import creator
from deap import tools



class Totek(bt.Strategy):
    
    params = dict(data_high=1, data_mid=6, data_low=12, mean_len=30, mean_per=0, tsmom_window=15, vol_long=30, vol_short=20, prob=0.4, prof_mag=0.5, stop_mag=0.5, trading_period=252, clean=True )
    
    def __init__(self): 

        
        self.totek_mean = pf.meanreversion(price_data=self.datas[self.params.data_high], ma=self.params.mean_len, n=self.params.mean_per)
        self.totek_vol = pf.volcrossing(price_data=self.datas[self.params.data_high], short=self.params.vol_short, long=self.params.vol_long)
        self.totek_tsmom = pf.tsmom(price_data=self.datas[self.params.data_mid], window = self.params.tsmom_window)
        self.totek_markov = pf.markov(price_data=self.datas[self.params.data_low], prob = self.params.prob  )        
        self.num = [0,1,2,3,4]
        self.my_dict = {str(self.num[0]): self.data0, str(self.num[1]):self.totek_mean , str(self.num[2]): self.totek_vol, str(self.num[3]): self.totek_tsmom, str(self.num[4]): self.totek_markov}
        
        self.df = pf.data_merge(**self.my_dict)
        self.entry = self.df

        
    def next(self):

        if self.position.size == 0:

            if self.entry == 1:
                self.price = self.datas[self.params.data_high].close[0]
                self.price_limit = self.price * ( 1 +  (self.params.profit_mag * pf.yangzhang(self.datas[self.params.data_high], window=self.params.vol_short, trading_periods=self.params.trading_period, clean=self.params.clean)))
                self.price_stop = self.price * ( 1 -  (self.params.stop_mag * pf.yangzhang(self.datas[self.params.data_high], window=self.params.vol_short,trading_periods=self.params.trading_period, clean=self.params.clean)))

                self.long_buy_order = self.buy_bracket(
                    data=self.datas[self.params.data_high],
                    size=1,
                    exectype=bt.Order.Limit,
                    plimit=self.price,
                    stopprice=self.price_stop,
                    stopexec=bt.Order.Stop,
                    limitprice=self.price_limit,
                    limitexec=bt.Order.Limit,
                )


STRATEGY_PARAMS = dict(data_high=12, data_mid=13, data_low=14, mean_len=30, mean_per=0, tsmom_window=15, vol_long=30, vol_short=20, prob=0.4, prof_mag=0.5, stop_mag=0.5,trading_period=252, clean=True )

def run_backtest(plot=False, **strategy_params):

    cerebro = bt.Cerebro()
   
    #####################
    conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectPK_one;Trusted_Connection=yes")
    conn2 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectPartition_one;Trusted_Connection=yes")
    conn3 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectWarehouse;Trusted_Connection=yes")


    stmt = "SELECT [dbo].[Gold_table].[Datetime_1min],[Ticker_1min],[ONE_MINUTEKEY],[FIVE_MINUTEKEY],[FIFTEEN_MINUTEKEY],[THIRTY_MINUTEKEY],[ONE_HOURKEY],[TWO_HOURKEY],[FOUR_HOURKEY],[FIVE_HOURKEY],[TEN_HOURKEY],[EIGHT_HOURKEY],[TWELVE_HOURKEY],[ONE_DAYKEY],[ONE_WEEKKEY],[ONE_MONTHKEY]FROM [ProjectPK_one].[dbo].[Gold_table]order by Datetime_1min ASC"   
    stmt_1m = "SELECT [Datetime],[ONE_MINUTEKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM [ProjectPartition_one].[dbo].[ONE_MINUTEDim]WHERE Ticker = 'G'ORDER BY [dbo].[ONE_MINUTEDim].[Datetime] ASC"
    stmt_5m = "SELECT [Datetime_5min],[FIVE_MINUTEKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[FIVE_MINUTEDim] WHERE Ticker = 'G'ORDER BY [Datetime_5min] ASC"
    stmt_15m = "SELECT [Datetime],[FIFTEEN_MINUTEKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[FIFTEEN_MINUTEDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
    stmt_30m = "SELECT [Datetime],[THIRTY_MINUTEKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[THIRTY_MINUTEDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
    stmt_1h = "SELECT [Datetime],[ONE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
    stmt_2h = "SELECT [Datetime],[TWO_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[TWO_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
    stmt_4h = "SELECT [Datetime],[FOUR_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[FOUR_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
    stmt_5h = "SELECT [Datetime],[FIVE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[FIVE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
    stmt_8h = "SELECT [Datetime],[EIGHT_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].EIGHT_HOURDim WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
    stmt_10h = "SELECT [Datetime],[TEN_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[TEN_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
    stmt_12h = "SELECT [Datetime],[TWELVE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[TWELVE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
    stmt_1d = "SELECT [Datetime],[ONE_DAYKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_DAYDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
    stmt_1w = "SELECT [Datetime],[ONE_WEEKKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_WEEKDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
    stmt_1mon = "SELECT [Datetime],[ONE_MONTHKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_MONTHDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
    
    
    df = pd.read_sql(stmt,conn).rename(columns={"Datetime_1min" : "Datetime"})
    df.fillna(method='ffill', inplace=True)
    feed1 = bt.feeds.PandasData(df)
    cerebro.adddata(feed1 , name = 'd0')
    
    df_1min = bt.feeds.PandasData(pd.read_sql(stmt_1m,conn2).rename(columns={"Datetime" : "Datetime", "ONE_MINUTEKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_1min , name = 'd1')
    
    df_5min = bt.feeds.PandasData(pd.read_sql(stmt_5m,conn3).rename(columns={"Datetime_5min" : "Datetime", "FIVE_MINUTEKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_5min , name = 'd2')
    
    df_15min = bt.feeds.PandasData(pd.read_sql(stmt_15m,conn3).rename(columns={"Datetime" : "Datetime", "FIFTEEN_MINUTEKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_15min , name = 'd3')
    
    df_30min = bt.feeds.PandasData(pd.read_sql(stmt_30m,conn3).rename(columns={"Datetime" : "Datetime", "THIRTY_MINUTEKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_30min , name = 'd4')
    
    df_1hour = bt.feeds.PandasData(pd.read_sql(stmt_1h,conn3).rename(columns={"Datetime" : "Datetime", "ONE_HOURKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_1hour , name = 'd5')
    
    df_2hour = bt.feeds.PandasData(pd.read_sql(stmt_2h,conn3).rename(columns={"Datetime" : "Datetime", "TWO_HOURKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_2hour , name = 'd6')
    
    df_4hour = bt.feeds.PandasData(pd.read_sql(stmt_4h,conn3).rename(columns={"Datetime" : "Datetime", "FOUR_HOURKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_4hour , name = 'd7')
    
    df_5hour = bt.feeds.PandasData(pd.read_sql(stmt_5h,conn3).rename(columns={"Datetime" : "Datetime", "FIVE_HOURKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_5hour , name = 'd8')
    
    df_10hour = bt.feeds.PandasData(pd.read_sql(stmt_10h,conn3).rename(columns={"Datetime" : "Datetime","TEN_HOURKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_10hour , name = 'd9')
    
    df_8hour = bt.feeds.PandasData(pd.read_sql(stmt_8h,conn3).rename(columns={"Datetime" : "Datetime","EIGHT_HOURKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_8hour , name = 'd10')
    
    df_12hour = bt.feeds.PandasData(pd.read_sql(stmt_12h,conn3).rename(columns={"Datetime" : "Datetime", "TWELVE_HOURKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_12hour , name = 'd11')
    
    df_1d = bt.feeds.PandasData(pd.read_sql(stmt_1d,conn3).rename(columns={"Datetime" : "Datetime", "ONE_DAYKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_1d , name = 'd12')
    
    df_1w = bt.feeds.PandasData(pd.read_sql(stmt_1w,conn3).rename(columns={"Datetime" : "Datetime", "ONE_WEEKKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_1w , name = 'd13')
    
    df_1mon = bt.feeds.PandasData(pd.read_sql(stmt_1mon,conn3).rename(columns={"Datetime" : "Datetime", "ONE_MONTHKEY": "key"}).set_index("Datetime").dropna())
    cerebro.adddata(df_1mon , name = 'd14')
    

    # Remember to set it high enough or the strategy may not
    # be able to trade because of short of cash
    cerebro.broker.setcash(100000.00)

    # Print out the starting conditions
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():,.2f}")

    # Although we have defined some default params in the strategy,
    # we can override it by passing in keyword arguments here.
    cerebro.addstrategy(Totek, **strategy_params)

    cerebro.addobserver(bt.observers.Trades)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio)

    # Let's say that we have 0.25% slippage and commission per trade,
    # that is 0.5% in total for a round trip.
    cerebro.broker.setcommission(commission=0.0025, margin=False)

    # Run over everything
    strats = cerebro.run()

    print(f"Final Portfolio Value:    {cerebro.broker.getvalue():,.2f}")

    if plot:
        cerebro.plot()


run_backtest(plot=False, **STRATEGY_PARAMS)