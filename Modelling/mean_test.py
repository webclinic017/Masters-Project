#Mean Reversion Backtest without calling the indicator module
import numpy as np
import pandas as pd
import math
import pandas as pd
import pyodbc
import statsmodels.api as sm
import matplotlib.pyplot as plt

import Project_Functions as pf
import MeanReversion_Class as mr
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
    
    params = dict( mean_len=30, mean_per=0, buy_limit_adjust =0.05, buy_stop_adjust =0.02 )
    
    def __init__(self): 

        #self.mean = mr.MeanReversion(self.data, ma=self.params.mean_len, n=self.params.mean_per)
        self.returns = np.diff(np.log(self.data.close))
        def moving_avg(x, n):
            cumsum = np.cumsum(np.insert(x, 0, 0)) 
            return (cumsum[n:] - cumsum[:-n]) / float(n)
        
        self.m_a = moving_avg(self.data.close, self.params.mean_len)
        #self.v_a = moving_avg(self.data.volume, self.params.mean_len )
        dist = len(self.returns) - len(self.m_a)
    

        ratio = self.returns[dist:] /self.m_a
        percentiles = [1,2,3,4,5,10,15]
        p = np.percentile(ratio, percentiles)

        long = p[self.params.mean_per]
        self.mean = [1 if long > i else 0 for i in ratio]
    
    
    
        
    def next(self):

        if self.position.size == 0:

            if self.mean == 1:
                price = self.data.close[0]
                price_limit = price * (1.0 + self.p.buy_limit_adjust)
                price_stop = price * (1.0 - self.p.buy_stop_adjust)
                
                self.long_buy_order = self.buy_bracket(
                    data=self.data,
                    size=65,
                    exectype=bt.Order.Limit,
                    plimit=price,
                    stopprice=price_stop,
                    stopexec=bt.Order.Stop,
                    limitprice=price_limit,
                    limitexec=bt.Order.Limit,
                )


STRATEGY_PARAMS = dict( mean_len=30, mean_per=0, buy_limit_adjust =0.05, buy_stop_adjust = 0.02 )


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


# df = pd.read_sql(stmt,conn).rename(columns={"Datetime_1min" : "Datetime"}).set_index("Datetime")
# df.fillna(method='ffill', inplace=True)

# min_1 = pd.read_sql(stmt_1m,conn2).rename(columns={"Datetime" : "Datetime", "ONE_MINUTEKEY": "key"}).set_index("Datetime").dropna()
# min_5 = pd.read_sql(stmt_5m,conn3).rename(columns={"Datetime_5min" : "Datetime", "FIVE_MINUTEKEY": "key"}).set_index("Datetime").dropna()
# min_15 = pd.read_sql(stmt_15m,conn3).rename(columns={"Datetime" : "Datetime", "FIFTEEN_MINUTEKEY": "key"}).set_index("Datetime").dropna()
# min_30 = pd.read_sql(stmt_30m,conn3).rename(columns={"Datetime" : "Datetime", "THIRTY_MINUTEKEY": "key"}).set_index("Datetime").dropna()
# hour_1 = pd.read_sql(stmt_1h,conn3).rename(columns={"Datetime" : "Datetime", "ONE_HOURKEY": "key"}).set_index("Datetime").dropna()
# hour_2 = pd.read_sql(stmt_2h,conn3).rename(columns={"Datetime" : "Datetime", "TWO_HOURKEY": "key"}).set_index("Datetime").dropna()
# hour_4 = pd.read_sql(stmt_4h,conn3).rename(columns={"Datetime" : "Datetime", "FOUR_HOURKEY": "key"}).set_index("Datetime").dropna()
# hour_5 = pd.read_sql(stmt_5h,conn3).rename(columns={"Datetime" : "Datetime", "FIVE_HOURKEY": "key"}).set_index("Datetime").dropna()
# hour_8 = pd.read_sql(stmt_8h,conn3).rename(columns={"Datetime" : "Datetime","EIGHT_HOURKEY": "key"}).set_index("Datetime").dropna()
# hour_10 = pd.read_sql(stmt_10h,conn3).rename(columns={"Datetime" : "Datetime","TEN_HOURKEY": "key"}).set_index("Datetime").dropna()
# hour_12 = pd.read_sql(stmt_12h,conn3).rename(columns={"Datetime" : "Datetime", "TWELVE_HOURKEY": "key"}).set_index("Datetime").dropna()
# day_1 = pd.read_sql(stmt_1d,conn3).rename(columns={"Datetime" : "Datetime", "ONE_DAYKEY": "key"}).set_index("Datetime").dropna()
# week_1 = pd.read_sql(stmt_1w,conn3).rename(columns={"Datetime" : "Datetime", "ONE_WEEKKEY": "key"}).set_index("Datetime").dropna()
# month_1 = pd.read_sql(stmt_1mon,conn3).rename(columns={"Datetime" : "Datetime", "ONE_MONTHKEY": "key"}).set_index("Datetime").dropna()    
  

# min_1 = pd.read_sql(stmt_1m,conn2).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# min_5 = pd.read_sql(stmt_5m,conn3).rename(columns={"Datetime_5min" : "Datetime"}).set_index("Datetime").dropna()
# min_15 = pd.read_sql(stmt_15m,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# min_30 = pd.read_sql(stmt_30m,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_1 = pd.read_sql(stmt_1h,conn3)[["Datetime", "Open", "High", "Low", "Close", "Volume"]].rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# hour_2 = pd.read_sql(stmt_2h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# hour_4 = pd.read_sql(stmt_4h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# hour_5 = pd.read_sql(stmt_5h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# hour_8 = pd.read_sql(stmt_8h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# hour_10 = pd.read_sql(stmt_10h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# hour_12 = pd.read_sql(stmt_12h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# day_1 = pd.read_sql(stmt_1d,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# week_1 = pd.read_sql(stmt_1w,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# month_1 = pd.read_sql(stmt_1mon,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()    
  
    
def run_backtest(plot=False, **strategy_params):

    cerebro = bt.Cerebro()
   
    #####################
    # df_1min = bt.feeds.PandasData(dataname=min_1)
    # cerebro.adddata(df_1min , name = 'd1')
    
    
    # feed1 = bt.feeds.PandasData(dataname=df)
    # cerebro.adddata(feed1 , name = 'd0')
    
 
    # df_5min = bt.feeds.PandasData(dataname=min_5)
    # cerebro.adddata(df_5min , name = 'd2')
    
    # df_15min = bt.feeds.PandasData(dataname=min_15)
    # cerebro.adddata(df_15min , name = 'd3')
    
    # df_30min = bt.feeds.PandasData(dataname=min_30)
    # cerebro.adddata(df_30min , name = 'd4')
    
    df_1hour = bt.feeds.PandasData(dataname=hour_1)
    cerebro.adddata(df_1hour , name = 'd5')
    
    # df_2hour = bt.feeds.PandasData(dataname=hour_2)
    # cerebro.adddata(df_2hour , name = 'd6')
    
    # df_4hour = bt.feeds.PandasData(dataname=hour_4)
    # cerebro.adddata(df_4hour , name = 'd7')
    
    # df_5hour = bt.feeds.PandasData(dataname=hour_5)
    # cerebro.adddata(df_5hour , name = 'd8')
    
    # df_8hour = bt.feeds.PandasData(dataname=hour_8)
    # cerebro.adddata(df_8hour , name = 'd8')   
    
    # df_10hour = bt.feeds.PandasData(dataname=hour_10)
    # cerebro.adddata(df_10hour , name = 'd10')
     
    # df_12hour = bt.feeds.PandasData(dataname=hour_12)
    # cerebro.adddata(df_12hour , name = 'd11')
    
    # df_1d = bt.feeds.PandasData(dataname=day_1)
    # cerebro.adddata(df_1d , name = 'd12')
    
    # df_1w = bt.feeds.PandasData(dataname=week_1)
    # cerebro.adddata(df_1w , name = 'd13')
    
    # df_1mon = bt.feeds.PandasData(dataname=month_1)
    # cerebro.adddata(df_1mon , name = 'd14')
    

    # Remember to set it high enough or the strategy may not
    # be able to trade because of short of cash
    cerebro.broker.setcash(100000.00)

    # Print out the starting conditions
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():,.2f}")

    # Although we have defined some default p in the strategy,
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


run_backtest(plot=True, **STRATEGY_PARAMS)


# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from tqdm import trange
# import time
# import random
# from deap import base
# from deap import creator
# from deap import tools


# # fix the seed so that we will get the same results
# # feel free to change it or comment out the line
# random.seed(1)

# # GA parameters
# PARAM_NAMES = ["mean_len", "mean_per", "buy_limit_adjust", "buy_stop_adjust"]
# NGEN = 10
# NPOP = 100
# CXPB = 0.5
# MUTPB = 0.3

# data = pd.read_sql(stmt_1h,conn3)[["Datetime", "Open", "High", "Low", "Close", "Volume"]].rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()

# def evaluate(individual, plot=False, log=False):

#     # convert list of parameter values into dictionary of kwargs
#     strategy_params = {k: v for k, v in zip(PARAM_NAMES, individual)}

#     # fast moving average by definition cannot be slower than the slow one
#     # if strategy_params["fast_period"] >= strategy_params["slow_period"]:
#     #     return [-np.inf]

#     # by setting stdstats to False, backtrader will not store the changes in
#     # statistics like number of trades, buys & sells, etc.
#     cerebro = bt.Cerebro(stdstats=False)
#     cerebro.adddata(data)

#     # Remember to set it high enough or the strategy may not
#     # be able to trade because of short of cash
#     initial_capital = 100000.0
#     cerebro.broker.setcash(initial_capital)

#     # Pass in the genes of the individual as kwargs
#     cerebro.addstrategy(Totek, **strategy_params)

#     # This is needed for calculating our fitness score
#     cerebro.addanalyzer(bt.analyzers.DrawDown)

#     # Let's say that we have 0.25% slippage and commission per trade,
#     # that is 0.5% in total for a round trip.
#     cerebro.broker.setcommission(commission=0.0025, margin=False)

#     # Run over everything
#     strats = cerebro.run()

#     profit = cerebro.broker.getvalue() - initial_capital
#     max_dd = strats[0].analyzers.drawdown.get_analysis()["max"]["moneydown"]
#     fitness = profit / (max_dd if max_dd > 0 else 1)

#     if log:
#         print(f"Starting Portfolio Value: {initial_capital:,.2f}")
#         print(f"Final Portfolio Value:    {cerebro.broker.getvalue():,.2f}")
#         print(f"Total Profit:             {profit:,.2f}")
#         print(f"Maximum Drawdown:         {max_dd:,.2f}")
#         print(f"Profit / Max DD:          {fitness}")

#     if plot:
#         cerebro.plot()

#     return [fitness]


# # our fitness score is supposed to be maximised and there is only 1 objective
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# # our individual is a list of genes, with the fitness score the higher the better
# creator.create("Individual", list, fitness=creator.FitnessMax)

# # register some handy functions for calling
# toolbox = base.Toolbox()
# toolbox.register("indices", random.sample, range(NPOP), NPOP)
# # crossover strategy
# toolbox.register("mate", tools.cxUniform, indpb=CXPB)
# # mutation strategy
# toolbox.register("mutate", tools.mutUniformInt, low=1, up=151, indpb=0.2)
# # selection strategy
# toolbox.register("select", tools.selTournament, tournsize=3)
# # fitness function
# toolbox.register("evaluate", evaluate)

# # definition of an individual & a population
# toolbox.register("attr_mean_len", random.randint, 1, 51)
# toolbox.register("attr_mean_per", random.randint, 10, 151)
# toolbox.register("attr_buy_limit_adjust", random.randint, 1, 101)
# toolbox.register("attr_buy_stop_adjust", random.randint, 1, 101)
# toolbox.register(
#     "individual",
#     tools.initCycle,
#     creator.Individual,
#     (
#         toolbox.attr_fast_period,
#         toolbox.attr_slow_period,
#         toolbox.attr_signal_period,
#         toolbox.attr_buy_stop_adjust
#     ),
# )
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# mean = np.ndarray(NGEN)
# best = np.ndarray(NGEN)
# hall_of_fame = tools.HallOfFame(maxsize=3)

# t = time.perf_counter()
# pop = toolbox.population(n=NPOP)
# for g in trange(NGEN):
#     # Select the next generation individuals
#     offspring = toolbox.select(pop, len(pop))
#     # Clone the selected individuals
#     offspring = list(map(toolbox.clone, offspring))

#     # Apply crossover on the offspring
#     for child1, child2 in zip(offspring[::2], offspring[1::2]):
#         if random.random() < CXPB:
#             toolbox.mate(child1, child2)
#             del child1.fitness.values
#             del child2.fitness.values

#     # Apply mutation on the offspring
#     for mutant in offspring:
#         if random.random() < MUTPB:
#             toolbox.mutate(mutant)
#             del mutant.fitness.values

#     # Evaluate the individuals with an invalid fitness
#     invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#     fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
#     for ind, fit in zip(invalid_ind, fitnesses):
#         ind.fitness.values = fit

#     # The population is entirely replaced by the offspring
#     pop[:] = offspring
#     hall_of_fame.update(pop)
#     print(
#         "HALL OF FAME:\n"
#         + "\n".join(
#             [
#                 f"    {_}: {ind}, Fitness: {ind.fitness.values[0]}"
#                 for _, ind in enumerate(hall_of_fame)
#             ]
#         )
#     )

#     fitnesses = [
#         ind.fitness.values[0] for ind in pop if not np.isinf(ind.fitness.values[0])
#     ]
#     mean[g] = np.mean(fitnesses)
#     best[g] = np.max(fitnesses)

# end_t = time.perf_counter()
# print(f"Time Elapsed: {end_t - t:,.2f}")

# fig, ax = plt.subplots(sharex=True, figsize=(16, 9))

# sns.lineplot(x=range(NGEN), y=mean, ax=ax, label="Average Fitness Score")
# sns.lineplot(x=range(NGEN), y=best, ax=ax, label="Best Fitness Score")
# ax.set_title("Fitness Score")
# ax.set_xticks(range(NGEN))
# ax.set_xlabel("Iteration")

# plt.tight_layout()
# plt.show()
