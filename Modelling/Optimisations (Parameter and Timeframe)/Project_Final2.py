#Time frame and parameter Optimisation - 16,000 seconds / 4.5 hour completion 
import numpy as np
import pandas as pd
import math
import pandas as pd
import pyodbc

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



class BBand_Strategy(bt.Strategy):
    params = dict(dat = 7, period = 20, devfactor= 2.0)

    def __init__(self):
        # keep track of close price in the series
        self.data_close = self.datas[self.p.dat].close
        self.data_open = self.datas[self.p.dat].open

        # keep track of pending orders/buy price/buy commission
        self.order = None
        self.price = None
        self.comm = None

        # add Bollinger Bands indicator and track the buy/sell signals
        self.b_band = bt.ind.BollingerBands(self.datas[self.p.dat], 
                                            period=self.p.period, 
                                            devfactor=self.p.devfactor)
        self.buy_signal = bt.ind.CrossOver(self.datas[self.p.dat], 
                                           self.b_band.lines.bot)
        self.sell_signal = bt.ind.CrossOver(self.datas[self.p.dat], 
                                            self.b_band.lines.top)
        
    # def log(self, txt):
    #         dt = self.datas[self.p.dat].datetime.date(0).isoformat()
    #         print(f'{dt}, {txt}')

    # def notify_order(self, order):
    #     if order.status in [order.Submitted, order.Accepted]:
    #         return
       
    #     if order.status in [order.Completed]:
    #         if order.isbuy():
    #             self.log(
    #                 f'BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}'
    #             )
    #             self.price = order.executed.price
    #             self.comm = order.executed.comm
    #         else:
    #             self.log(
    #                 f'SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}'
    #             )
       
    #     elif order.status in [order.Canceled, order.Margin, 
    #                           order.Rejected]:
    #         self.log('Order Failed')
    
    #     self.order = None
            
            
    # def notify_trade(self, trade):
    #         if not trade.isclosed:
    #             return
    
    #         self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')

    def next(self):
            if not self.position:
                if self.buy_signal > 0:
                    size = int(self.broker.getcash() / self.datas[self.p.dat].open)
                    #self.log(f'BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]}')
                    self.buy(size=size)
            else: 
                if self.sell_signal < 0:
                    #self.log(f'SELL CREATED --- Size: {self.position.size}')
                    self.sell(size=self.position.size)




STRATEGY_PARAMS = dict(dat = 7, period = 20, devfactor= 2.0)

conn2 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectPartition_one;Trusted_Connection=yes")
conn3 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectWarehouse;Trusted_Connection=yes")


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


# min_1 = pd.read_sql(stmt_1m,conn2).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# min_5 = pd.read_sql(stmt_5m,conn3).rename(columns={"Datetime_5min" : "Datetime"}).set_index("Datetime").dropna()
# min_15 = pd.read_sql(stmt_15m,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# min_30 = pd.read_sql(stmt_30m,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_1 = pd.read_sql(stmt_1h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_2 = pd.read_sql(stmt_2h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_4 = pd.read_sql(stmt_4h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_5 = pd.read_sql(stmt_5h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_8 = pd.read_sql(stmt_8h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_10 = pd.read_sql(stmt_10h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_12 = pd.read_sql(stmt_12h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
day_1 = pd.read_sql(stmt_1d,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# week_1 = pd.read_sql(stmt_1w,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# month_1 = pd.read_sql(stmt_1mon,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()    
  
    
def run_backtest(plot=True, **strategy_params):

    cerebro = bt.Cerebro(stdstats = False, cheat_on_open=True)
    cerebro.addstrategy(BBand_Strategy)


    df_1hour = bt.feeds.PandasData(dataname=hour_1)
    cerebro.adddata(df_1hour , name = 'd5')
    
    df_2hour = bt.feeds.PandasData(dataname=hour_2)
    cerebro.adddata(df_2hour , name = 'd6')
    
    df_4hour = bt.feeds.PandasData(dataname=hour_4)
    cerebro.adddata(df_4hour , name = 'd7')
    
    df_5hour = bt.feeds.PandasData(dataname=hour_5)
    cerebro.adddata(df_5hour , name = 'd8')
    
    df_8hour = bt.feeds.PandasData(dataname=hour_8)
    cerebro.adddata(df_8hour , name = 'd8')   
    
    df_10hour = bt.feeds.PandasData(dataname=hour_10)
    cerebro.adddata(df_10hour , name = 'd10')
     
    df_12hour = bt.feeds.PandasData(dataname=hour_12)
    cerebro.adddata(df_12hour , name = 'd11')
    
    df_1d = bt.feeds.PandasData(dataname=day_1)
    cerebro.adddata(df_1d , name = 'd12')
    
    # Remember to set it high enough or the strategy may not
    # be able to trade because of short of cash

    
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
    

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    backtest_result = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())


    if plot:
        cerebro.plot(iplot=True, volume=False)
        
        
#run_backtest(plot=True, **STRATEGY_PARAMS)


# fix the seed so that we will get the same results
# feel free to change it or comment out the line
random.seed(1)

# GA parameters
PARAM_NAMES = ["dat", "period", "devfactor"]
NGEN = 10
NPOP = 100
CXPB = 0.5
MUTPB = 0.3

conn2 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectPartition_one;Trusted_Connection=yes")
conn3 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectWarehouse;Trusted_Connection=yes")


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


# min_1 = pd.read_sql(stmt_1m,conn2).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# min_5 = pd.read_sql(stmt_5m,conn3).rename(columns={"Datetime_5min" : "Datetime"}).set_index("Datetime").dropna()
# min_15 = pd.read_sql(stmt_15m,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# min_30 = pd.read_sql(stmt_30m,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_1 = pd.read_sql(stmt_1h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_2 = pd.read_sql(stmt_2h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_4 = pd.read_sql(stmt_4h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_5 = pd.read_sql(stmt_5h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_8 = pd.read_sql(stmt_8h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_10 = pd.read_sql(stmt_10h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
hour_12 = pd.read_sql(stmt_12h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
day_1 = pd.read_sql(stmt_1d,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# week_1 = pd.read_sql(stmt_1w,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
# month_1 = pd.read_sql(stmt_1mon,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()    
  

def evaluate(individual, plot=False, log=False):

    # convert list of parameter values into dictionary of kwargs
    strategy_params = {k: v for k, v in zip(PARAM_NAMES, individual)}

    # fast moving average by definition cannot be slower than the slow one


    # by setting stdstats to False, backtrader will not store the changes in
    # statistics like number of trades, buys & sells, etc.
    cerebro = bt.Cerebro(stdstats = False, cheat_on_open=True)


    # df_1hour = bt.feeds.PandasData(dataname=hour_1)
    # cerebro.adddata(df_1hour , name = 'd5')
    
    # df_2hour = bt.feeds.PandasData(dataname=hour_2)
    # cerebro.adddata(df_2hour , name = 'd6')
    
    # df_4hour = bt.feeds.PandasData(dataname=hour_4)
    # cerebro.adddata(df_4hour , name = 'd7')
    
    df_5hour = bt.feeds.PandasData(dataname=hour_5)
    cerebro.adddata(df_5hour , name = 'd8')
    
    df_8hour = bt.feeds.PandasData(dataname=hour_8)
    cerebro.adddata(df_8hour , name = 'd8')   
    
    # df_10hour = bt.feeds.PandasData(dataname=hour_10)
    # cerebro.adddata(df_10hour , name = 'd10')
     
    df_12hour = bt.feeds.PandasData(dataname=hour_12)
    cerebro.adddata(df_12hour , name = 'd11')
    
    df_1d = bt.feeds.PandasData(dataname=day_1)
    cerebro.adddata(df_1d , name = 'd12')

    # Remember to set it high enough or the strategy may not
    # be able to trade because of short of cash
    initial_capital = 100000.0
    cerebro.broker.setcash(initial_capital)

    # Pass in the genes of the individual as kwargs
    cerebro.addstrategy(BBand_Strategy, **strategy_params)

    # This is needed for calculating our fitness score
    cerebro.addanalyzer(bt.analyzers.DrawDown)

    # Let's say that we have 0.25% slippage and commission per trade,
    # that is 0.5% in total for a round trip.
    cerebro.broker.setcommission(commission=0.0025, margin=False)

    # Run over everything
    strats = cerebro.run()

    profit = cerebro.broker.getvalue() - initial_capital
    max_dd = strats[0].analyzers.drawdown.get_analysis()["max"]["moneydown"]
    fitness = profit / (max_dd if max_dd > 0 else 1)

    if log:
        print(f"Starting Portfolio Value: {initial_capital:,.2f}")
        print(f"Final Portfolio Value:    {cerebro.broker.getvalue():,.2f}")
        print(f"Total Profit:             {profit:,.2f}")
        print(f"Maximum Drawdown:         {max_dd:,.2f}")
        print(f"Profit / Max DD:          {fitness}")

    if plot:
        cerebro.plot()

    return [fitness]


# our fitness score is supposed to be maximised and there is only 1 objective
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# our individual is a list of genes, with the fitness score the higher the better
creator.create("Individual", list, fitness=creator.FitnessMax)

# register some handy functions for calling
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(NPOP), NPOP)
# crossover strategy
toolbox.register("mate", tools.cxUniform, indpb=CXPB)
# mutation strategy

#toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.5, low=[0,10,1], up=[7,151,4], indpb=0.2)
toolbox.register("mutate", tools.mutUniformInt, low=[1,10,1], up=[3,151,3], indpb=0.2)
# selection strategy
toolbox.register("select", tools.selTournament, tournsize=3)
# fitness function
toolbox.register("evaluate", evaluate)

# definition of an individual & a population
toolbox.register("attr_dat", random.randint, 1, 3)
toolbox.register("attr_period", random.randint, 10, 151)
toolbox.register("attr_devfactor", random.randint, 1, 3)
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (
        toolbox.attr_dat,
        toolbox.attr_period,
        toolbox.attr_devfactor,
    ),
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

mean = np.ndarray(NGEN)
best = np.ndarray(NGEN)
hall_of_fame = tools.HallOfFame(maxsize=3)

t = time.perf_counter()
pop = toolbox.population(n=NPOP)
for g in trange(NGEN):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring
    hall_of_fame.update(pop)
    print(
        "HALL OF FAME:\n"
        + "\n".join(
            [
                f"    {_}: {ind}, Fitness: {ind.fitness.values[0]}"
                for _, ind in enumerate(hall_of_fame)
            ]
        )
    )

    fitnesses = [
        ind.fitness.values[0] for ind in pop if not np.isinf(ind.fitness.values[0])
    ]
    mean[g] = np.mean(fitnesses)
    best[g] = np.max(fitnesses)

end_t = time.perf_counter()
print(f"Time Elapsed: {end_t - t:,.2f}")

fig, ax = plt.subplots(sharex=True, figsize=(16, 9))

sns.lineplot(x=range(NGEN), y=mean, ax=ax, label="Average Fitness Score")
sns.lineplot(x=range(NGEN), y=best, ax=ax, label="Best Fitness Score")
ax.set_title("Fitness Score")
ax.set_xticks(range(NGEN))
ax.set_xlabel("Iteration")

plt.tight_layout()
plt.show()
