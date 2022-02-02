#Backtest Sample to extract information from cerebro engine 
import backtrader as bt
from datetime import datetime
from collections import OrderedDict
import pandas as pd
import pyodbc


stmt_1d = "SELECT [Datetime],[ONE_DAYKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_DAYDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
conn3 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectWarehouse;Trusted_Connection=yes")

#df_1d = pd.read_sql(stmt_1d,conn3)[["Datetime", "Open", "High", "Low", "Close", "Volume"]].dropna().set_index("Datetime")
df_1d = pd.read_sql(stmt_1d,conn3).dropna().set_index("Datetime")

class firstStrategy(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(fast_period=44, slow_period=121, signal_period=22)

        
    def __init__(self):

        self.fast_ma = bt.indicators.EMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.EMA(self.data.close, period=self.p.slow_period)
        self.macd_line = self.fast_ma - self.slow_ma
        self.signal_line = bt.indicators.EMA(self.macd_line, period=self.p.signal_period)
        self.macd_crossover = bt.indicators.CrossOver(self.macd_line, self.signal_line)

    def next(self):

        if self.macd_crossover > 0:
            self.buy(size=100)  # enter long position
        elif self.macd_crossover < 0:
            self.close()  # close long position
            
        


def printTradeAnalysis(analyzer):
    '''
    Function to print the Technical Analysis results in a nice format.
    '''
    #Get the results we are interested in
    total_open = analyzer.total.open
    total_closed = analyzer.total.closed
    total_won = analyzer.won.total
    total_lost = analyzer.lost.total
    win_streak = analyzer.streak.won.longest
    lose_streak = analyzer.streak.lost.longest
    pnl_net = round(analyzer.pnl.net.total,2)
    strike_rate = round((total_won / total_closed) * 100,2)
    #Designate the rows
    h1 = ['Total Open', 'Total Closed', 'Total Won', 'Total Lost']
    h2 = ['Strike Rate','Win Streak', 'Losing Streak', 'PnL Net']
    r1 = [total_open, total_closed,total_won,total_lost]
    r2 = [strike_rate, win_streak, lose_streak, pnl_net]
    #Check which set of headers is the longest.
    if len(h1) > len(h2):
        header_length = len(h1)
    else:
        header_length = len(h2)
    #Print the rows
    print_list = [h1,r1,h2,r2]
    row_format ="{:<15}" * (header_length + 1)
    print("Trade Analysis Results:")
    for row in print_list:
        print(row_format.format('',*row))

def printSQN(analyzer):
    sqn = round(analyzer.sqn,2)
    print('SQN: {}'.format(sqn))
    
# def printSharpeRatio(analyzer):
#     sr = round(analyzer.sr,2)
#     print('SharpeRatio: {}'.format(sr))
    

#Variable for our starting cash
startcash = 100000.00

#Create an instance of cerebro
cerebro = bt.Cerebro(tradehistory=True)

#Add our strategy
cerebro.addstrategy(firstStrategy)

feed = bt.feeds.PandasData(dataname=df_1d)
cerebro.adddata(feed)

# Set our desired cash start
cerebro.broker.setcash(startcash)

# Add the analyzers we are interested in
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="mysharpe")
cerebro.addanalyzer(bt.analyzers.Transactions, _name="transact")
cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annualret")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")


# Run over everything
strategies = cerebro.run()
firstStrat = strategies[0]

# print the analyzers
printTradeAnalysis(firstStrat.analyzers.ta.get_analysis())
printSQN(firstStrat.analyzers.sqn.get_analysis())
#printSharpeRatio(firstStrat.analyzers.sr.get_analysis())


#Get final portfolio Value
portvalue = cerebro.broker.getvalue()

#Print out the final result
print('Final Portfolio Value: ${}'.format(portvalue))
print('Sharpe Ratio:', firstStrat.analyzers.mysharpe.get_analysis())
print('Transactions:', firstStrat.analyzers.transact.get_analysis())
print('Annual Return:', firstStrat.analyzers.annualret.get_analysis())

dat_transact = firstStrat.analyzers.transact.get_analysis()
dat_annualret = firstStrat.analyzers.annualret.get_analysis()
dat_drawdown = firstStrat.analyzers.drawdown.get_analysis()['max']['drawdown']

dat_transact = pd.DataFrame.from_dict(dat_transact)

#Finally plot the end results
cerebro.plot(style='candlestick')