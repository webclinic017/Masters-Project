import backtrader as bt
import datetime
import pandas as pd
import pyodbc

class BBand_Strategy(bt.Strategy):
    params = dict(period = 20, devfactor= 2.0)

    def __init__(self):
        # keep track of close price in the series
        self.data_close = self.datas[0].close
        self.data_open = self.datas[0].open

        # keep track of pending orders/buy price/buy commission
        self.order = None
        self.price = None
        self.comm = None

        # add Bollinger Bands indicator and track the buy/sell signals
        self.b_band = bt.ind.BollingerBands(self.datas[0], 
                                            period=self.p.period, 
                                            devfactor=self.p.devfactor)
        self.buy_signal = bt.ind.CrossOver(self.datas[0], 
                                           self.b_band.lines.bot)
        self.sell_signal = bt.ind.CrossOver(self.datas[0], 
                                            self.b_band.lines.top)
        
    def log(self, txt):
            dt = self.datas[0].datetime.date(0).isoformat()
            print(f'{dt}, {txt}')

    def notify_order(self, order):
       if order.status in [order.Submitted, order.Accepted]:
           return
       
       if order.status in [order.Completed]:
           if order.isbuy():
               self.log(
                   f'BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}'
               )
               self.price = order.executed.price
               self.comm = order.executed.comm
           else:
               self.log(
                   f'SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}'
               )
       
       elif order.status in [order.Canceled, order.Margin, 
                             order.Rejected]:
           self.log('Order Failed')
    
       self.order = None
            
            
    def notify_trade(self, trade):
            if not trade.isclosed:
                return
    
            self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')

    def next_open(self):
            if not self.position:
                if self.buy_signal > 0:
                    size = int(self.broker.getcash() / self.datas[0].open)
                    self.log(f'BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]}')
                    self.buy(size=size)
            else: 
                if self.sell_signal < 0:
                    self.log(f'SELL CREATED --- Size: {self.position.size}')
                    self.sell(size=self.position.size)


# Add the data to Cerebro
conn3 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectWarehouse;Trusted_Connection=yes")
stmt_1d = "SELECT [Datetime],[ONE_DAYKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_DAYDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
day_1 = pd.read_sql(stmt_1d,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()


data = bt.feeds.PandasData(dataname=day_1)
cerebro = bt.Cerebro(stdstats = False, cheat_on_open=True)

cerebro.addstrategy(BBand_Strategy)
cerebro.adddata(data)
cerebro.broker.setcash(100000.0)
cerebro.broker.setcommission(commission=0.001)
cerebro.addobserver(bt.observers.Trades)
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
backtest_result = cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.plot(iplot=True, volume=False)
        
print(backtest_result[0].analyzers.returns.get_analysis())

returns_dict = backtest_result[0].analyzers.time_return.get_analysis()
returns_df = pd.DataFrame(list(returns_dict.items()), 
                          columns = ['report_date', 'return']) \
               .set_index('report_date')
               
returns_df["return"].plot(title='Portfolio returns')
