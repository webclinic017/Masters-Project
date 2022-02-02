#Increasing the accuracy of predictions using logistic regression. 63% increased to 80% for long positions
import backtrader as bt
from datetime import datetime
from collections import OrderedDict
import pandas as pd
import pyodbc
import statsmodels.api as sm
import numpy as np
import Project_Functions as pf

stmt_1d = "SELECT [Datetime],[ONE_DAYKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[ONE_DAYDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
conn3 = pyodbc.connect("Driver={SQL Server Native Client 11.0};Server=DESKTOP-QVFMQUE\SQLEXPRESS;DATABASE=ProjectWarehouse;Trusted_Connection=yes")
stmt_8h = "SELECT [Datetime],[EIGHT_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].EIGHT_HOURDim WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
hour_8 = pd.read_sql(stmt_8h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()
stmt_5h = "SELECT [Datetime],[FIVE_HOURKEY],[Ticker],[Open],[Low],[High],[Close],[Volume]FROM[ProjectWarehouse].[dbo].[FIVE_HOURDim] WHERE Ticker = 'G'ORDER BY [Datetime] ASC"
hour_5 = pd.read_sql(stmt_5h,conn3).rename(columns={"Datetime" : "Datetime"}).set_index("Datetime").dropna()


#df_1d = pd.read_sql(stmt_1d,conn3)[["Datetime", "Open", "High", "Low", "Close", "Volume"]].dropna().set_index("Datetime")
df_1d = pd.read_sql(stmt_1d,conn3).dropna().set_index("Datetime")

ma = 22

class firstStrategy(bt.Strategy):
    params = dict(period = ma , devfactor= 1.0)
    
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
       
       
    def next(self):
            if not self.position:
                if self.buy_signal > 0:
                    size = int(self.broker.getcash() / self.datas[0].open)
                    self.log(f'BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]}')
                    self.buy(size=size)
            else: 
                if self.sell_signal < 0:
                    self.log(f'SELL CREATED --- Size: {self.position.size}')
                    self.sell(size=self.position.size)
        
#Class object to extract the trade list 
class trade_list(bt.Analyzer):

    def get_analysis(self):

        return self.trades


    def __init__(self):

        self.trades = []
        self.cumprofit = 0.0


    def notify_trade(self, trade):

        if trade.isclosed:

            brokervalue = self.strategy.broker.getvalue()

            dir = 'short'
            if trade.history[0].event.size > 0: dir = 'long'

            pricein = trade.history[len(trade.history)-1].status.price
            priceout = trade.history[len(trade.history)-1].event.price
            datein = bt.num2date(trade.history[0].status.dt)
            dateout = bt.num2date(trade.history[len(trade.history)-1].status.dt)
            if trade.data._timeframe >= bt.TimeFrame.Days:
                datein = datein.date()
                dateout = dateout.date()

            pcntchange = 100 * priceout / pricein - 100
            pnl = trade.history[len(trade.history)-1].status.pnlcomm
            pnlpcnt = 100 * pnl / brokervalue
            barlen = trade.history[len(trade.history)-1].status.barlen
            pbar = pnl / barlen
            self.cumprofit += pnl

            size = value = 0.0
            for record in trade.history:
                if abs(size) < abs(record.status.size):
                    size = record.status.size
                    value = record.status.value

            highest_in_trade = max(trade.data.high.get(ago=0, size=barlen+1))
            lowest_in_trade = min(trade.data.low.get(ago=0, size=barlen+1))
            hp = 100 * (highest_in_trade - pricein) / pricein
            lp = 100 * (lowest_in_trade - pricein) / pricein
            if dir == 'long':
                mfe = hp
                mae = lp
            if dir == 'short':
                mfe = -lp
                mae = -hp

            # self.trades.append({'ref': trade.ref, 'ticker': trade.data._name, 'dir': dir,
            #      'datein': datein, 'pricein': pricein, 'dateout': dateout, 'priceout': priceout,
            #      'chng%': round(pcntchange, 2), 'pnl': pnl, 'pnl%': round(pnlpcnt, 2),
            #      'size': size, 'value': value, 'cumpnl': self.cumprofit,
            #      'nbars': barlen, 'pnl/bar': round(pbar, 2),
            #      'mfe%': round(mfe, 2), 'mae%': round(mae, 2)})
            
            self.trades.append([datein,pnl])


#Extract trade parameters 
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
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name = "trades")
cerebro.addanalyzer(trade_list, _name='trade_list')


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
#print('Transactions:', firstStrat.analyzers.transact.get_analysis())
print('Annual Return:', firstStrat.analyzers.annualret.get_analysis())

cerebro.plot(iplot=True, volume=False)

dat_transact = firstStrat.analyzers.transact.get_analysis()

dat_annualret = firstStrat.analyzers.annualret.get_analysis()
dat_drawdown = firstStrat.analyzers.drawdown.get_analysis()['max']['drawdown']



dat_transact = pd.DataFrame.from_dict(dat_transact )
dat_transposed = dat_transact.T
y = pd.DataFrame(dat_transposed[dat_transposed.columns[0]].tolist(),  columns=['amount', 'price', 'sid', 'symbol', 'value'])
dat_transposed.reset_index()
dat_transposed["amount"]= y["amount"]
dat_transposed["value"]= y["value"]
dat_transposed["Position"] = [1 if i > 0 else -1 for i in dat_transposed["amount"]]
dat_transposed["Strike"] =  [1 if i > 0 else -1 for i in dat_transposed["value"]]


dat_transposed = dat_transact.T
#Finally plot the end results
cerebro.plot(style='candlestick')





from tabulate import tabulate

           
trade_list = firstStrat.analyzers.trade_list.get_analysis()
dat_trade = pd.DataFrame(trade_list, columns=['Date', 'pnl'] )
dat_trade["y"] = [1 if i > 0 else 0 for i in dat_trade["pnl"]]
trade_dat = pd.DataFrame()
trade_dat = dat_trade[["Date","y"]]

#print (tabulate(trade_list, headers="keys"))  


#Volume Weighted Average Price

dat_mr = df_1d
dat_mr['ma'] = ((((dat_mr.High + dat_mr.Close + dat_mr.Low)/3)*dat_mr.Volume).rolling(ma).mean())/(dat_mr.Volume.rolling(ma).mean())
ratio = pd.DataFrame()
ratio["ratio"] = dat_mr['Close'] / dat_mr['ma'].shift(1)
ratio = ratio.reset_index()
ratio["Date"] = pd.to_datetime(ratio["Datetime"]).dt.date
rat = ratio[["Date", "ratio"]]


#Markov Regime Switching
dat_mr["Change"] = dat_mr[["Close"]].pct_change()
mod_parameter = sm.tsa.MarkovRegression(dat_mr.Change.dropna(), k_regimes=2, trend='c', switching_variance=True)
mod = mod_parameter.fit()
prob = pd.DataFrame()
prob["prob"] = mod.smoothed_marginal_probabilities[0].shift(1)
prob = prob.reset_index()
prob["Date"] = pd.to_datetime(prob["Datetime"]).dt.date
mark = prob[["Date","prob"]]

#TimeSeries Momentum
def tsmom(df, window = 1):
    df = df.copy()
    df['ret']= np.log(df.Close.pct_change()+1)
    df['prior_n']= df.ret.rolling(window).sum().shift(1)
    df.dropna(inplace=True)
    df['position']= [1 if i > 0 else -1 for i in df.prior_n]
    df['strat'] = df.position.shift(1)*df.ret

    
    #return np.exp(df[['ret', 'strat']].cumsum()).plot(figsize=(12,5))
    return df[['ret','prior_n','position']]

ts = tsmom(dat_mr, window=ma)
ts = ts.reset_index()
ts["Date"] = pd.to_datetime(ts["Datetime"]).dt.date
tsm = ts[["Date", "prior_n"]]


#YangZhang
yan = pd.DataFrame()
yan["vol"] = pf.yangzhang(dat_mr,  window=ma, trading_periods=252, clean=True)
yan = yan.reset_index()
yan["Date"] = pd.to_datetime(yan["Datetime"]).dt.date
yanz = yan[["Date","vol"]]


#VolCrossing
def volcrossing(price_data, short = 20, long = 30 ):
    
    price_data = price_data.copy()
    short = 20
    long = 30
    price_data["short_vol"] = pf.yangzhang(price_data, window=short)
    price_data["long_vol"] = pf.yangzhang(price_data, window=long)
    price_data = price_data.dropna()
    
    price_data['Vol_Signal']= np.where(price_data['short_vol'] > price_data['long_vol'],1,-1)
    price_data['volval'] =  price_data.short_vol - price_data.long_vol.shift(1) 
    
    price_data['Vol_Signal'] = price_data['Vol_Signal'].shift(1)
    
    return price_data[["volval",'Vol_Signal']]

volpd = pd.DataFrame()
volpd= volcrossing(dat_mr, short = ma-5, long = ma)
volpd = volpd.reset_index()
volpd["Date"] = pd.to_datetime(volpd["Datetime"]).dt.date
volf = volpd[["Date","volval"]]


abc = [0,1,2,3,4,5]
a = trade_dat
b = rat
c = mark
d = tsm
e = volf
f = yanz
my_dict = {str(abc[0]): a, str(abc[1]):b , str(abc[2]): c, str(abc[3]): d, str(abc[4]): e, str(abc[5]): f }

def data_merge(**my_dict):
    ref_data = my_dict['0']
    ref_data = ref_data.merge(my_dict['1'], on = my_dict['1'].columns[0], how='left')
    ref_data = ref_data.merge(my_dict['2'], on = my_dict['2'].columns[0], how='left')
    ref_data = ref_data.merge(my_dict['3'], on = my_dict['3'].columns[0], how='left')
    ref_data = ref_data.merge(my_dict['4'], on = my_dict['4'].columns[0], how='left')
    ref_data = ref_data.merge(my_dict['5'], on = my_dict['5'].columns[0], how='left')
    
    # ref_data = ref_data.set_index("Datetime")
    # ref_data = ref_data.iloc[:,15:]
    # ref_data = my_dict['1'].merge(ref_data, left_index=True, right_index=True, how='inner').iloc[:,2:]
    # ref_data["Entry"] = ref_data[ref_data.columns[0]] + ref_data[ref_data.columns[1]] + ref_data[ref_data.columns[2]] + ref_data[ref_data.columns[3]]
    # ref_data["Entry"] = np.where(ref_data["Entry"] == 4 , 1,0)
    
    return ref_data

y = data_merge(**my_dict)
#########################################################



import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#data = y[["y","ratio","prob", "prior_n","volval","vol" ]]
data = y.set_index("Date").dropna()

data['y'].value_counts()
sns.countplot(x="y", data=data, palette="hls")
plt.show()
plt.savefig("Count_plot")



count_no_sub = len(data[data['y']==0])
count_sub = len(data[data['y']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of losing trades", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of winning trades", pct_of_sub*100)


data.ratio.hist()
plt.title('Histogram of moving average ratio')
plt.xlabel('ratio')
plt.ylabel('Frequency')
plt.savefig('Ratio_Frenquency')

data.prob.hist()
plt.title('Histogram of Low Regime Probability')
plt.xlabel('Markov_Probability')
plt.ylabel('Frequency')
plt.savefig('Regime_Frenquency')

data.volval.hist()
plt.title('Histogram of VolCrossing Difference')
plt.xlabel('Vol Difference')
plt.ylabel('Frequency')
plt.savefig('Vol_Cross')

data.prior_n.hist()
plt.title('Histogram of Time Series Momentum')
plt.xlabel('TSMOM')
plt.ylabel('Frequency')
plt.savefig('TSMOM_Hist')

data.vol.hist()
plt.title('Histogram of YangZhang Volatility Values')
plt.xlabel('YangZhang')
plt.ylabel('Frequency')
plt.savefig('YZ_Hist')

X = data.loc[:, data.columns != 'y']
y = data.loc[:, data.columns == 'y']


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


score = logreg.score(X_test, y_test)
print(score)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('Confusion_Matrix')
plt.show()


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.savefig('Log_ROC2')
plt.show()



#data.to_csv("ML_Data.csv")













