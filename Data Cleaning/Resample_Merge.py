#Resampling Draft
import pandas as pd

agg_dict = {'Open': 'first',
          'High': 'max',
          'Low': 'min',
          'Close': 'last',
          'Volume': 'sum'}

i=0 
j=0  

symbol=["A6"]


r_df = pd.read_csv("1_hour/A6_continuous_adjusted_1hour.txt", index_col=0, sep=',', names=['Open','High', 'Low','Close','Volume'], parse_dates=True)
print ("STEP 2/15: Data insertion into Global DataFrame completed", symbol[i], i+1,"/",len(symbol))
print ("STEP 3/15: Initiating Data Resampling", symbol[i], i+1,"/",len(symbol))
df = r_df.resample('D').agg(agg_dict) #ALTER RESAMPLING FREQUENCY HERE


samp = df[:20]


print ("STEP 4/15: Data Resampling completed", symbol[i], i+1,"/",len(symbol))
print ("STEP 5/15: Insertion of Ticker Symbol", symbol[i], i+1,"/",len(symbol))
df.insert(0,"Ticker",symbol[i])
print ("STEP 6/15: Ticker Symbol Insertion Complete", symbol[i], i+1,"/",len(symbol))
print("STEP 7/15: Datetime Column Extraction", symbol[i], i+1,"/",len(symbol))
data_datetime = df.index.strftime("%m/%d/%Y %H:%M:%S")
print("STEP 8/15: Datetime Column Extraction Complete", symbol[i], i+1,"/",len(symbol))
print("STEP 9/15: Ticker Column Extraction", symbol[i], i+1,"/",len(symbol))
data_ticker = df["Ticker"].copy()
print("STEP 10/15: Datetime Column Extraction Complete", symbol[i], i+1,"/",len(symbol))
print("STEP 11/15: Creating Primary Key/ID", symbol[i], i+1,"/",len(symbol))
prim_key = data_ticker.str.cat(data_datetime, sep='_', na_rep= "")
print("STEP 12/15: Primary Key/ID has been created", symbol[i], i+1,"/",len(symbol))
print("STEP 13/15: Inserting Primary Key/ID into DataFrame", symbol[i], i+1,"/",len(symbol))
df.insert(0,"ID",prim_key)
print ("STEP 14/15: Exporting Data to External file", symbol[i], i+1,"/",len(symbol))
