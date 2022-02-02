#Resampling Data into multiple frequencies 
import pandas as pd
import glob
import os
import io

file_list = glob.glob(os.path.join(os.getcwd(), "1_hour", "*.txt")) #ALTER SOURCE FOLDER HERE

corpus = [open(file).read() for file in file_list]

symbol = []
with open("ticker_sym.txt") as f:
    for line in f:
        symbol.append(line.strip())
f.close

agg_dict = {'Open': 'first',
          'High': 'max',
          'Low': 'min',
          'Close': 'last',
          'Volume': 'sum'}

i=0 
j=0  

#freq = ["10H"] #ALTER RESAMPLING FREQUENCY HERE
#file_path = ["Output/10_hour_output.csv"] #ALTERFILE HERE
#file_path_1 = ["Output/10_hour_output.csv"] #ALTERFILE HERE

while i<len(symbol):
    try:
        print ("STEP 1/15: Inserting Data into Global DataFrame", symbol[i], i+1,"/",len(symbol))
        r_df = pd.read_csv(io.StringIO(corpus[i]), index_col=0, sep=',', names=['Open','High', 'Low','Close','Volume'], parse_dates=True)
        print ("STEP 2/15: Data insertion into Global DataFrame completed", symbol[i], i+1,"/",len(symbol))
        print ("STEP 3/15: Initiating Data Resampling", symbol[i], i+1,"/",len(symbol))
        df = r_df.resample('M').agg(agg_dict) #ALTER RESAMPLING FREQUENCY HERE
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
 
        if i == 0:
            
            df.to_csv("Output/1_month_output.csv") #ALTERFILE HERE
            print ("STEP 15/15: Export Complete. Data Cleaning for", symbol[i], i+1,"/",len(symbol), "is Successful")
        else:
            df.to_csv("Output/1_month_output.csv",mode = "a",header=False) #ALTERFILE HERE
        j=0    
    except:
        print ("from except", i,j, symbol[i])
        if j <=9:
            print(i, symbol[i], j,"Eligible for retry")
            j = j+1
            continue
        if j == 10:
            j=0
            i=i+1
            continue
    i=i+1
