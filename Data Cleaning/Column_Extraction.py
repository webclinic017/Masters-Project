import pandas as pd
import glob
import os
import io

file_list = glob.glob(os.path.join(os.getcwd(), "1_min", "*.txt"))

corpus = [open(file).read() for file in file_list]

symbol = []
with open("ticker_sym.txt") as f:
    for line in f:
        symbol.append(line.strip())
f.close

i=0 
j=0  


while i<len(symbol):
    try:
        print ("STEP 1/13: Inserting Data into Global DataFrame", symbol[i], i+1,"/",len(symbol))
        df = pd.read_csv(io.StringIO(corpus[i]), 
                         index_col=0, sep=',', 
                         names=['Open','High', 'Low','Close','Volume'], 
                         parse_dates=True)
        print ("STEP 2/13: Data insertion into Global DataFrame completed", symbol[i], i+1,"/",len(symbol))
        print ("STEP 3/13: Insertion of Ticker Symbol", symbol[i], i+1,"/",len(symbol))
        df.insert(0,"Ticker",symbol[i])
        print ("STEP 4/13: Ticker Symbol Insertion Compled", symbol[i], i+1,"/",len(symbol))
        print("STEP 5/13: Datetime Column Extraction", symbol[i], i+1,"/",len(symbol))
        data_datetime = df.index.strftime("%m/%d/%Y %H:%M:%S")
        print("STEP 6/13: Datetime Column Extraction Complete", symbol[i], i+1,"/",len(symbol))
        print("STEP 7/13: Ticker Column Extraction", symbol[i], i+1,"/",len(symbol))
        data_ticker = df["Ticker"].copy()
        print("STEP 8/13: Datetime Column Extraction Complete", symbol[i], i+1,"/",len(symbol))
        print("STEP 9/13: Creating Primary Key/ID", symbol[i], i+1,"/",len(symbol))
        prim_key = data_ticker.str.cat(data_datetime, sep='_', na_rep= "")
        print("STEP 10/13: Primary Key/ID has been created", symbol[i], i+1,"/",len(symbol))
        print("STEP 11/13: Inserting Primary Key/ID into DataFrame", symbol[i], i+1,"/",len(symbol))
        df.insert(0,"ID",prim_key)
        df.drop(['Open','High', 'Low','Close','Volume'], axis=1, inplace=True)
        print ("STEP 12/13: Exporting Data to External file", symbol[i], i+1,"/",len(symbol))
 
        if i == 0:
            
            df.to_csv("Output/1_Min_Detail_output.csv")
            print ("STEP 13/13: Export Complete. Data Cleaning for", symbol[i], i+1,"/",len(symbol), "is Successful")
        else:
            df.to_csv("Output/1_Min_Detail_output.csv",mode = "a",header=False)
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
