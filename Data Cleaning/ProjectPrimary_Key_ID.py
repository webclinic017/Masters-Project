#Primary key creation draft
print("STEP 1/17: Importing Pandas... ")
import pandas as pd
print("STEP 2/17: Pandas imported successfully. ")
print("STEP 3/17: Importing Data into python environment (DataFrame)... ")
data = pd.read_csv("1_hour/A6_continuous_adjusted_1hour.txt", index_col=0, sep=',', names=['Open','High', 'Low','Close','Volume'], parse_dates=True) #Alter File Path Here
print("STEP 4/17: Data Loaded into memory succesfulliy.")
data.insert(0,"Ticker","A6")
print("STEP 5/17: Datetime Column Extraction..")
data_datetime = data.index.strftime("%m/%d/%Y %H:%M:%S")
print("STEP 6/17: Datetime Column Extraction complete.")
print("STEP 7/17: Ticker Column Extraction...")
data_ticker = data["Ticker"].copy()
print("STEP 8/17: Ticker Column Extraction complete.")
print("STEP 9/17: Creating Primary Key/ID... ")
prim_key = data_ticker.str.cat(data_datetime, sep='_', na_rep= "")
print("STEP 10/17: Primary Key/ID has been created.")
print("STEP 11/17: Inserting Primary Key/ID into DataFrame .. ")
data.insert(0,"ID",prim_key)
print("STEP 12/17: Primary Key/ID insertion completed")
print("STEP 13/17: Transfer to Local Disk C...")
data.to_csv("1_min_output.csv") #Alter File Path Here
print("STEP 14/17: Transfer to Local Disk C Completed")
print("STEP 15/17: Creating sample DataFrame.. ")
sample = data.iloc[:20]
print("STEP 16/17: DataFrame Sample Created. ")
print("STEP 17/17: END: CODE SUCCESS ")
