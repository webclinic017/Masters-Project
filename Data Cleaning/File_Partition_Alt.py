#Partitioning 1 minute data into several files
import pandas as pd

filename = 'Output/1_min_output.csv'

#get the number of lines of the csv file to be read
number_lines = sum(1 for row in (open(filename)))
print("Number of Rows: ", number_lines)
#size of rows of data to write to the csv, 
#you can change the row size according to your need
sets = 4
rowsize = int(number_lines/sets)
print("Number of Sets: ",sets)
print("Number of Rows per Set: ",rowsize)


chunksize = rowsize

for i, chunk in enumerate(pd.read_csv(filename, 
          chunksize=chunksize,
          index_col=0, sep=',', 
          names=['ID','Ticker','Open','High', 'Low','Close','Volume'], 
          parse_dates=True,
          skiprows = 1)):                      
  chunk.to_csv('Output/Partition' + str(i) + '.csv', header=['ID','Ticker','Open','High', 'Low','Close','Volume'])
  
