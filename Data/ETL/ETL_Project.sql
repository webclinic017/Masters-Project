USE ProjectStagging 

BULK INSERT [dbo].[ONE_MINUTEDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\1_min_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=25000000000000,
    MAXERRORS=2);

BULK INSERT [dbo].[FIVE_MINUTEDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\5_min_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=2500000000000,
    MAXERRORS=2);

BULK INSERT [dbo].[FIFTEEN_MINUTEDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\15_min_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=2500000000000000000,
    MAXERRORS=2);

BULK INSERT [dbo].[THIRTY_MINUTEDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\30_min_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=2500000000000000,
    MAXERRORS=2);

BULK INSERT [dbo].[ONE_HOURDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\1_hour_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=2500000000000,
    MAXERRORS=2);


BULK INSERT [dbo].[TWO_HOURDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\2_hour_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=2500000000000,
    MAXERRORS=2);

BULK INSERT [dbo].[FOUR_HOURDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\4_hour_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=2500000000000000,
    MAXERRORS=2);


BULK INSERT [dbo].[FIVE_HOURDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\5_hour_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=25000000000000,
    MAXERRORS=2);

BULK INSERT [dbo].[EIGHT_HOURDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\12_hour_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=250000000000000,
    MAXERRORS=2);

BULK INSERT [dbo].[TEN_HOURDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\10_hour_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=2500000000,
    MAXERRORS=2);


BULK INSERT [dbo].[TWELVE_HOURDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\12_hour_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=2500000000000,
    MAXERRORS=2);

BULK INSERT [dbo].[ONE_DAYDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\1_day_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=25000000000,
    MAXERRORS=2);




-----------
BULK INSERT [dbo].[ONE_WEEKDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\1_week_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=2500000000,
    MAXERRORS=2);

BULK INSERT [dbo].[ONE_MONTHDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\1_month_output.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=250000,
    MAXERRORS=2);




	---------------
BULK INSERT [dbo].[InstrumentDim]
FROM 'C:\Users\cex\.spyder-py3\Algo_DataClean\Output\instrument_info.csv'
WITH (FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR='\n',
    BATCHSIZE=250000,
    MAXERRORS=2);
