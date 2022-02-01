 Use ProjectPK_one
 --drop table Gold_table

--create table Gold_table (
--Datetime_1min smalldatetime,
--Ticker_1min varchar(5),
--[ONE_MINUTEKEY] varchar(50),
--[FIVE_MINUTEKEY] varchar(50),
--[FIFTEEN_MINUTEKEY] varchar(50),
--[THIRTY_MINUTEKEY] varchar(50),  
--[ONE_HOURKEY] varchar(50),
--[TWO_HOURKEY] varchar(50),
--[FOUR_HOURKEY] varchar(50),
--[FIVE_HOURKEY] varchar(50),
--[TEN_HOURKEY] varchar(50),
--[EIGHT_HOURKEY] varchar(50),
--[TWELVE_HOURKEY] varchar(50),
--[ONE_DAYKEY] varchar(50),
--[ONE_WEEKKEY] varchar(50),
--[ONE_MONTHKEY] varchar(50),
--) ON [PRIMARY];
--GO

INSERT INTO Gold_table (Datetime_1min,Ticker_1min,[ONE_MINUTEKEY],FIVE_MINUTEKEY,FIFTEEN_MINUTEKEY, THIRTY_MINUTEKEY,
ONE_HOURKEY,TWO_HOURKEY, FOUR_HOURKEY,FIVE_HOURKEY, EIGHT_HOURKEY, TEN_HOURKEY,TWELVE_HOURKEY,
ONE_DAYKEY,ONE_WEEKKEY ,ONE_MONTHKEY )
SELECT Datetime_1min,Ticker_1min,[ONE_MINUTEKEY],FIVE_MINUTEKEY,FIFTEEN_MINUTEKEY, THIRTY_MINUTEKEY,
ONE_HOURKEY,TWO_HOURKEY, FOUR_HOURKEY,FIVE_HOURKEY, EIGHT_HOURKEY, TEN_HOURKEY,TWELVE_HOURKEY,
ONE_DAYKEY,ONE_WEEKKEY ,ONE_MONTHKEY 
FROM [ProjectPK_one].[dbo].[ONE_MINUTEDim] AS A
LEFT JOIN [ProjectWarehouse].[dbo].[FIVE_MINUTEDim] AS B on A.ONE_MINUTEKEY=B.FIVE_MINUTEKEY 
LEFT JOIN [ProjectWarehouse].[dbo].[FIFTEEN_MINUTEDim] AS C on A.ONE_MINUTEKEY=C.FIFTEEN_MINUTEKEY 
LEFT JOIN [ProjectWarehouse].[dbo].[THIRTY_MINUTEDim] AS D on A.ONE_MINUTEKEY=D.THIRTY_MINUTEKEY 
LEFT JOIN [ProjectWarehouse].[dbo].[ONE_HOURDim] AS E on A.ONE_MINUTEKEY=E.ONE_HOURKEY
LEFT JOIN [ProjectWarehouse].[dbo].[TWO_HOURDim] AS F on A.ONE_MINUTEKEY=F.TWO_HOURKEY
LEFT JOIN [ProjectWarehouse].[dbo].[FOUR_HOURDim] AS G on A.ONE_MINUTEKEY=G.FOUR_HOURKEY
LEFT JOIN [ProjectWarehouse].[dbo].[FIVE_HOURDim] AS H on A.ONE_MINUTEKEY=H.FIVE_HOURKEY
LEFT JOIN [ProjectWarehouse].[dbo].[EIGHT_HOURDim] AS I on A.ONE_MINUTEKEY=I.EIGHT_HOURKEY
LEFT JOIN [ProjectWarehouse].[dbo].[TEN_HOURDim] AS J on A.ONE_MINUTEKEY=J.TEN_HOURKEY
LEFT JOIN [ProjectWarehouse].[dbo].[TWELVE_HOURDim] AS K on A.ONE_MINUTEKEY=K.TWELVE_HOURKEY
LEFT JOIN [ProjectWarehouse].[dbo].[ONE_DAYDim] AS L on A.ONE_MINUTEKEY=L.ONE_DAYKEY 
LEFT JOIN [ProjectWarehouse].[dbo].[ONE_WEEKDim] AS M on A.ONE_MINUTEKEY=M.ONE_WEEKKEY 
LEFT JOIN [ProjectWarehouse].[dbo].[ONE_MONTHDim] AS N on A.ONE_MINUTEKEY=N.ONE_MONTHKEY 
WHERE Ticker_1min = 'G'
order by Datetime_1min ASC


UPDATE Gold_table 
--1
SET FIVE_MINUTEKEY = (
SELECT
    
    CASE 
        WHEN FIVE_MINUTEKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.FIVE_MINUTEKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.FIVE_MINUTEKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        FIVE_MINUTEKEY
    END as key_5),

--2
 FIFTEEN_MINUTEKEY = (
SELECT
    
    CASE 
        WHEN FIFTEEN_MINUTEKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.FIFTEEN_MINUTEKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.FIFTEEN_MINUTEKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        FIFTEEN_MINUTEKEY
    END as key_15),

--3
 THIRTY_MINUTEKEY = (
SELECT
    
    CASE 
        WHEN THIRTY_MINUTEKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.THIRTY_MINUTEKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.THIRTY_MINUTEKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        THIRTY_MINUTEKEY
    END as key_30),

--4
 ONE_HOURKEY = (
SELECT
    
    CASE 
        WHEN ONE_HOURKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.ONE_HOURKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.ONE_HOURKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        ONE_HOURKEY
    END as key_1h),

--5
 TWO_HOURKEY = (
SELECT
    
    CASE 
        WHEN TWO_HOURKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.TWO_HOURKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.TWO_HOURKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        TWO_HOURKEY
    END as key_2h),

--6
 FOUR_HOURKEY = (
SELECT
    
    CASE 
        WHEN FOUR_HOURKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.FOUR_HOURKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.FOUR_HOURKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        FOUR_HOURKEY
    END as key_4h),

--7
 FIVE_HOURKEY = (
SELECT
    
    CASE 
        WHEN FIVE_HOURKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.FIVE_HOURKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.FIVE_HOURKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        FIVE_HOURKEY
    END as key_5h),

--8
 EIGHT_HOURKEY = (
SELECT
    
    CASE 
        WHEN EIGHT_HOURKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.EIGHT_HOURKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.EIGHT_HOURKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        EIGHT_HOURKEY
    END as key_8h),

--9
 TEN_HOURKEY = (
SELECT
    
    CASE 
        WHEN TEN_HOURKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.TEN_HOURKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.TEN_HOURKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        TEN_HOURKEY
    END as key_10h),

--10
 TWELVE_HOURKEY = (
SELECT
    
    CASE 
        WHEN TWELVE_HOURKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.TWELVE_HOURKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.TWELVE_HOURKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        TWELVE_HOURKEY
    END as key_12h),

--11
 ONE_DAYKEY = (
SELECT
    
    CASE 
        WHEN ONE_DAYKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.ONE_DAYKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.ONE_DAYKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        ONE_DAYKEY
    END as key_1d),

--12
 ONE_WEEKKEY = (
SELECT
    
    CASE 
        WHEN ONE_WEEKKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.ONE_WEEKKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.ONE_WEEKKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        ONE_WEEKKEY
    END as key_1w),

--13
 ONE_MONTHKEY = (
SELECT
    
    CASE 
        WHEN ONE_MONTHKEY IS NULL THEN (
            SELECT TOP 1
                inner_table.ONE_MONTHKEY
            FROM
                Gold_table as inner_table
            WHERE
                  inner_table.Ticker_1min = Gold_table.Ticker_1min
                 AND inner_table.Datetime_1min <= Gold_table.Datetime_1min
                AND inner_table.ONE_MONTHKEY IS NOT NULL
            ORDER BY
                inner_table.Datetime_1min DESC
        )
    ELSE
        ONE_MONTHKEY
    END as key_1mm)

FROM
    Gold_table