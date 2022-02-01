--USE ProjectWarehouse
--INSERT INTO [ProjectWarehouse].[dbo].[DateFact](Datetime,MonthNumber,MonthDay,WeekNumber,DayWeek,DayNumber,HourOfDay,MinuteOfHour,Year)
--SELECT Date,MonthNumber,MonthDay,WeekNumber,DayWeek,DayNumber,HourOfDay,MinuteOfHour,Year
--FROM [ProjectWarehouse].[dbo].[DateTest]


USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[ONE_MINUTEDim]([ONE_MINUTEKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [ONE_MINUTEKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[ONE_MINUTEDim]

 
USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[FIVE_MINUTEDim]([FIVE_MINUTEKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [FIVE_MINUTEKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[FIVE_MINUTEDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[FIFTEEN_MINUTEDim]([FIFTEEN_MINUTEKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [FIFTEEN_MINUTEKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[FIFTEEN_MINUTEDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[THIRTY_MINUTEDim]([THIRTY_MINUTEKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [THIRTY_MINUTEKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[THIRTY_MINUTEDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[ONE_HOURDim]([ONE_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [ONE_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[ONE_HOURDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[TWO_HOURDim]([TWO_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [TWO_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[TWO_HOURDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[FOUR_HOURDim]([FOUR_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [FOUR_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[FOUR_HOURDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[FIVE_HOURDim]([FIVE_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [FIVE_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[FIVE_HOURDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[EIGHT_HOURDim]([EIGHT_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [EIGHT_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[EIGHT_HOURDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[TEN_HOURDim]([TEN_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [TEN_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[TEN_HOURDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[TWELVE_HOURDim]([TWELVE_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [TWELVE_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[TWELVE_HOURDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[ONE_HOURDim]([ONE_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [ONE_HOURKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[ONE_HOURDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[ONE_MONTHDim]([ONE_MONTHKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [ONE_MONTHKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[ONE_MONTHDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[ONE_WEEKDim]([ONE_WEEKKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [ONE_WEEKKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[ONE_WEEKDim]

USE ProjectStagging
INSERT INTO [ProjectWarehouse].[dbo].[ONE_DAYDim]([ONE_DAYKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume])
SELECT [ONE_DAYKEY],[Datetime],[Ticker],[Open],[Low],[High],[Close],[Volume]
FROM [ProjectStagging].[dbo].[ONE_DAYDim]
