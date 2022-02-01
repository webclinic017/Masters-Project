USE master

CREATE DATABASE ProjectStagging
    ON (NAME = 'ProjectStagging_Data', FILENAME =    'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\ProjectStagging_Data.mdf', SIZE = 10240, FILEGROWTH = 96)
    LOG ON (NAME = 'ProjectStagging_Data_Log', FILENAME = 'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\ProjectStagging_Log.ldf' , SIZE = 20, FILEGROWTH = 96);
GO

USE ProjectStagging

CREATE TABLE dbo.DateFact(
[MonthNumber] int,
[Datetime] smalldatetime,
[MonthDay] int,
[WeekNumber] int,
[DayWeek] int,
[DayNumber] int, 
[HourOfDay] int,
[MinuteOfHour] int,
[Year] int,
) ON [PRIMARY];
GO


CREATE TABLE dbo.ONE_MINUTEDim(
 [Datetime] smalldatetime,
 [ONE_MINUTEKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.FIVE_MINUTEDim(
 [Datetime] smalldatetime,
 [FIVE_MINUTEKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.FIFTEEN_MINUTEDim(
 [Datetime] smalldatetime,
 [FIFTEEN_MINUTEKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.THIRTY_MINUTEDim(
 [Datetime] smalldatetime,
 [THIRTY_MINUTEKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.ONE_HOURDim(
 [Datetime] smalldatetime,
 [ONE_HOURKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO


CREATE TABLE dbo.TWO_HOURDim(
 [Datetime] smalldatetime,
 [TWO_HOURKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.FOUR_HOURDim(
 [Datetime] smalldatetime,
 [FOUR_HOURKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.FIVE_HOURDim(
 [Datetime] smalldatetime,
 [FIVE_HOURKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.EIGHT_HOURDim(
 [Datetime] smalldatetime,
 [EIGHT_HOURKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.TEN_HOURDim(
 [Datetime] smalldatetime,
 [TEN_HOURKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.TWELVE_HOURDim(
 [Datetime] smalldatetime,
 [TWELVE_HOURKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.ONE_DAYDim(
 [Datetime] smalldatetime,
 [ONE_DAYKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.ONE_WEEKDim(
 [Datetime] smalldatetime,
 [ONE_WEEKKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.ONE_MONTHDim(
 [Datetime] smalldatetime,
 [ONE_MONTHKEY] varchar(50) not null,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.InstrumentDim(
 [TickerKey] varchar(5) not null,
 [Instrument] varchar(70) not null,
 [Type] varchar(25) not null,
 [Exchange] varchar(10) not null,
) ON [PRIMARY];
GO


--ALTER TABLE dbo.ONE_MINUTEDim ADD 
--    CONSTRAINT PK_ONEMINUTEKey PRIMARY KEY (ONE_MINUTEKEY) ON [PRIMARY];

--ALTER TABLE dbo.FIVE_MINUTEDim ADD 
--    CONSTRAINT PK_FIVEMINUTEKey PRIMARY KEY (FIVE_MINUTEKEY) ON [PRIMARY];

--ALTER TABLE dbo.FIFTEEN_MINUTEDim ADD 
--    CONSTRAINT PK_FIFTEENMINUTEKey PRIMARY KEY (FIFTEEN_MINUTEKEY) ON [PRIMARY];

--ALTER TABLE dbo.THIRTY_MINUTEDim ADD 
--   CONSTRAINT PK_THIRTYMINUTEKey PRIMARY KEY (THIRTY_MINUTEKEY) ON [PRIMARY];

--ALTER TABLE dbo.ONE_HOURDim ADD 
--    CONSTRAINT PK_ONEHOURKey PRIMARY KEY (ONE_HOURKEY) ON [PRIMARY];

--ALTER TABLE dbo.TWO_HOURDim ADD 
--    CONSTRAINT PK_TWOHOURKey PRIMARY KEY (TWO_HOURKEY) ON [PRIMARY];

--ALTER TABLE dbo.FOUR_HOURDim ADD 
--    CONSTRAINT PK_FOURHOURKey PRIMARY KEY (FOUR_HOURKEY) ON [PRIMARY];

--ALTER TABLE dbo.FIVE_HOURDim ADD 
--    CONSTRAINT PK_FIVEHOURKey PRIMARY KEY (FIVE_HOURKEY) ON [PRIMARY];

--ALTER TABLE dbo.EIGHT_HOURDim ADD 
--    CONSTRAINT PK_EIGHTHOURKey PRIMARY KEY (EIGHT_HOURKEY) ON [PRIMARY];

--ALTER TABLE dbo.TEN_HOURDim ADD 
--    CONSTRAINT PK_TENHOURKey PRIMARY KEY (TEN_HOURKEY) ON [PRIMARY];

--ALTER TABLE dbo.TWELVE_HOURDim ADD 
--    CONSTRAINT PK_TWELVEHOURKey PRIMARY KEY (TWELVE_HOURKEY) ON [PRIMARY];

--ALTER TABLE dbo.ONE_DAYDim ADD 
--    CONSTRAINT PK_ONEDAYKey PRIMARY KEY (ONE_DAYKEY) ON [PRIMARY];

--ALTER TABLE dbo.ONE_WEEKDim ADD 
--    CONSTRAINT PK_ONEWEEKKey PRIMARY KEY (ONE_WEEKKEY) ON [PRIMARY];

--ALTER TABLE dbo.ONE_MONTHDim ADD 
--    CONSTRAINT PK_ONEMONTHKey PRIMARY KEY (ONE_MONTHKEY) ON [PRIMARY];

--ALTER TABLE dbo.InstrumentDim ADD 
--    CONSTRAINT PK_TickerKey PRIMARY KEY (TickerKey) ON [PRIMARY];


--drop table dbo.DateFact
--drop table dbo.ONE_MINUTEDim
--drop table dbo.FIVE_MINUTEDim
--drop table dbo.FIFTEEN_MINUTEDim
--drop table dbo.THIRTY_MINUTEDim
--drop table dbo.ONE_HOURDim
--drop table dbo.TWO_HOURDim
--drop table dbo.FOUR_HOURDim
--drop table dbo.FIVE_HOURDim
--drop table dbo.EIGHT_HOURDim
--drop table dbo.TEN_HOURDim
--drop table dbo.TWELVE_HOURDim
--drop table dbo.ONE_DAYDim
--drop table dbo.ONE_WEEKDim
--drop table dbo.ONE_MONTHDim
--drop table dbo.InstrumentDim
