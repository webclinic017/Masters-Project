  CREATE DATABASE ProjectWarehouse
    ON (NAME = 'ProjectWarehouse_Data', FILENAME =    'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\ProjectWarehouse2016_Data.mdf', SIZE = 1000, FILEGROWTH = 50)
    LOG ON (NAME = 'ProjectWarehouse_Data_Log', FILENAME = 'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\ProjectWarehouse_Log.ldf' , SIZE = 20, FILEGROWTH = 96);
GO

USE ProjectWarehouse

CREATE TABLE dbo.DateFact(
[Datetime] smalldatetime,
[MonthNumber] int,
[MonthDay] int,
[WeekNumber] int,
[DayWeek] int, 
[DayNumber] int,
[HourOfDay] int,
[MinuteOfHour] int,
[Year] int,
[ONE_MINUTEKEY] varchar(30) not null,
[FIVE_MINUTEKEY] varchar(30) not null,
[FIFTEEN_MINUTEKEY] varchar(30) not null,
[THIRTY_MINUTEKEY] varchar(30) not null,  
[ONE_HOURKEY] varchar(30) not null,
[TWO_HOURKEY] varchar(30) not null,
[FOUR_HOURKEY] varchar(30) not null ,
[FIVE_HOURKEY] varchar(30) not null ,
[TEN_HOURKEY] varchar(30) not null ,
[EIGHT_HOURKEY] varchar(30) not null,
[TWELVE_HOURKEY] varchar(30) not null,
[ONE_DAYKEY] varchar(30) not null,
[ONE_WEEKKEY] varchar(30) not null,
[ONE_MONTHKEY] varchar(30) not null,
[TickerKey] varchar(5) not null,
) ON [PRIMARY];
GO


CREATE TABLE dbo.ONE_MINUTEDim(
 [ONE_MINUTEKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.FIVE_MINUTEDim(
 [FIVE_MINUTEKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.FIFTEEN_MINUTEDim(
 [FIFTEEN_MINUTEKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.THIRTY_MINUTEDim(
 [THIRTY_MINUTEKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.ONE_HOURDim(
 [ONE_HOURKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO


CREATE TABLE dbo.TWO_HOURDim(
 [TWO_HOURKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.FOUR_HOURDim(
 [FOUR_HOURKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.FIVE_HOURDim(
 [FIVE_HOURKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO


CREATE TABLE dbo.EIGHT_HOURDim(
 [EIGHT_HOURKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.TEN_HOURDim(
 [TEN_HOURKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.TWELVE_HOURDim(
 [TWELVE_HOURKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.ONE_DAYDim(
 [ONE_DAYKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.ONE_WEEKDim(
 [ONE_WEEKKEY] varchar(30) not null,
 [Datetime] smalldatetime,
 [Ticker] varchar(5),
 [Open] float,
 [Low] float,
 [High] float,
 [Close] float,
 [Volume] int,
) ON [PRIMARY];
GO

CREATE TABLE dbo.ONE_MONTHDim(
 [ONE_MONTHKEY] varchar(30) not null,
 [Datetime] smalldatetime,
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
 [Instrument] varchar(40) not null,
 [Type] varchar(25) not null,
 [Exchange] varchar(10) not null,
) ON [PRIMARY];
GO


ALTER TABLE dbo.ONE_MINUTEDim ADD 
    CONSTRAINT PK_ONEMINUTEKey PRIMARY KEY (ONE_MINUTEKEY) ON [PRIMARY];

ALTER TABLE dbo.FIVE_MINUTEDim ADD 
    CONSTRAINT PK_FIVEMINUTEKey PRIMARY KEY (FIVE_MINUTEKEY) ON [PRIMARY];

ALTER TABLE dbo.FIFTEEN_MINUTEDim ADD 
    CONSTRAINT PK_FIFTEENMINUTEKey PRIMARY KEY (FIFTEEN_MINUTEKEY) ON [PRIMARY];

ALTER TABLE dbo.THIRTY_MINUTEDim ADD 
    CONSTRAINT PK_THIRTYMINUTEKey PRIMARY KEY (THIRTY_MINUTEKEY) ON [PRIMARY];

ALTER TABLE dbo.ONE_HOURDim ADD 
    CONSTRAINT PK_ONEHOURKey PRIMARY KEY (ONE_HOURKEY) ON [PRIMARY];

ALTER TABLE dbo.TWO_HOURDim ADD 
    CONSTRAINT PK_TWOHOURKey PRIMARY KEY (TWO_HOURKEY) ON [PRIMARY];

ALTER TABLE dbo.FOUR_HOURDim ADD 
    CONSTRAINT PK_FOURHOURKey PRIMARY KEY (FOUR_HOURKEY) ON [PRIMARY];

ALTER TABLE dbo.FIVE_HOURDim ADD 
    CONSTRAINT PK_FIVEHOURKey PRIMARY KEY (FIVE_HOURKEY) ON [PRIMARY];

ALTER TABLE dbo.EIGHT_HOURDim ADD 
    CONSTRAINT PK_EIGHTHOURKey PRIMARY KEY (EIGHT_HOURKEY) ON [PRIMARY];

ALTER TABLE dbo.TEN_HOURDim ADD 
    CONSTRAINT PK_TENHOURKey PRIMARY KEY (TEN_HOURKEY) ON [PRIMARY];

ALTER TABLE dbo.TWELVE_HOURDim ADD 
    CONSTRAINT PK_TWELVEHOURKey PRIMARY KEY (TWELVE_HOURKEY) ON [PRIMARY];

ALTER TABLE dbo.ONE_DAYDim ADD 
    CONSTRAINT PK_ONEDAYKey PRIMARY KEY (ONE_DAYKEY) ON [PRIMARY];

ALTER TABLE dbo.ONE_WEEKDim ADD 
    CONSTRAINT PK_ONEWEEKKey PRIMARY KEY (ONE_WEEKKEY) ON [PRIMARY];

ALTER TABLE dbo.ONE_MONTHDim ADD 
    CONSTRAINT PK_ONEMONTHKey PRIMARY KEY (ONE_MONTHKEY) ON [PRIMARY];

ALTER TABLE dbo.InstrumentDim ADD 
    CONSTRAINT PK_TickerKey PRIMARY KEY (TickerKey) ON [PRIMARY];

ALTER TABLE dbo.DateFact ADD  CONSTRAINT 
	PK_DateFact PRIMARY KEY (ONE_MINUTEKEY,FIVE_MINUTEKEY,FIFTEEN_MINUTEKEY,THIRTY_MINUTEKEY,ONE_HOURKEY,TWO_HOURKEY,FOUR_HOURKEY,FIVE_HOURKEY,EIGHT_HOURKEY,TEN_HOURKEY,TWELVE_HOURKEY,ONE_DAYKEY,ONE_WEEKKEY,ONE_MONTHKEY,TickerKey) ON [PRIMARY];


Alter Table dbo.DateFact ADD 
CONSTRAINT FK_ONEMINUTEKEY FOREIGN KEY (ONE_MINUTEKEY)
REFERENCES dbo.ONE_MINUTEDim (ONE_MINUTEKEY),

CONSTRAINT FK_FIVEMINUTEKEY FOREIGN KEY (FIVE_MINUTEKEY)
REFERENCES dbo.FIVE_MINUTEDim (FIVE_MINUTEKEY),

CONSTRAINT FK_FIFTEENMINUTEKEY FOREIGN KEY (FIFTEEN_MINUTEKEY)
REFERENCES dbo.FIFTEEN_MINUTEDim (FIFTEEN_MINUTEKEY),

CONSTRAINT FK_THIRTYMINUTEKEY FOREIGN KEY (THIRTY_MINUTEKEY)
REFERENCES dbo.THIRTY_MINUTEDim (THIRTY_MINUTEKEY),

CONSTRAINT FK_ONEHOURKEY FOREIGN KEY (ONE_HOURKEY)
REFERENCES dbo.ONE_HOURDim (ONE_HOURKEY),

CONSTRAINT FK_TWOHOURKEY FOREIGN KEY (TWO_HOURKEY)
REFERENCES dbo.TWO_HOURDim (TWO_HOURKEY),

CONSTRAINT FK_FOURHOURKEY FOREIGN KEY (FOUR_HOURKEY)
REFERENCES dbo.FOUR_HOURDim (FOUR_HOURKEY),

CONSTRAINT FK_FIVEHOURKEY FOREIGN KEY (FIVE_HOURKEY)
REFERENCES dbo.FIVE_HOURDim (FIVE_HOURKEY),

CONSTRAINT FK_EIGHTHOURKEY FOREIGN KEY (EIGHT_HOURKEY)
REFERENCES dbo.EIGHT_HOURDim (EIGHT_HOURKEY),

CONSTRAINT FK_TENHOURKEY FOREIGN KEY (TEN_HOURKEY)
REFERENCES dbo.TEN_HOURDim (TEN_HOURKEY),

CONSTRAINT FK_TWELVEHOURKEY FOREIGN KEY (TWELVE_HOURKEY)
REFERENCES dbo.TWELVE_HOURDim (TWELVE_HOURKEY),

CONSTRAINT FK_ONEDAYKEY FOREIGN KEY (ONE_DAYKEY)
REFERENCES dbo.ONE_DAYDim (ONE_DAYKEY),

CONSTRAINT FK_ONEWEEKKEY FOREIGN KEY (ONE_WEEKKEY)
REFERENCES dbo.ONE_WEEKDim (ONE_WEEKKEY),

CONSTRAINT FK_ONEMONTHKEY FOREIGN KEY (ONE_MONTHKEY)
REFERENCES dbo.ONE_MONTHDim (ONE_MONTHKEY),

CONSTRAINT FK_TickerKey FOREIGN KEY (TickerKey)
REFERENCES dbo.InstrumentDim (TickerKey);

GO

--Alter Table dbo.DateFact 
--DROP constraint [FK_ONEMINUTEKEY]
--GO 

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