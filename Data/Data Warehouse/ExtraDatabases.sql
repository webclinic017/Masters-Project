  CREATE DATABASE ProjectPartition_one
    ON (NAME = 'ProjectPartition_one_Data', FILENAME =    'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\ProjectPartition_one_Data.mdf', SIZE = 1000, FILEGROWTH = 50)
    LOG ON (NAME = 'ProjectPartition_one_Data_Log', FILENAME = 'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\ProjectPartition_one_Log.ldf' , SIZE = 20, FILEGROWTH = 96);
GO

USE ProjectPartition_one


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

ALTER TABLE dbo.ONE_MINUTEDim ADD 
    CONSTRAINT PK_ONEMINUTEKey PRIMARY KEY (ONE_MINUTEKEY) ON [PRIMARY];


  CREATE DATABASE ProjectPartition_two
    ON (NAME = 'ProjectPartition_two_Data', FILENAME =    'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\ProjectPartition_two_Data.mdf', SIZE = 1000, FILEGROWTH = 50)
    LOG ON (NAME = 'ProjectPartition_two_Data_Log', FILENAME = 'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\ProjectPartition_two_Log.ldf' , SIZE = 20, FILEGROWTH = 96);
GO

USE ProjectPartition_two


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

ALTER TABLE dbo.ONE_MINUTEDim ADD 
    CONSTRAINT PK_ONEMINUTEKey PRIMARY KEY (ONE_MINUTEKEY) ON [PRIMARY];



  CREATE DATABASE ProjectPK_one
    ON (NAME = 'ProjectPK_one_Data', FILENAME =    'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\ProjectPK_one_Data.mdf', SIZE = 1000, FILEGROWTH = 50)
    LOG ON (NAME = 'ProjectPK_one_Log', FILENAME = 'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\ProjectPK_one_Log.ldf' , SIZE = 20, FILEGROWTH = 96);
GO

USE ProjectPK_one


CREATE TABLE dbo.ONE_MINUTEDim(
 [Datetime] smalldatetime,
 [ONE_MINUTEKEY] varchar(50) not null,
 [Ticker] varchar(5),
) ON [PRIMARY];
GO

ALTER TABLE dbo.ONE_MINUTEDim ADD 
    CONSTRAINT PK_ONEMINUTEKey PRIMARY KEY (ONE_MINUTEKEY) ON [PRIMARY];