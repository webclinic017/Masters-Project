-- declare variables to hold the start and end date

use ProjectWarehouse
DECLARE @StartDate smalldatetime
DECLARE @EndDate smalldatetime

--- assign values to the start date and end date we 
-- want our reports to cover (this should also take
-- into account any future reporting needs)
SET @StartDate = '2006-01-01 00:00:00'
SET @EndDate = '2021-11-02 00:00:00' 

IF EXISTS (SELECT * 
			FROM sysobjects 
			WHERE type = 'U' 
			AND ID = OBJECT_ID('[dbo].[DateTest]') )
BEGIN
	DROP TABLE [dbo].[DateTest]
	PRINT 'Table dropped'
END

CREATE TABLE dbo.DateTest(
[Date] smalldatetime NOT NULL,
[MonthNumber] int not null,
[MonthDay] int not null,
[WeekNumber] int not null,
[DayWeek] int not null,
[DayNumber] int not null,
[HourOfDay] int not null,
[MinuteOfHour] int not null,
[Year] int not null,
) 

-- using a while loop increment from the start date 
-- to the end date
DECLARE @LoopDate smalldatetime
SET @LoopDate = @StartDate

WHILE @LoopDate <= @EndDate
BEGIN
 -- add a record into the date dimension table for this date
INSERT INTO [dbo].[DateTest] VALUES (
  @LoopDate,
  month(@LoopDate),
  day(@loopdate),
  DATEPART(WK,@LoopDate), 
  datepart(WEEKDAY,@LoopDate),
  datepart(dy, @loopdate),
  datepart(hh, @loopdate),
  datepart(mi, @loopdate),
  year(@loopdate)
 )  
 
 
 -- increment the LoopDate by 1 day before
 -- we start the loop again
 SET @LoopDate = DateAdd(MINUTE, 1, @LoopDate)
END
