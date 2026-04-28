"""
 @file datumtid.py

 @brief Utility module for datetime conversion and formatting operations.

 @details Provides helpers for checking, converting, formatting, and reducing date
 and datetime values used throughout package workflows.

 *Version History*:
 - Created: 2017-12-24

 @author Thomas Gumbricht

 @date Created: 2017-12-24
"""

# Standard library imports
import datetime

# Package imports
from .code_log import Log

# ==============================
# date type check
# ==============================

def Is_date(dt):
    """
    @brief Checks if an object is a datetime.date instance.
    """
    return isinstance(dt, datetime.date)

def Is_datetime(dt):
    """
    @brief Checks if an object is a datetime.datetime instance.
    """
    return isinstance(dt,datetime.datetime)

# ==============================
# Reduce datetime
# ==============================

def Date_2_str_YYYYMM(date):
    """
    @brief Converts a date object to YYYYMM string format.
    """
    y = '%(y)d' %{'y':date.year}
    m = Month_2_str(date.month)
    return '%s%s' %(y,m)

def Date_from_timestamp(t):
    """
    @brief Converts Unix timestamp (time.time() format) to a date object.
    """
    return datetime.datetime.fromtimestamp(t).date()

def Datetime_from_yyyy_mm_dd_hh_mn_ss_sec(yyyy,mm,dd,hh,mn,ss,sec):
    """
    @brief Creates date and time objects from individual datetime components including fractional seconds.
    """
    tm = datetime.datetime(yyyy,mm,dd,hh,mn,ss)
    dt = tm + datetime.timedelta(0,sec)
    return dt.date(),dt.time()

# ==============================
# Expande datetime
# ==============================

def YYYYMM_str_plus_days_2_date(yyyymmStr,d):
    """
    @brief Converts YYYYMM string plus day number to a date object, handling end-of-month correctly.
    """
    from calendar import monthrange

    y = int(yyyymmStr[0:4])
    m = int(yyyymmStr[4:6])

    if d >= 31:
        d = monthrange(y, m)[1]

    dt = datetime.datetime(y,m,d)
    return dt.date()

# ==============================
# Get specific datetime/dates
# ==============================

def Now():
    """
    @brief Returns the current datetime as a datetime.datetime object.
    """
    return datetime.datetime.now()

def Today():
    """
    @brief Returns today's date as a datetime.date object.
    """
    return datetime.datetime.now().date()

def Get_days_of_YYYY_months(y,m):
    """
    @brief Returns tuple (weekday_of_first_day, number_of_days) for a given year and month.
    """
    from calendar import monthrange
    return monthrange(y, m)

def Set_YYYY1Jan_date(year):
    """
    @brief Creates a date object for January 1st of the specified year.
    """
    return datetime.datetime(year=year, month=1, day=1).date()

def Reset_date_2_YYYYMM01_date(date):
    """
    @brief Resets a date to the first day of its month (YYYY-MM-01).
    """
    return datetime.datetime(year=date.year, month=date.month, day=1).date()

# ==============================
# Conversion to/from string date
# ==============================

def yyyymmdd_str_to_date(yyyymmdd):
    """
    @brief Converts YYYYMMDD string to a date object.
    """
    dt = datetime.datetime(int(yyyymmdd[0:4]),int(yyyymmdd[4:6]),int(yyyymmdd[6:8]))
    return dt.date()

def Date_str_parts_yyyy_mm_dd_2_date(yyyy,mm,dd):
    """
    @brief Converts separate string parts (YYYY, MM, DD) to a date object.
    """
    dt = datetime.datetime(int(yyyy),int(mm),int(dd))
    return dt.date()

def Date_int_parts_yyyy_mm_dd_2_date(yyyy,mm,dd):
    """
    @brief Converts separate integer parts (YYYY, MM, DD) to a date object.
    """
    dt = datetime.datetime(yyyy,mm,dd)
    return dt.date()

def Today_as_str_YYYYMMDD():
    """
    @brief Returns today's date as a YYYYMMDD string.
    """
    return datetime.datetime.now().strftime('%Y%m%d')

def Today_as_hyphen_YYYYMMDD():
    """
    @brief Returns today's date as a YYYY-MM-DD string with hyphens.
    """
    return datetime.datetime.now().strftime('%Y-%m-%d')

def Today_as_point_YYYYMMDD():
    """
    @brief Returns today's date as a YYYY.MM.DD string with periods.
    """
    return datetime.datetime.now().strftime('%Y.%m.%d')

def Now_as_str_YYYYMMDD_HHMMSS():
    """
    @brief Returns current datetime as a YYYYMMDD_HHMMSS string.
    """
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def Now_as_str_4_postgres():
    """
    @brief Returns today's date and time as a string formatted for PostgreSQL.
    """
    #return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    dt = datetime.datetime.now()
    ts_str = f"{dt:%Y-%m-%d %H:%M:%S}.{dt.microsecond // 100000}"
    return ts_str

def yyyymmdd_HH_MM_SS_s_as_str_4_postgres(yyyymmdd_HH_MM_SS_s):
    """
    @brief Converts a YYYYMMDD_HHMMSS_s string to a PostgreSQL-compatible datetime string.
    """
    dt = datetime.datetime.strptime(yyyymmdd_HH_MM_SS_s, '%Y%m%d_%H%M%S_%f')
    ts_str = f"{dt:%Y-%m-%d %H:%M:%S}.{dt.microsecond // 100000}"
    return ts_str

def Date_str_yyyy_mm_dd_2_datetime(yyyymmdd):
    """
    @brief Converts YYYY-MM-DD string (with hyphens) to a date object.
    """
    dt = datetime.datetime(int(yyyymmdd[0:4]),int(yyyymmdd[5:7]),int(yyyymmdd[8:10]))
    return dt.date()

def YYYYDOY_2_DOY_str(date):
    """
    @brief Extracts Day Of Year (DOY) from a date and returns it as a zero-padded 3-digit string.
    """
    DOY = date.timetuple().tm_yday

    if DOY < 10:
        doyStr = '00%(d)d' %{'d':DOY}
    elif DOY < 100:
        doyStr = '0%(d)d' %{'d':DOY}
    else:
        doyStr = '%(d)d' %{'d':DOY}

    return doyStr

def Month_2_str(m):
    """
    @brief Converts month number (1-12) to zero-padded 2-digit string.
    """
    if m < 10:
        return '0%(m)d' %{'m':m}
    else:
        return '%(m)d' %{'m':m}
    
def Date_2_str_date(date):
    """
    @brief Converts date object to YYYYMMDD string format.
    """
    return date.strftime("%Y%m%d")

def Date_2_str_point_dat(datum):
    """
    @brief Converts date object to YYYY.MM.DD string format with periods.
    """
    return datum.strftime("%Y.%m.%d")

def Date_2_str_hyphen_dat(datum):
    """
    @brief Converts date object to YYYY-MM-DD string format with hyphens.
    """
    return datum.strftime("%Y-%m-%d")

def DOY_int_2_str(DOY):
    """
    @brief Converts integer Day Of Year (1-366) to zero-padded 3-digit string.
    """
    if DOY < 10:
        DOY_str = '00%(d)d' %{'d':DOY}
    elif DOY < 100:
        DOY_str = '0%(d)d' %{'d':DOY}
    else:
        DOY_str = '%(d)d' %{'d':DOY}

    return DOY_str

def IntYYYYMMDDDate(yyyy,mm,dd):
    """
    @brief Creates date object from integer year, month, and day components.
    """
    dt = datetime.datetime(yyyy,mm,dd)
    return dt.date()

# ==============================
# Conversion to/from numpy datetime64
# ==============================

def Datetime_to_numpydate(dt):
    """
    @brief Converts Python datetime object to NumPy datetime64 object.
    """
    from numpy import datetime64
    return datetime64(dt)

# ==============================
# Add day/month/year to date
# ==============================

def Add_days_to_date(dt,days_to_add):
    """
    @brief Adds specified number of days to a date object.
    """
    dt += datetime.timedelta(days=days_to_add)
    return dt

def Add_months_to_date(dt,months_to_add):
    """
    @brief Adds specified number of months to a date object using dateutil.
    """
    from dateutil.relativedelta import relativedelta
    delta_months = relativedelta(months=months_to_add)
    return dt+delta_months

def Add_years_to_date(dt,years_to_add):
    """
    @brief Adds specified number of years to a date object using dateutil.
    """
    from dateutil.relativedelta import relativedelta
    delta_years = relativedelta(years=years_to_add)
    return dt+delta_years

# ==============================
# Day Of Year (DOY) functions
# ==============================

def yyyydoy_2_date(yyyydoy):
    """
    @brief Converts YYYYDOY string (year + day-of-year) to a date object.
    """
    dt = datetime.datetime(int(yyyydoy[0:4]),1,1)
    dtdelta = datetime.timedelta(days=int(yyyydoy[4:7])-1)
    datum = dt + dtdelta
    return datum.date()

def Date_2_DOY(date):
    """
    @brief Extracts Day Of Year (DOY) as integer from a date object.
    """
    return date.timetuple().tm_yday

def DateToYYYYDOY(date):
    """
    @brief Converts date object to YYYYDOY string format.
    """
    doy = YYYYDOY_2_DOY_str(date)
    yyyydoyStr = '%(y)d%(doy)s' %{'y':date.year,'doy':doy}
    return yyyydoyStr

def YYYY_plus_DOY_2_date(year,doy):
    """
    @brief Creates date object from separate year integer and Day Of Year integer.
    """
    datum = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
    return datum.date()

# ==============================
# Range functions
# ==============================

def Get_month_range(startdate,enddate):
    """
    @brief Returns list of datetime objects representing monthly intervals between start and end dates.
    """
    from dateutil import rrule
    return list(rrule.rrule(rrule.MONTHLY, dtstart=startdate, until=enddate))

def Get_date_range(date1, date2):
    """
    @brief Returns list of all date objects between date1 and date2 (inclusive).
    """
    dateL = []
    for n in range(int ((date2 - date1).days)+1):
        dateL.append(date1 + datetime.timedelta(n))
    return dateL

# ==============================
# Delta functions
# ==============================

def Delta_days(start_date,end_date):
    """
    @brief Calculates number of days between two dates, supporting datetime, YYYYMMDD, and YYYY-MM-DD formats.
    """
    if isinstance(start_date,datetime.datetime):
        start_date_date = start_date.date()
        end_date_date = end_date.date()

    elif isinstance(start_date,str):
        if len(start_date) == 8:
            start_date_date = yyyymmdd_str_to_date(start_date)
            end_date_date = yyyymmdd_str_to_date(end_date)

        elif len(start_date) == 10:
            start_date_date = Date_str_yyyy_mm_dd_2_datetime(start_date)
            end_date_date = Date_str_yyyy_mm_dd_2_datetime(end_date)

        else:
            log_msg = 'Date string format not recognized: %s, %s' %(start_date, end_date) 
            Log(log_msg)
            return None

    return (end_date_date-start_date_date).days