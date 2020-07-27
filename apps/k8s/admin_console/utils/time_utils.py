"""
Some tool functions about time
"""

import datetime


def get_datetime_str(date_obj):
    """
    Converts the time object to a string
    """
    if isinstance(date_obj, datetime.datetime):
        cn_time = date_obj.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
        return cn_time.strftime("%Y-%m-%d %H:%M:%S")
    return date_obj


def days_ago(day):
    """
    Get the past timestamp
    """
    return datetime.datetime.now() - datetime.timedelta(days=day)


# week_flag format is 201435（year:2014 week:35）
def get_first_day(week_flag):
    """
    Get the firt day date of week
    """
    year_num = week_flag[0:4]   # get year
    week_num = week_flag[4:len(week_flag)]   # get week
    str_year_start = year_num + '0101'   # the firt day of the year
    year_start = datetime.datetime.strptime(str_year_start, '%Y%m%d')  # format the date
    year_start_calendar_msg = year_start.isocalendar()  # the week info of the firt day
    year_start_week = year_start_calendar_msg[1]
    year_start_weekday = year_start_calendar_msg[2]
    year_start_year = year_start_calendar_msg[0]
    # sunday is the first day of oneweek
    if year_start_year < int(year_num):
        day_delat = (8 - int(year_start_weekday)) + int(week_num) * 7 - 1
    else:
        day_delat = (8 - int(year_start_weekday)) + (int(week_num) - 1) * 7 - 1

    first_date = (year_start + datetime.timedelta(days=day_delat)).date()
    return first_date
