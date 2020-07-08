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


# weekflag format is 201435（year:2014 week:35）
def getfirstday(weekflag):
    """
    Get the firt day date of week
    """
    yearnum = weekflag[0:4]   # get year
    weeknum = weekflag[4:len(weekflag)]   # get week
    stryearstart = yearnum + '0101'   # the firt day of the year
    yearstart = datetime.datetime.strptime(stryearstart, '%Y%m%d')  # format the date
    yearstartcalendarmsg = yearstart.isocalendar()  # the week info of the firt day
    yearstartweek = yearstartcalendarmsg[1]
    yearstartweekday = yearstartcalendarmsg[2]
    yearstartyear = yearstartcalendarmsg[0]
    # sunday is the first day of oneweek
    if yearstartyear < int(yearnum):
        daydelat = (8 - int(yearstartweekday)) + int(weeknum) * 7 - 1
    else:
        daydelat = (8 - int(yearstartweekday)) + (int(weeknum) - 1) * 7 - 1

    firtdate = (yearstart + datetime.timedelta(days=daydelat)).date()
    return firtdate
