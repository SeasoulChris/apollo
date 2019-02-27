"""Time related utils."""
#!/usr/bin/env python

import datetime
import pytz

def msg_time_to_datetime(msg_time, local_tz='America/Los_Angeles', msg_tz='UTC'):
    """Cyber message timestamp is generally in UTC nano seconds."""
    dt = datetime.datetime.fromtimestamp(msg_time / (10 ** 9), pytz.timezone(msg_tz))
    return dt.astimezone(pytz.timezone(local_tz))
