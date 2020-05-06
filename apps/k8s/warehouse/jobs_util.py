#!/usr/bin/env python
"""Serve data from jobs service"""

from datetime import datetime

import apps.k8s.warehouse.display_util as display_util
import fueling.common.kubectl_utils as kubectl_utils

def get_pods():
    res = kubectl_utils.get_pods()
    curr_datetime = datetime.utcnow()
    for r in res:
        creation_timestamp = r['creation_timestamp']
        running_time_in_seconds = curr_datetime.timestamp() - creation_timestamp
        duration = display_util.ns_to_duration(running_time_in_seconds * 1e9)
        r['duration'] = duration
        r['createtion_time'] = display_util.timestamp_to_time(creation_timestamp)
    return res

