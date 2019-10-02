#!/usr/bin/env python
"""Serve data from redis service"""

import fueling.common.redis_utils as redis_utils

def GetMetricsByPrefix(prefix):
    if prefix is None:
        prefix = ''
    redis_instance = redis_utils.get_redis_instance()
    redis_keys = redis_instance.keys('{}*'.format(prefix))
    metrics = {}
    for key in redis_keys:
        metrics[key] = redis_instance.get(key)
    return metrics 
