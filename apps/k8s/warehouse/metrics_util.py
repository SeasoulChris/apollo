#!/usr/bin/env python
"""Serve data from redis service"""

import fueling.common.redis_utils as redis_utils

def get_metrics_by_prefix(prefix):
    max_count = 50
    if prefix is None:
        prefix = ''
    redis_instance = redis_utils.get_redis_instance()
    redis_keys = sorted(redis_instance.keys('{}*'.format(prefix)))
    metrics = {}
    for key in redis_keys:
        metrics[key] = redis_instance.get(key)
        max_count -= 1
        if max_count <= 0:
            break
    return metrics 
