#!/usr/bin/env python
"""Serve data from redis service"""

import fueling.common.redis_utils as redis_utils


def get_metrics_by_prefix(prefix):
    max_count, range_left, range_right = 50, 0, 10
    # ignore some testing keys
    qa_test_prefix = 'BDRP'
    qa_status_prefix = 'q:'
    if prefix is None:
        prefix = ''
    redis_instance = redis_utils.get_redis_instance()

    redis_keys = []
    for key in redis_instance.scan_iter('{}*'.format(prefix), max_count):
        redis_keys.append(key)
    redis_keys.sort()

    metrics = {}
    for key in redis_keys:
        if key.startswith(qa_test_prefix) or key.startswith(qa_status_prefix):
            continue
        if redis_instance.type(key) == 'list':
            values = [str(x) for x in redis_instance.lrange(key, range_left, range_right)]
            metrics[key] = '[' + ','.join(values) + ',...]'
        elif redis_instance.type(key) == 'hash':
            values = [str(x) for x in redis_instance.hvals(key)[-range_right:]]
            metrics[key] = '[...,' + ','.join(values) + ']'
        elif redis_instance.type(key) == 'set':
            values = [str(x) for x in redis_instance.srandmember(key, range_right - range_left)]
            metrics[key] = '[' + ','.join(values) + ',...]'
        else:
            metrics[key] = redis_instance.get(key)
        max_count -= 1
        if max_count <= 0:
            break
    return metrics
