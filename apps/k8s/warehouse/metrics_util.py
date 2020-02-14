#!/usr/bin/env python
"""Serve data from redis service"""

import fueling.common.redis_utils as redis_utils


def is_numeric(value):
    """Check if a string can be converted to numbers"""
    if value.isnumeric():
        return True
    try:
        float(value)
    except ValueError:
        return False
    return True


def get_display_value(values):
    """Generate a string to display values in UI"""
    nums_prefix, nums_suffix = '[', ',...]'
    norm_prefix, norm_suffix = '{', '}'
    values = [str(v) for v in values]
    if all(is_numeric(v) for v in values):
        return nums_prefix + ','.join(values) + nums_suffix
    return norm_prefix + ','.join(values) + norm_suffix


def get_metrics_by_prefix(prefix):
    """Get Redis values by given key prefix, for display purpose"""
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
            metrics[key] = get_display_value(redis_instance.lrange(key, range_left, range_right)) 
        elif redis_instance.type(key) == 'hash':
            metrics[key] = get_display_value(redis_instance.hvals(key)[-range_right:])
        elif redis_instance.type(key) == 'set':
            metrics[key] = get_display_value(redis_instance.srandmember(key, range_right - range_left))
        else:
            metrics[key] = redis_instance.get(key)
        max_count -= 1
        if max_count <= 0:
            break
    return metrics
