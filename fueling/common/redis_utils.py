#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""
Redis utils.

Requirements: redis-py
"""

import os
import time

from absl import flags
import redis

import fueling.common.logging as logging

REDIS_KEY_PREFIX_DECODE_VIDEO = 'decode_video'

flags.DEFINE_string('redis_server_ip', '192.168.48.6', 'Internal Redis server IP address.')
flags.DEFINE_integer('redis_port', 6379, 'Redis service port.')
flags.DEFINE_integer('redis_timeout', 5, 'Timeout exception will throw after configured seconds.')


class RedisConnectionPool(object):
    """
    By default every Redis instance connects the server when initialized,
    and each holds a connection pool, this might cause the slowness and resource waste.
    Trying to mitigate this by making a connection pool that can be reused by all instances.
    """
    connection_pool = None

    @staticmethod
    def get_connection_pool(flags_dict=None):
        if not RedisConnectionPool.connection_pool:
            if not flags_dict:
                flags_dict = flags.FLAGS.flag_values_dict()
            RedisConnectionPool.connection_pool = redis.ConnectionPool(
                host=flags_dict['redis_server_ip'],
                port=flags_dict['redis_port'],
                password=os.environ.get('REDIS_PASSWD'),
                decode_responses=True,
                socket_connect_timeout=flags_dict['redis_timeout'])
        return RedisConnectionPool.connection_pool


def get_redis_instance():
    """
    API to return a Redis instance.
    The instance can be used in scenarios with multiple operations executing as a batch
    """
    return redis.Redis(connection_pool=RedisConnectionPool.get_connection_pool())


def redis_type(redis_key):
    """Get the value type from Redis by using given key"""
    return _retry(get_redis_instance().type, [redis_key])


def redis_set(redis_key, redis_value):
    """Instant API to set a key value pair"""
    _retry(get_redis_instance().set, [redis_key, redis_value])


def redis_get(redis_key):
    """Instant API to get a value by using key"""
    return _retry(get_redis_instance().get, [redis_key])


def redis_incr(redis_key, amount=1):
    """Instant API to increment a value by its key"""
    _retry(get_redis_instance().incr, [redis_key, amount])


def redis_extend(redis_key, redis_values):
    """Instant API to extend a list if the given key exists, create the list otherwise"""
    _retry(get_redis_instance().rpush, *redis_values)


def redis_range(redis_key, left=0, right=-1):
    """Instant API to get a list with left and right ranges by using key"""
    return _retry(get_redis_instance().lrange, [redis_key, left, right])


def redis_extend_dict(redis_key, mapping):
    """Extend a dict in redis if the given key exists, create the dict otherwise"""
    if not isinstance(mapping, dict):
        logging.error('redis_set_dict function requires a dict type of parameter as mapping')
        return
    _retry(get_redis_instance().hmset, [redis_key, mapping])


def redis_get_dict(redis_key):
    """Get a whole dict out by using the given key"""
    return _retry(get_redis_instance().hgetall, [redis_key])


def redis_get_dict_values(redis_key):
    """Get values of dict by using given key"""
    return _retry(get_redis_instance().hvals, [redis_key])


def _retry(func, params):
    """A wrapper for exponential retry in case redis connection is not stable"""
    cur_retries, max_retries = 0, 3
    while cur_retries < max_retries:
        try:
            return func(*params)
        except redis.exceptions.TimeoutError as ex:
            logging.error('redis connection timeout. params: {}'.format(params))
            if cur_retries >= max_retries:
                # Silently swallow it instead of raising
                return None
            time.sleep(2 ** cur_retries)
            cur_retries += 1
        except Exception as ex:
            # Silently swallow it instead of raising
            logging.error('redis error: {}. params: {}'.format(ex, params))
            return None
