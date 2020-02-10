#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""
Redis utils.

Requirements: redis-py
"""

import os
import time

import redis

import fueling.common.logging as logging


SERVER_HOST = '192.168.48.6'
SERVER_PORT = 6379
TIMEOUT = 5


class RedisConnectionPool(object):
    """
    By default every Redis instance connects the server when initialized,
    and each holds a connection pool, this might cause the slowness and resource waste.
    Trying to mitigate this by making a connection pool that can be reused by all instances.
    """
    connection_pool = None

    @staticmethod
    def get_connection_pool():
        if not RedisConnectionPool.connection_pool:
            redis_passwd = os.environ.get('REDIS_PASSWD')
            if not redis_passwd:
                return None
            RedisConnectionPool.connection_pool = redis.ConnectionPool(
                host=SERVER_HOST, port=SERVER_PORT,
                password=redis_passwd,
                decode_responses=True,
                socket_connect_timeout=TIMEOUT)
        return RedisConnectionPool.connection_pool


def get_redis_instance():
    """
    API to return a Redis instance.
    The instance can be used in scenarios with multiple operations executing as a batch
    """
    connection_pool = RedisConnectionPool.get_connection_pool()
    if not connection_pool:
        logging.error('redis connection pool not available.')
        return None
    return redis.Redis(connection_pool=connection_pool)


def redis_type(redis_key):
    """Get the value type from Redis by using given key"""
    redis_instance = get_redis_instance()
    if not redis_instance:
        logging.error('unable to create redis instance')
        return None
    return _retry(redis_instance.type, [redis_key])


def redis_set(redis_key, redis_value):
    """Instant API to set a key value pair"""
    redis_instance = get_redis_instance()
    if not redis_instance:
        logging.error('unable to create redis instance')
        return
    _retry(redis_instance.set, [redis_key, redis_value])


def redis_get(redis_key):
    """Instant API to get a value by using key"""
    redis_instance = get_redis_instance()
    if not redis_instance:
        logging.error('unable to create redis instance')
        return None
    return _retry(redis_instance.get, [redis_key])


def redis_incr(redis_key, amount=1):
    """Instant API to increment a value by its key"""
    redis_instance = get_redis_instance()
    if not redis_instance:
        logging.error('unable to create redis instance')
        return
    _retry(redis_instance.incr, [redis_key, amount])


def redis_extend(redis_key, redis_values):
    """Instant API to extend a list if the given key exists, create the list otherwise"""
    redis_instance = get_redis_instance()
    if not redis_instance:
        logging.error('unable to create redis instance')
        return
    _retry(redis_instance.rpush, *redis_values)


def redis_range(redis_key, left=0, right=-1):
    """Instant API to get a list with left and right ranges by using key"""
    redis_instance = get_redis_instance()
    if not redis_instance:
        logging.error('unable to create redis instance')
        return None
    return _retry(redis_instance.lrange, [redis_key, left, right])


def redis_extend_dict(redis_key, mapping):
    """Extend a dict in redis if the given key exists, create the dict otherwise"""
    redis_instance = get_redis_instance()
    if not redis_instance:
        logging.error('unable to create redis instance')
        return
    if not isinstance(mapping, dict):
        logging.error('redis_set_dict function requires a dict type of parameter as mapping')
        return
    _retry(redis_instance.hmset, [redis_key, mapping])


def redis_get_dict(redis_key):
    """Get a whole dict out by using the given key"""
    redis_instance = get_redis_instance()
    if not redis_instance:
        logging.error('unable to create redis instance')
        return None
    return _retry(redis_instance.hgetall, [redis_key])


def redis_get_dict_values(redis_key):
    """Get values of dict by using given key"""
    redis_instance = get_redis_instance()
    if not redis_instance:
        logging.error('unable to create redis instance')
        return None
    return _retry(redis_instance.hvals, [redis_key])


def redis_add_smembers(redis_key, members):
    """Add key with set type of values"""
    redis_instance = get_redis_instance()
    if not redis_instance:
        logging.error('unable to create redis instance')
        return
    if not isinstance(members, set):
        logging.error('redis_set_dict function requires a set type of parameter as members')
        return
    _retry(redis_instance.sadd, [redis_key, members])


def redis_get_smembers(redis_key):
    """Get set members out by using the given key"""
    redis_instance = get_redis_instance()
    if not redis_instance:
        logging.error('unable to create redis instance')
        return None
    return _retry(redis_instance.smembers, [redis_key])


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
