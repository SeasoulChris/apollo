#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""
Redis utils.

Requirements: redis-py
"""

from absl import flags
import redis

import fueling.common.logging as logging

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
    def connection_pool(flags_dict=None):
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
    The instance can be used in scenarios when multiple operations need to be done
    """
    return Redis(connection_pool=RedisConnectionPool.connection_pool())

def redis_set(redis_key, redis_value):
    """Instant API to set a key value pair"""
    get_redis_instance().set(redis_key, redis_value)

def redis_get(redis_key):
    """Instant API to get a value by using key"""
    return get_redis_instance().get(redis_key)

# TODO(longtao): Add more utils and error handling logic later
