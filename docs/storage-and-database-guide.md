# How to use storage and database

## Baidu BOS for internal and external data

## Azure Blob for external data

## MongoDB for internal data indexing

## Redis for distributed metrics collection

Metrics are used for recording and monitoring various of intereted points of data pipeline jobs, for example how many tasks have been processed, how much resource has been used and etc.  We use Redis to make sure the metrics recording is consistent and persistent in the distributed system.

### Metrics Collection

The collection of metrics means checking, inserting or updating keys to Redis database.  We are now supporting the following APIs:
* redis_set(redis_key, redis_value): insert a key to Redis if it does not exist, otherwise update its value
* redis_get(redis_key): get value by using key
* redis_incr(redis_key, amount=1): increase the key's value by certain mount
* get_redis_instance(): return a raw Redis client instance that can directly execute standard [Redis operations] (https://redis-py.readthedocs.io/en/latest/)

All these APIs have encapsulated retrying and error handling mechanism, except for "get_redis_instance()".  By default the failure of APIs execution will *not* throw execptions but just log the error messages.  But for "get_redis_instance()" you have to handle errors or timeouts by your own if necessary. 

Usage example: 

```python
import fueling.common.redis_utils as redis_utils
# Key can be any string by your design, good practice is to use "." as the separator
redis_utils.redis_set('abc.123.xxx', 100)
redis_utils.redis_incr('abc.123.xxx', 10)
# should return 110
print(redis_utils.get('abc.123.xxx'))
```

### Metrics Dashboard

You can view the metrics by going to [Dashboard] (http://usa-data.baidu.com), and clicking the "Metrics" navigate button.  Currently it shows all the keys in the system by default.

You can narrow down the scope by using the "Prefix..." form, which will then retrieve only the keys with specified prefix.
