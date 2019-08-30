"""
DB-backed utils.

** Cautious!! **
1. This util only works in cluster.
2. You'd better run it with things like `mapPartitions()` to gain best
   performace, as it processes looking up in batch mode.
3. The input should be generally a LIST. Avoid generators or iterators.
4. The output is generally a dict which may NOT contain all elements. You should
   handle missing data by yourself.
"""
#!/usr/bin/env python

import colored_glog as glog

from fueling.common.mongo_utils import Mongo
from modules.data.fuel.fueling.data.proto.record_meta_pb2 import RecordMeta


def lookup_existing_records(records, collection):
    """[record] -> [record which exists in DB]"""
    query = {'path': {'$in': records}}
    fields = {'path': 1}
    return [doc['path'] for doc in collection.find(query, fields)]


def lookup_header_for_records(records, collection):
    """[record] -> {record: record_header}"""
    query = {'path': {'$in': records}}
    fields = {'path': 1, 'header': 1}
    docs = collection.find(query, fields)
    return {doc['path']: Mongo.doc_to_pb(doc, RecordMeta()).header for doc in docs}


def lookup_stat_for_records(records, collection):
    """[record] -> {record: RecordMeta.Stat}"""
    query = {'path': {'$in': records}}
    fields = {'path': 1, 'stat': 1}
    docs = collection.find(query, fields)
    return {doc['path']: Mongo.doc_to_pb(doc, RecordMeta()).stat for doc in docs}


def lookup_map_for_dirs(record_dirs, collection):
    """[record_dir] -> {record_dir: map_name}."""
    query = {
        'dir': {'$in': record_dirs},
        'hmi_status.current_map': {'$exists': True},
    }
    fields = {'dir': 1, 'hmi_status.current_map': 1}
    docs = collection.find(query, fields)
    return {doc['dir']: doc['hmi_status']['current_map'] for doc in docs}


def lookup_vehicle_for_dirs(record_dirs, collection):
    """[record_dir] -> {record_dir: vehicle_name}."""
    query = {
        'dir': {'$in': record_dirs},
        'hmi_status.current_vehicle': {'$exists': True},
    }
    fields = {'dir': 1, 'hmi_status.current_vehicle': 1}
    docs = collection.find(query, fields)
    return {doc['dir']: doc['hmi_status']['current_vehicle'] for doc in docs}
