"""DB-backed utils. Only works in cluster."""
#!/usr/bin/env python

import colored_glog as glog

from fueling.common.mongo_utils import Mongo
from modules.data.fuel.fueling.data.proto.record_meta_pb2 import RecordMeta


def lookup_existing_records(records, collection):
    """[record] -> [record which exists in DB]"""
    query = {'path': {'$in': records}}
    fields = {'path': 1}
    return [doc['path'] for doc in collection.find(query, fields)]

def lookup_hmi_status_for_dirs(record_dirs, collection):
    """[record_dir] -> {record_dir: hmi_status}"""
    query = {'dir': {'$in': record_dirs}}
    fields = {'dir': 1, 'hmi_status': 1}
    dir_to_result = {}
    for doc in collection.find(query, fields):
        if doc['dir'] in dir_to_result:
            # Already found for this target.
            continue
        record_meta = Mongo.doc_to_pb(doc, RecordMeta())
        hmi_status = record_meta.hmi_status
        # Simple validate hmi_status.
        if hmi_status.current_mode or hmi_status.current_map:
            dir_to_result[record_meta.dir] = hmi_status
            glog.info('Got HMIStatus for task {}'.format(record_meta.dir))
    return dir_to_result

def lookup_map_for_dirs(record_dirs, collection):
    """[record_dir] -> {record_dir: map_name}."""
    dir_to_hmi_status = lookup_hmi_status_for_dirs(record_dirs, collection)
    return {key: hmi_status.current_map for key, hmi_status in dir_to_hmi_status.items()}

def lookup_vehicle_for_dirs(record_dirs, collection):
    """[record_dir] -> {record_dir: vehicle_name}."""
    dir_to_hmi_status = lookup_hmi_status_for_dirs(record_dirs, collection)
    return {key: hmi_status.current_vehicle for key, hmi_status in dir_to_hmi_status.items()}
