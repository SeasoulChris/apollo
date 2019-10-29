#!/usr/bin/env python

from flask_restful import Resource

from fueling.common.mongo_utils import Mongo
from modules.data.fuel.fueling.data.proto.record_meta_pb2 import RecordMeta
import fueling.common.proto_utils as proto_utils
import fueling.common.redis_utils as redis_utils


class MapLookup(Resource):
    """Map lookup service"""

    def get(self, lat, lon):
        redis_utils.redis_incr('apps.warehouse.pv.map_lookup')
        try:
            lat = float(lat)
            lon = float(lon)
        except BaseException:
            return 'Error: Cannot parse input as float numbers.'

        query = {
            'path': {'$regex': '^/mnt/bos/public-test/.*'},
            'stat.driving_path': {'$exists': True},
        }
        fields = {
            'path': 1,
            'stat.driving_path': 1,
        }

        result = []
        for doc in Mongo().record_collection().find(query, fields):
            record_meta = proto_utils.dict_to_pb(doc, RecordMeta())
            points = record_meta.stat.driving_path
            min_diff = 1.0
            min_index = -1
            for index, point in enumerate(points):
                diff = (lat - point.lat) ** 2 + (lon - point.lon) ** 2
                if diff < min_diff:
                    min_diff = diff
                    min_index = index
            if min_diff < 1e-7:
                result.append((record_meta.path, float(min_index) / len(points)))
        return sorted(result, reverse=True)
