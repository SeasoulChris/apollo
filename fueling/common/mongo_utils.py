#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""
MongoDB util.

Requirements: pymongo 3.x
"""
import os
import sys

from absl import flags
import colored_glog as glog
import google.protobuf.json_format as json_format
import pymongo


flags.DEFINE_string(
    'mongo_url',
    'mongodb://bJVmYB0.mongodb.bj.baidubce.com:27017,bJVmYB1.mongodb.bj.baidubce.com:27017',
    'MongoDB url.')
flags.DEFINE_string('mongo_db_name', 'apollo', 'MongoDB DB name to access.')
flags.DEFINE_string('mongo_record_collection_name', 'records', 'MongoDB record collection name.')


class Mongo(object):
    """MongoDB util"""

    def __init__(self, flags_dict=None):
        if flags_dict is None:
            flags_dict = flags.FLAGS.flag_values_dict()
        if flags_dict.get('running_mode') == 'TEST':
            glog.error('MongoDB is not reachable in TEST mode.')
            return None

        self.url = flags_dict['mongo_url']
        self.db_name = flags_dict['mongo_db_name']
        self.record_collection_name = flags_dict['mongo_record_collection_name']

        self.user = os.environ.get('MONGO_USER')
        self.passwd = os.environ.get('MONGO_PASSWD')
        if not self.user or not self.passwd:
            glog.fatal('No credential found for MongoDB authentication.')
            sys.exit(1)

    def db_connection(self):
        """Create a connection to MongoDB instance."""
        db_connection = pymongo.MongoClient(self.url)[self.db_name]
        db_connection.authenticate(self.user, self.passwd)
        return db_connection

    def collection(self, collection_name):
        """
        Get collection handler. To use it, please refer
        https://api.mongodb.com/python/current/api/pymongo/collection.html
        """
        conn = self.db_connection()
        return conn[collection_name] if conn else None

    def record_collection(self):
        """Get record collection."""
        return self.collection(self.record_collection_name)

    @staticmethod
    def pb_to_doc(pb):
        """Convert proto to mongo document."""
        including_default_value_fields = False
        preserving_proto_field_name = True
        return json_format.MessageToDict(pb, including_default_value_fields,
                                         preserving_proto_field_name)

    @staticmethod
    def doc_to_pb(doc, pb):
        """Convert mongo document to proto."""
        ignore_unknown_fields = True
        return json_format.ParseDict(doc, pb, ignore_unknown_fields)


if __name__ == '__main__':
    def main(argv):
        mongo = Mongo(flags.FLAGS.flag_values_dict())
        glog.info(Mongo.db().collection_names())

    from absl import app
    app.run(main)
