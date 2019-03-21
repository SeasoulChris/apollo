#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""
MongoDB util.

Requirements: pymongo 3.x
"""
import os
import sys

import gflags
import google.protobuf.json_format as json_format
import pymongo


gflags.DEFINE_string('mongo_host', '127.0.0.1', 'MongoDB host ip.')
gflags.DEFINE_integer('mongo_port', 27017, 'MongoDB port.')
gflags.DEFINE_string('mongo_db_name', 'apollo', 'MongoDB db name.')
gflags.DEFINE_string('mongo_user', None, 'MongoDB user (optional).')
gflags.DEFINE_string('mongo_pass', None, 'MongoDB password (optional).')


class Mongo(object):
    """MongoDB util"""

    @staticmethod
    def db():
        """Connect to MongoDB instance."""
        # Try to read config from environ, and the flags.
        G = gflags.FLAGS
        host = os.environ.get('MONGO_HOST', G.mongo_host)
        port = int(os.environ.get('MONGO_PORT', G.mongo_port))
        user = os.environ.get('MONGO_USER', G.mongo_user)
        passwd = os.environ.get('MONGO_PASS', G.mongo_pass)

        client = pymongo.MongoClient(host, port)
        db = client[G.mongo_db_name]
        if user and passwd:
            db.authenticate(user, passwd)
        return db

    @staticmethod
    def collection(collection_name):
        """
        Get collection handler. To use it, please refer
        https://api.mongodb.com/python/current/api/pymongo/collection.html
        """
        return Mongo.db()[collection_name]

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
    gflags.FLAGS(sys.argv)
    print Mongo.db().collection_names()
