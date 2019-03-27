#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""
MongoDB util.

Requirements: pymongo 3.x
"""
import os
import sys

import google.protobuf.json_format as json_format
import pymongo

import fueling.common.colored_glog as glog


class Mongo(object):
    """MongoDB util"""
    URL='mongodb://bJVmYB0.mongodb.bj.baidubce.com:27017,bJVmYB1.mongodb.bj.baidubce.com:27017'
    DB='apollo'

    @staticmethod
    def db():
        """Connect to MongoDB instance."""
        user, passwd = os.environ.get('MONGO_USER'), os.environ.get('MONGO_PASSWD')
        if not user or not passwd:
            glog.fatal('No credential found for MongoDB authentication.')
            return None
        db_connection = pymongo.MongoClient(Mongo.URL)[Mongo.DB]
        db_connection.authenticate(user, passwd)
        return db_connection

    @staticmethod
    def collection(collection_name):
        """
        Get collection handler. To use it, please refer
        https://api.mongodb.com/python/current/api/pymongo/collection.html
        """
        db_connection = Mongo.db()
        return db_connection[collection_name] if db_connection else None

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
    print Mongo.db().collection_names()
