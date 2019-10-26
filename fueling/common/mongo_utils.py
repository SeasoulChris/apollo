#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""
MongoDB util.

Requirements: pymongo 3.x
"""
import os

import google.protobuf.json_format as json_format
import pymongo

import fueling.common.logging as logging


MONGO_URL = 'mongodb://bJVmYB0.mongodb.bj.baidubce.com:27017,bJVmYB1.mongodb.bj.baidubce.com:27017'
MONGO_DB_NAME = 'apollo'


class Mongo(object):
    """MongoDB util"""

    def __init__(self):
        self.user = os.environ.get('MONGO_USER')
        self.passwd = os.environ.get('MONGO_PASSWD')
        if not self.user or not self.passwd:
            logging.error('No credential found for MongoDB authentication.')
            return None

    def db_connection(self):
        """Create a connection to MongoDB instance."""
        db_connection = pymongo.MongoClient(MONGO_URL)[MONGO_DB_NAME]
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
        return self.collection('records')

    def job_collection(self):
        """Get job collection."""
        return self.collection('jobs')


if __name__ == '__main__':
    def main(argv):
        logging.info(Mongo().db().collection_names())

    from absl import app
    app.run(main)
