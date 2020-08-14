#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""
MongoDB util.

Requirements: pymongo 3.x
"""
import os

import pymongo

import fueling.common.logging as logging


MONGO_URL = ('mongodb://auPpfadGy.mongodb.bj.baidubce.com:27017,'
             'auPpfaZwf.mongodb.bj.baidubce.com:27017')
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

    def job_log_collection(self):
        """Get job log collection."""
        return self.collection('job_logs')

    def fuel_job_collection(self):
        """Get fuel job collection"""
        return self.collection('fuel_job')

    def admin_collection(self):
        """Get admin_console admin collection"""
        return self.collection('dkit_admins')

    def account_collection(self):
        """Get admin_console account collection"""
        return self.collection('dkit_accounts')

    def account_suffix_collection(self):
        """Get admin_console account suffix collection"""
        return self.collection("dkit_account_suffix")


if __name__ == '__main__':
    def main(argv):
        logging.info(Mongo().db().collection_names())

    from absl import app
    app.run(main)
