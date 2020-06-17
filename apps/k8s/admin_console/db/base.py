#!/usr/bin/env python3
"""
Basic mongoDB module
"""

import pymongo

import application


class MongoBase(object):
    """
    MongoDB's basic class, including connection method and db
    """

    def __init__(self):
        self.host = application.app.config.get("DB_HOST")
        self.port = application.app.config.get("DB_PORT")

    def connection(self):
        return pymongo.MongoClient(host=self.host, port=self.port)

    @property
    def db(self):
        return getattr(self.connection(), application.app.config.get("DB_NAME"))
