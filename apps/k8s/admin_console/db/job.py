#!/usr/bin/env python3
"""
Mongodb job module
"""

from db import base


class MongoJob(base.MongoBase):
    """
    MongoJob class including collection of job curd
    """

    def __init__(self, collection_name):
        super().__init__()
        self.collection = getattr(self.db, collection_name)

    def insert(self, target_dict):
        self.collection.insert(target_dict)

    def update(self, old_dict, new_dict):
        self.collection.update(old_dict, new_dict)

    def delete(self, target_dict):
        self.collection.remove(target_dict)

    def find_all(self):
        for job_data in self.collection.find():
            job_data['_id'] = job_data['_id'].__str__()
            yield job_data
