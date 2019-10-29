#!/usr/bin/env python3

"""Sqlite DB lightweight app"""

import os
import sys

import colored_glog as glog
import sqlite3
import textwrap


class SqlLite3_DB(object):
    """SqlLite DB utils"""
    _db_file_path = '/home/apollo/Documents/AutoCopy/mounted_disk_db.sqlite'
    _table_name = 'mounted_disks'

    @staticmethod
    def search_tasks_by_status(status):
        """Search DB for tasks with given status"""
        sql = 'SELECT * FROM {} WHERE status="{}"'.format(SqlLite3_DB._table_name, status.name)
        return SqlLite3_DB._execute_sql(sql)

    @staticmethod
    def update(task_path, status):
        """Update a row"""
        sql = 'UPDATE {} SET status = "{}" WHERE path="{}"'.format(
            SqlLite3_DB._table_name, status.name, task_path)
        SqlLite3_DB._execute_sql(sql)

    @staticmethod
    def delete(task_path):
        """Delete a row"""
        sql = 'DELETE FROM {} WHERE path="{}"'.format(SqlLite3_DB._table_name, task_path)
        SqlLite3_DB._execute_sql(sql)

    @staticmethod
    def insert(task_path, status):
        """Insert a row"""
        sql = 'INSERT INTO {} (path, status) VALUES ("{}","{}")'.format(
            SqlLite3_DB._table_name, task_path, status)
        SqlLite3_DB._execute_sql(sql)

    @staticmethod
    def search_all():
        """Search all rows"""
        sql = 'SELECT * FROM {}'.format(SqlLite3_DB._table_name)
        return SqlLite3_DB._execute_sql(sql)

    @staticmethod
    def create_table():
        """Create the default table"""
        sql = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS %(table_name)s
        (
          %(column_1)s integer PRIMARY KEY AUTOINCREMENT,
          %(column_2)s text NOT NULL UNIQUE,
          %(column_3)s text NOT NULL);
        """ % {
            'table_name': SqlLite3_DB._table_name,
            'column_1': 'id',
            'column_2': 'path',
            'column_3': 'status',
        })
        SqlLite3_DB._execute_sql(sql)

    @staticmethod
    def _execute_sql(sql):
        """Create connection and execute sql query"""
        connection = sqlite3.connect(SqlLite3_DB._db_file_path)
        cur = connection.cursor()
        cur.execute(sql)
        result = cur.fetchall()
        connection.commit()
        connection.close()
        return result


if __name__ == '__main__':
    if len(sys.argv) < 3:
        glog.error('Invalid Arguments.')
        glog.info('Usage: python3 Sqliter.py <remove/insert/searchall> /dev/sdb2')
        exit(1)

    opr = sys.argv[1]
    task = sys.argv[2]
    SqlLite3_DB.create_table()

    if opr == 'remove' or opr == 'delete':
        SqlLite3_DB.delete(task)
        glog.info('Removed {} from DB'.format(task))
    elif opr == 'insert':
        SqlLite3_DB.insert(task, 'INITIAL')
        glog.info('Inserted {} into DB'.format(task))

    glog.info('\nDB rows now:\n {}'.format(SqlLite3_DB.search_all()))
