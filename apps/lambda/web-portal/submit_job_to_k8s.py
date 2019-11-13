#!/usr/bin/env python

from google.protobuf.json_format import MessageToJson
import requests


def submit_job(spark_submit_arg):
    """Submit a job through spark_submitter service."""
    SUBMITTER = ('http://localhost:8001/api/v1/namespaces/default/services/'
                 'http:spark-submitter-service:8000/proxy/')
    requests.post(SUBMITTER, json=MessageToJson(spark_submit_arg))
