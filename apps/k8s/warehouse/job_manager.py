#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""Manager for fuel jobs"""

from datetime import timezone
import datetime
import dateutil.parser
import json

from ansi2html import Ansi2HTMLConverter

# uncomment below for testing
# import fueling.common.file_utils as file_utils
from fueling.common.kubectl_utils import Kubectl
from fueling.common.mongo_utils import Mongo

TIMEZONE = 'America/Los_Angeles'


class JobManager(object):
    """Handle all the job related stuffs"""
    def __init__(self):
        """init class"""
        # apply the kubenetes config file
        # uncomment below for testing
        # self.kubectl = Kubectl(file_utils.fuel_path('apps/k8s/warehouse/kubectl.conf'))
        # comment below for testing
        self.kubectl = Kubectl()
        self.mongo = Mongo()

    def __extract_pod_info(self, pod):
        """extract pod data from json"""
        namespace = pod['metadata']['namespace']
        podname = pod['metadata']['name']
        phase = pod['status']['phase']
        creation_timestamp = (dateutil.parser.parse(pod['metadata']['creationTimestamp'])
                              .replace(tzinfo=timezone.utc))
        curr_datetime = datetime.datetime.now(timezone.utc)
        duration_ns = (curr_datetime - creation_timestamp).seconds * 1e9
        pod_uid = pod['metadata']['uid']
        parent_uid = ''
        spark_job_owner = ''
        spark_job_id = ''
        # only executor has an parent_id indicating the uid of corresponding driver
        if 'ownerReferences' in pod['metadata']:
            parent_uid = pod['metadata']['ownerReferences'][0]['uid']
        else:
            # extract job_owner for future filtering
            try:
                for env in pod['spec']['containers'][0]['env']:
                    if env['name'] == 'PYSPARK_APP_ARGS':
                        for v in env['value'].split(' '):
                            if '=' in v:
                                arg_type, arg_value = v.split('=')
                                if arg_type == '--job_owner':
                                    spark_job_owner = arg_value
                                elif arg_type == '--job_id':
                                    spark_job_id = arg_value
            except Exception as ex:
                pass
        return {'namespace': namespace,
                'podname': podname,
                'phase': phase,
                'creation_timestamp': creation_timestamp,
                'duration_ns': duration_ns,
                'pod_uid': pod_uid,
                'parent_uid': parent_uid,
                'spark_job_id': spark_job_id,
                'spark_job_owner': spark_job_owner}

    def __save_pod_info(self, podinfo, podfrom, jobs_dict, kubectl_jobs_set):
        """save pod info to dict
        podfrom: ['kubectl', 'mongodb']
        Notice: load the kubectl jobs frist, then mongodb jobs
        """
        # only display fuel jobs
        if not podinfo['podname'].startswith('job-'):
            return
        # executor
        if podinfo['parent_uid'] != '':
            appuid = podinfo['parent_uid']
        # driver
        else:
            appuid = podinfo['pod_uid']
        # skip mongodb jobs which were already loaded from kubectl
        if podfrom == 'mongodb':
            if appuid in kubectl_jobs_set:
                return
        if appuid not in jobs_dict:
            jobs_dict[appuid] = {}
            jobs_dict[appuid]['pods'] = []
            # save jobs loaded from kubectl
            if podfrom == 'kubectl':
                kubectl_jobs_set.add(appuid)
        # job(same to driver) info
        if podinfo['parent_uid'] == '':
            jobs_dict[appuid]['spark_job_owner'] = podinfo['spark_job_owner']
            jobs_dict[appuid]['spark_job_id'] = podinfo['spark_job_id']
            jobs_dict[appuid]['namespace'] = podinfo['namespace']
            jobs_dict[appuid]['name'] = podinfo['podname']
            jobs_dict[appuid]['creation_timestamp'] = podinfo['creation_timestamp'].timestamp()
            jobs_dict[appuid]['phase'] = podinfo['phase']
        # driver & executor pod info
        jobs_dict[appuid]['pods'].append({
            'podname': podinfo['podname'],
            'phase': podinfo['phase'],
            'creation_timestamp': podinfo['creation_timestamp'].timestamp(),
            'duration_ns': podinfo['duration_ns']
        })

    def job_list(self):
        """list all the jobs"""
        jobs_dict = {}
        kubectl_jobs_set = set()

        # load jobs from kubectl api
        kubectljobs = self.kubectl.get_pods(namespace='default', tojson=True) or []
        for kubectljob in kubectljobs:
            self.__save_pod_info(self.__extract_pod_info(kubectljob), 'kubectl', jobs_dict,
                                 kubectl_jobs_set)

        # load jobs from mongodb
        mongojobs = self.mongo.job_log_collection().find({}, {'logs': 0, '_id': 0}) or []
        for mongojob in mongojobs:
            self.__save_pod_info(self.__extract_pod_info(json.loads(mongojob['desc'])), 'mongodb',
                                 jobs_dict, kubectl_jobs_set)

        sorted_job_list = sorted(
            list(jobs_dict.items()),
            key=lambda x: x[1]['creation_timestamp'],
            reverse=True)

        return sorted_job_list

    def pod_describe(self, pod_name, namespace='default'):
        """Handler of the pod info"""
        mongodesc = self.mongo.job_log_collection().find_one({'pod_name': pod_name,
                                                             'namespace': namespace},
                                                             {'desc': 1, '_id': 0})
        if mongodesc:
            return mongodesc['desc']
        return json.dumps(self.kubectl.describe_pod(pod_name, namespace, tojson=True),
                          sort_keys=True, indent=4, separators=(', ', ': '))

    def pod_delete(self, pod_name, namespace='default'):
        """delete pod"""
        return self.kubectl.delete_pod(pod_name, namespace)

    def pod_log_streaming(self, pod_name, namespace='default'):
        """Handler of the pod streaming log"""
        conv = Ansi2HTMLConverter()

        def decorate_logs(generator):
            """decorate logs"""
            for log in generator:
                log = log.decode('utf-8')
                yield conv.convert(ansi=log, full=False)
        mongologs = self.mongo.job_log_collection().find_one({'pod_name': pod_name,
                                                             'namespace': namespace})
        if mongologs:
            # if the log exists in mongodb, return full log directly
            mongologstext = mongologs['logs'] + '\ncurrent log was loaded from database.'
            return conv.convert(ansi=mongologstext, full=False)
        logs_generator = None
        try:
            logs_generator = self.kubectl.log_stream(pod_name, namespace)
        except Exception as ex:
            pass
        # if the log is loaded from kubernetes streaming api, return a generator
        if logs_generator:
            return decorate_logs(logs_generator)
        else:
            return 'pod does not exist or has been deleted'
