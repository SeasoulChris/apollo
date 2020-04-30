#!/usr/bin/env python
"""Kubectl utils."""
# encoding=utf8

from datetime import datetime

from kubernetes import client, config


class Kubectl(object):
    """kubectl key operation"""

    def __init__(self):
        """init"""
        config.load_kube_config(config_file='kubectl.conf')
        self.coreV1Api = client.CoreV1Api()
        self.appsV1Api = client.AppsV1Api()

    def get_pods(self, name='', namespace='default'):
        """kubectl get pods"""
        ret = self.coreV1Api.list_pod_for_all_namespaces(watch=False)
        res = []
        for item in ret.items:
            itemname = item.metadata.name
            itemnamespace = item.metadata.namespace
            nodetype = itemname.split('-')[-1]
            owner = itemname.split('-')[2]
            phase = item.status.phase
            start_datetime = item.metadata.creation_timestamp
            start_datetimestr = start_datetime.strftime('%Y%d%m %H:%M:%S')
            tz_info = start_datetime.tzinfo
            curr_datetime = datetime.now(tz_info)
            running_time_in_seconds = (curr_datetime - start_datetime).seconds
            running_time_in_hours = '{:.2f} Hour'.format(running_time_in_seconds / 3600)
            running_time_in_minutes = '{:.2f} Min'.format(running_time_in_seconds / 60)
            running_time_show = running_time_in_hours if running_time_in_seconds >= 3600 \
                else running_time_in_minutes
            if name != '' and name != itemname:
                continue
            if namespace != '' and namespace != itemnamespace:
                continue
            res.append({'name': itemname,
                        'owner': owner,
                        'nodetype': nodetype,
                        'phase': phase,
                        'namespace': itemnamespace,
                        'start_datetime': start_datetimestr,
                        'running_time': running_time_show,
                        'running_time_in_seconds': running_time_in_seconds})
            print(F'{namespace} {name} {owner} {nodetype} {phase} {running_time_show}')
        return res

    def logs(self, pod_name, namespace='default'):
        """kubectl logs"""
        full_log = self.coreV1Api.read_namespaced_pod_log(
            name=pod_name, namespace=namespace)
        return full_log

    def delete_pods(self, name, namespace='default'):
        """delete pods"""
        res = self.coreV1Api.delete_namespaced_pod(name=name, namespace=namespace)
        return res

    def get_deployments(self):
        """get deployments"""
        ret = self.appsV1Api.list_deployment_for_all_namespaces()
        res = []
        for item in ret.items:
            name = item.metadata.name
            namespace = item.metadata.namespace
            uid = item.metadata.uid
            start_datetime = item.metadata.creation_timestamp
            start_datetimestr = start_datetime.strftime('%Y%d%m %H:%M:%S')
            tz_info = start_datetime.tzinfo
            curr_datetime = datetime.now(tz_info)
            running_time_in_seconds = (curr_datetime - start_datetime).seconds
            running_time_in_hours = '{:.2f} Hour'.format(running_time_in_seconds / 3600)
            running_time_in_minutes = '{:.2f} Min'.format(running_time_in_seconds / 60)
            running_time_show = running_time_in_hours if running_time_in_seconds >= 3600 \
                else running_time_in_minutes
            res.append({'name': name,
                        'uid': uid,
                        'namespace': namespace,
                        'start_datetime': start_datetimestr,
                        'running_time': running_time_show,
                        'running_time_in_seconds': running_time_in_seconds})
            print(F'{namespace} {name} {uid} {start_datetimestr} {running_time_show}')
        return res

    def delete_deployments(self, deployment_name, namespace='default'):
        """delete deployments"""
        res = self.appsV1Api.delete_namespaced_deployment(name=deployment_name, namespace=namespace)
        return res


if __name__ == '__main__':
    kubectl = Kubectl()
    res = kubectl.get_pods()
    print(res)
