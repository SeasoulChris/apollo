# encoding=utf8

from datetime import datetime
from kubernetes import client, config

class Kubectl:
    """kubectl key opration"""

    def __init__(self):
        """init"""
        config.load_kube_config(config_file='kubectl.txt')
        self.coreV1Api = client.CoreV1Api()
        self.appsV1Api = client.AppsV1Api()

    def get_pods(self):
        """kubectl get pods"""
        ret = self.coreV1Api.list_pod_for_all_namespaces(watch=False)
        res = []
        for i in ret.items:
            name = i.metadata.name
            owner = name.split('-')[2]
            nodetype = name.split('-')[-1]
            phase = i.status.phase
            namespace = i.metadata.namespace
            start_datetime = i.metadata.creation_timestamp
            start_datetimestr = start_datetime.strftime('%Y%d%m %H:%M:%S')
            tz_info = start_datetime.tzinfo
            curr_datetime = datetime.now(tz_info)
            running_time_in_hours = '{:.2f} Hour'.format(
                (curr_datetime - start_datetime).seconds / 3600)
            running_time_in_minutes = '{:.2f} Min'.format(
                (curr_datetime - start_datetime).seconds / 60)
            res.append({'name': name,
                        'owner': owner,
                        'nodetype': nodetype,
                        'phase': phase,
                        'namespace': namespace,
                        'start_datetime': start_datetimestr,
                        'running_hours': running_time_in_hours,
                        'running_minutes': running_time_in_minutes})
            print(F'{namespace} {name} {owner} {nodetype} {phase} '
                  F'{start_datetimestr} {running_time_in_hours}')
        return res

    def logs(self, pod_name, namespace='default'):
        """kubectl logs"""
        full_log = self.coreV1Api.read_namespaced_pod_log(
            name=pod_name, namespace=namespace)
        return full_log

    def delete_pods(self, pod_name, namespace='default'):
        """delete pods"""
        res = self.coreV1Api.delete_namespaced_pod(name=pod_name, namespace=namespace)
        return res

    def get_deployments(self):
        """get deployments"""
        res = self.appsV1Api.list_deployment_for_all_namespaces()
        return res

    def delete_deployments(self, deployment_name, namespace='default'):
        """delete deployments"""
        res = self.appsV1Api.delete_namespaced_deployment(name=deployment_name, namespace=namespace)
        return res

if __name__ == '__main__':
    kubectl = Kubectl()
    res = kubectl.get_pods()
    print(res)
