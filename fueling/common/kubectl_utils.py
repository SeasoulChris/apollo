#!/usr/bin/env python
"""Kubectl utils."""


from kubernetes import client, config


class Kubectl:
    """kubectl basic operations"""
    def __init__(self, config_file=None):
        """init"""
        config.load_kube_config(config_file)
        self.coreV1Api = client.CoreV1Api()
        self.appsV1Api = client.AppsV1Api()

    def get_pods(self, namespace='default'):
        """kubectl get pods"""
        ret = self.coreV1Api.list_namespaced_pod(namespace=namespace)
        for item in ret.items:
            yield item

    def logs(self, pod_name, namespace='default'):
        """kubectl logs"""
        full_log = self.coreV1Api.read_namespaced_pod_log(name=pod_name, namespace=namespace)
        return full_log

    def delete_pods(self, name, namespace='default'):
        """delete pods"""
        res = self.coreV1Api.delete_namespaced_pod(name=name, namespace=namespace)
        return res

    def get_deployments(self):
        """get deployments"""
        ret = self.appsV1Api.list_deployment_for_all_namespaces()
        for item in ret.items:
            yield item

    def delete_deployments(self, deployment_name, namespace='default'):
        """delete deployments"""
        res = self.appsV1Api.delete_namespaced_deployment(name=deployment_name, namespace=namespace)
        return res

