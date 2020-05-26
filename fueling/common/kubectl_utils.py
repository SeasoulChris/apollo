#!/usr/bin/env python
"""Kubectl utils."""

import fnmatch

from kubernetes import client, config


class Kubectl(object):
    """kubectl basic operations"""

    def __init__(self, config_file=None):
        """init"""
        if config_file:
            config.load_kube_config(config_file)
        else:
            config.load_incluster_config()
        self.coreV1Api = client.CoreV1Api()
        self.appsV1Api = client.AppsV1Api()
        self.apiClient = client.ApiClient()

    def get_pods(self, namespace='default', tojson=False):
        """kubectl get pods
        return type: list[V1Pod] if not tojson else json
        https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Pod.md
        """
        pods_info = self.coreV1Api.list_namespaced_pod(namespace=namespace).items
        if not tojson:
            return pods_info
        else:
            return self.apiClient.sanitize_for_serialization(pods_info)

    def get_pods_by_pattern(self, pattern, namespace='default'):
        """Similar to get_pods() but filter by fnmatch-style pattern."""
        return [pod for pod in self.get_pods(namespace)
                if fnmatch.fnmatch(pod.metadata.name, pattern)]

    def describe_pod(self, name, namespace='default', tojson=False):
        """describe pod details
        return type: V1Pod if not tojson else json
        https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Pod.md
        """
        descinfo = self.coreV1Api.read_namespaced_pod(name=name, namespace=namespace)
        if not tojson:
            return descinfo
        else:
            return self.apiClient.sanitize_for_serialization(descinfo)

    def logs(self, name, namespace='default'):
        """ kubectl logs
        return a string
        """
        return self.coreV1Api.read_namespaced_pod_log(name=name, namespace=namespace)

    def log_stream(self, name, namespace='default'):
        """kubectl logs
        return a generator of the logs
        """
        return self.coreV1Api.read_namespaced_pod_log(
            name=name, namespace=namespace, follow=True,
            _preload_content=False, _request_timeout=10).stream()

    def delete_pod(self, name, namespace='default'):
        """delete pods
        return type: V1Status
        https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Status.md
        """
        return self.coreV1Api.delete_namespaced_pod(name=name, namespace=namespace)

    def get_deployments(self):
        """get deployments
        return type: list[V1Deployment]
        https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Deployment.md
        """
        return self.appsV1Api.list_deployment_for_all_namespaces().items

    def delete_deployments(self, deployment_name, namespace='default'):
        """delete deployments
        return type: V1Status
        https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Status.md
        """
        return self.appsV1Api.delete_namespaced_deployment(name=deployment_name,
                                                           namespace=namespace)
