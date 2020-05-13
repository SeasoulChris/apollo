#!/usr/bin/env python
"""Kubectl utils."""


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

    def get_pods(self, namespace='default'):
        """kubectl get pods
        return type: list[V1Pod]
        https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Pod.md
        """
        return self.coreV1Api.list_namespaced_pod(namespace=namespace).items

    def describe_pod(self, name, namespace='default'):
        """describe pod details
        return type: V1Pod
        https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/V1Pod.md
        """
        return self.coreV1Api.read_namespaced_pod(name=name, namespace=namespace)

    def logs(self, name, namespace='default'):
        """kubectl logs"""
        return self.coreV1Api.read_namespaced_pod_log(name=name, namespace=namespace)

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
