#!/usr/bin/env python
"""Kubectl utils."""

from datetime import timezone

from kubernetes import client, config

import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


# TODO(Andrew): Prefer Object-oriented API. Just gather them to a `class Kubectl`. And better to
# have an explicit `init()` function which load the config.


# uncomment the next line when test
# config.load_kube_config(config_file=file_utils.fuel_path('fueling/common/kubectl.conf'))
# comment the next line when test
config.load_kube_config()
coreV1Api = client.CoreV1Api()
appsV1Api = client.AppsV1Api()


def get_pods(name='', namespace='default'):
    """kubectl get pods"""
    # TODO(Andrew): Use list_namespaced_pod.
    ret = coreV1Api.list_pod_for_all_namespaces(watch=False)
    res = []
    for item in ret.items:
        itemname = item.metadata.name
        itemnamespace = item.metadata.namespace
        # TODO(Andrew): pod name is not guaranteed to follow some kind of pattern. So it may crash
        # with index-out-of-bound. I would prefer to get a list of items' metadata directly. And the
        # user of this function can parse fields as their wish.
        nodetype = itemname.split('-')[-1]
        owner = itemname.split('-')[2]
        phase = item.status.phase
        creation_timestamp = (item.metadata.creation_timestamp.replace(tzinfo=timezone.utc)
                              .timestamp())
        if name != '' and name != itemname:
            continue
        if namespace != '' and namespace != itemnamespace:
            continue
        # TODO(Andrew): In such case we can simply `yield` items.
        res.append({'name': itemname,
                    'owner': owner,
                    'nodetype': nodetype,
                    'phase': phase,
                    'namespace': itemnamespace,
                    'creation_timestamp': creation_timestamp})
        logging.info(F'{namespace} {name} {owner} {nodetype} {phase} {creation_timestamp}')
    return res


def logs(pod_name, namespace='default'):
    """kubectl logs"""
    full_log = coreV1Api.read_namespaced_pod_log(name=pod_name, namespace=namespace)
    return full_log


def delete_pods(name, namespace='default'):
    """delete pods"""
    res = coreV1Api.delete_namespaced_pod(name=name, namespace=namespace)
    return res


def get_deployments():
    """get deployments"""
    ret = appsV1Api.list_deployment_for_all_namespaces()
    res = []
    for item in ret.items:
        name = item.metadata.name
        namespace = item.metadata.namespace
        uid = item.metadata.uid
        creation_timestamp = (item.metadata.creation_timestamp.replace(tzinfo=timezone.utc)
                              .timestamp())
        res.append({'name': name,
                    'uid': uid,
                    'namespace': namespace,
                    'creation_timestamp': creation_timestamp})
        logging.info(F'{namespace} {name} {uid} {creation_timestamp}')
    return res


def delete_deployments(deployment_name, namespace='default'):
    """delete deployments"""
    res = appsV1Api.delete_namespaced_deployment(name=deployment_name, namespace=namespace)
    return res

