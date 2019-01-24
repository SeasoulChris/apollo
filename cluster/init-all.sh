#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"

set -e  # Fail on first error
set -x  # Verbose log

# Add dashboard.
kubectl create -f https://raw.githubusercontent.com/kubernetes/dashboard/master/aio/deploy/recommended/kubernetes-dashboard.yaml

# Add glusterfs.
kubectl apply -f glusterfs.yaml

# Add spark.
function AddClusterAdmin() {
    namespace=$1
    user=$2
    kubectl create serviceaccount --namespace ${namespace} ${user}
    kubectl create clusterrolebinding ${namespace}-${user}_cluster-role \
        --clusterrole=cluster-admin --serviceaccount=${namespace}:${user}
}
AddClusterAdmin default spark
