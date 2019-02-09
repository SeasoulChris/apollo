#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"

set -e  # Fail on first error

# Add spark user.
function AddClusterAdmin() {
    namespace=$1
    user=$2
    kubectl create serviceaccount --namespace ${namespace} ${user}
    kubectl create clusterrolebinding ${namespace}-${user}_cluster-role \
        --clusterrole=cluster-admin --serviceaccount=${namespace}:${user}
}
AddClusterAdmin default spark
