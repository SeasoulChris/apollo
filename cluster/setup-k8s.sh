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

# Add docker secret.
DOCKER_REGISTRY_SERVER="docker.io"
DOCKER_USER="autoapollo"
DOCKER_PASSWORD="<TODO>"
DOCKER_EMAIL="apollo.baidu@gmail.com"

kubectl create secret docker-registry dockerhub.com \
    --docker-server="${DOCKER_REGISTRY_SERVER}" \
    --docker-username="${DOCKER_USER}" \
    --docker-password="${DOCKER_PASSWORD}" \
    --docker-email="${DOCKER_EMAIL}"
