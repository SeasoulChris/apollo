#!/usr/bin/env bash

JOB_NAME=$1

BLUE='\e[34m'
NO_COLOR='\033[0m'

# --accept-hosts='^*$' -p 8001 --address='0.0.0.0'
nohup kubectl proxy > /tmp/kubectl-proxy.log 2>&1 &
SECRET=$(kubectl get secret -n kube-system | grep kubernetes-dashboard-token | awk '{print $1}')
TOKEN=$(kubectl describe secret ${SECRET} -n kube-system | grep ^token)

while true; do
  if [ ! -f /tmp/spark-submit.log ]; then
    sleep 1
    continue
  fi

  grep 'phase: Running' /tmp/spark-submit.log > /dev/null
  if [ $? -ne 0 ]; then
    sleep 1
    continue
  fi

  DRIVER_SVC=$(kubectl get services | grep "${JOB_NAME}" | tail -n 1 | awk '{print $1}')
  echo ""
  echo -e "Kubernetes Dashboard:${BLUE} http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:443/proxy/#!/pod?namespace=default ${NO_COLOR}"
  echo -e "    Auth with${BLUE} ${TOKEN} ${NO_COLOR}"
  #echo ""
  #echo "Spark UI:"
  #echo -e "${BLUE}  http://localhost:8001/api/v1/namespaces/default/services/http:${DRIVER_SVC}:4040/proxy/ ${NO_COLOR}"
  exit 0
done
