#!/usr/bin/env bash

echo "Started dashboard at http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:443/proxy/#!/overview?namespace=default"
kubectl proxy
