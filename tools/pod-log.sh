#!/usr/bin/env bash

POD=$1

kubectl logs -f ${POD} | grep -v '^20'
