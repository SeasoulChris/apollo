#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

COUNTRY="CN"
STATE="Beijing"
LOCALITY="Beijing"
ORGANIZATION="Baidu"
UNIT="IDG"
COMMON_NAME="apollofuel.bceapp.com"
VALID_DAYS="3650"

openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days ${VALID_DAYS} \
   -subj "/C=${COUNTRY}/ST=${STATE}/L=${LOCALITY}/O=${ORGANIZATION}/OU=${UNIT}/CN=${COMMON_NAME}"
