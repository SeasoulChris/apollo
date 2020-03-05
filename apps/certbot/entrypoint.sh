#!/usr/bin/env bash

DOMAIN="apollofuel.bceapp.com"
CONTAINER_PORT="8080"

CHAIN_FILE="/etc/letsencrypt/live/${DOMAIN}/fullchain.pem"
KEY_FILE="/etc/letsencrypt/live/${DOMAIN}/privkey.pem"
LOG_TO="/home/bae/log/certbot.log"

# Start file server.
python -m SimpleHTTPServer ${CONTAINER_PORT} &
sleep 1

# Get cert.
certbot certonly --webroot -w "$(pwd)" -d ${DOMAIN} -n -m "usa-data@baidu.com" --agree-tos

cat ${CHAIN_FILE} > ${LOG_TO}
cat ${KEY_FILE} >> ${LOG_TO}

fg 1
