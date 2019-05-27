#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

openssl req -x509 -newkey rsa:2048 -nodes -out cert.pem -keyout key.pem -days 3650 -extensions 'v3_req' \
    -config ssl-req.conf
