#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

openssl req -x509 -newkey rsa:2048 -nodes -out cert.pem -keyout key.pem -days 3650 -config ssl_req.conf -extensions 'v3_req'
