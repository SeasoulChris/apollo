FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && \
    apt install -y certbot && \
    apt clean

COPY entrypoint.sh /home/bae/entrypoint.sh
ENTRYPOINT [ "/home/bae/entrypoint.sh" ]
