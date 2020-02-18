FROM ubuntu:18.04

# Basic tools.
RUN apt update -y && \
    apt install -y \
        curl \
        openjdk-8-jre \
        python3-pip \
        software-properties-common \
        zip && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Kubectl.
RUN curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-add-repository "deb http://apt.kubernetes.io/ kubernetes-xenial main" && \
    apt update -y && \
    apt install -y kubectl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Python libs.
RUN pip3 install --no-cache-dir \
    absl-py \
    flask \
    gunicorn \
    protobuf \
    pymongo \
    pyopenssl \
    requests

# Resource.
COPY entrypoint.sh /home/bae/entrypoint.sh
COPY kube.config /root/.kube/config

ENTRYPOINT [ "/home/bae/entrypoint.sh" ]
