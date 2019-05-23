FROM ubuntu:18.04

RUN apt-get update -y && \
    apt-get install -y openjdk-11-jre rsync wget zip && \
    wget -O install.sh "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    bash install.sh -b -p /usr/local/miniconda && \
    rm install.sh

COPY apps/lambda/bae-proxy/deploy/conda-py37.yaml /tmp/conda-py37.yaml
COPY tools/tool-env.yaml /tmp/tool-env.yaml
RUN /usr/local/miniconda/bin/conda env update -f /tmp/conda-py37.yaml && \
    /usr/local/miniconda/bin/conda env update -f /tmp/tool-env.yaml

COPY apps/lambda/bae-proxy /apollo/modules/data/fuel/apps/lambda/bae-proxy
COPY apps/local /apollo/modules/data/fuel/apps/local
COPY fueling /apollo/modules/data/fuel/fueling
COPY tools /apollo/modules/data/fuel/tools
COPY kube.config /root/.kube/config

ENTRYPOINT [ "/apollo/modules/data/fuel/apps/lambda/bae-proxy/entrypoint.sh" ]
