FROM ubuntu:18.04

# Basic tools
RUN apt update -y && \
    apt install -y \
        build-essential \
        cmake \
        curl \
        libboost-all-dev \
        libboost-filesystem-dev \
        libboost-program-options-dev \
        libbz2-dev \
        libconsole-bridge-dev \
        libgoogle-glog-dev \
        libpoco-dev \
        liblz4-dev \
        liblz4-tool \
        libpython2.7-dev \
        libreadline-dev \
        libssl1.0.0 \
        libssl-dev \
        libyaml-cpp-dev \
        libyaml-dev \
        python-pip \
        software-properties-common && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Python libs
RUN pip install --no-cache-dir \
    absl-py \
    future \
    protobuf==3.12.2 \
    pyyaml \
    grpcio-tools==1.29.0

# Setup adbsdk
COPY adbsdk /home/afs_data_service/adbsdk
RUN cd /home/afs_data_service/adbsdk && python setup.py install

# Resource
COPY server /home/afs_data_service/
COPY entrypoint.sh /home/afs_data_service/entrypoint.sh

WORKDIR /home/afs_data_service
ENTRYPOINT ["/home/afs_data_service/entrypoint.sh"]
