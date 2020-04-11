FROM ubuntu:18.04

# Basic tools
RUN apt update -y && \
    apt install -y \
        curl \
        python-pip \
        software-properties-common && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Python libs
RUN pip install --no-cache-dir \
    absl-py \
    protobuf \
    grpcio \
    grpcio-tools

# Setup adbsdk

# Resource
COPY server /home/afs_data_service/

WORKDIR /home/afs_data_service
RUN export PYTHONPATH=/home/afs_data_service:$PYTHONPATH && \
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./afs_data_service.proto

ENTRYPOINT ["python", "server.py"]

