FROM k8s.gcr.io/spark
MAINTAINER Xiangquan Xiao <xiaoxiangquan@baidu.com>

RUN apt-get update -y && apt-get install -y \
    openssl \
    python-pip

RUN pip install --upgrade pip

RUN pip install \
    grpcio \
    protobuf
