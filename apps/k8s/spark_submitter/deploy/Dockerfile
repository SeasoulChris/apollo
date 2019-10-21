FROM ubuntu:18.04

# Basic tools.
RUN apt-get update -y && \
    apt-get install -y \
        curl \
        openjdk-8-jre \
        python3-pip \
        software-properties-common

# Kubectl.
RUN curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-add-repository "deb http://apt.kubernetes.io/ kubernetes-xenial main" && \
    apt-get update -y && \
    apt-get install -y kubectl

# Python libs.
RUN python3 -m pip install \
        absl-py \
        boto3 \
        flask \
        Flask-RESTful \
        protobuf \
        pymongo \
        pyspark==2.4.0

# Resource.
# Get dest path with `python -c "import site; print(site.getsitepackages()[0])"`
COPY apps/k8s/spark_submitter/deploy/spark-kubernetes_2.11-2.4.0.jar \
    /usr/local/lib/python3.6/dist-packages/pyspark/jars/
COPY kube.config /root/.kube/config

# Source.
COPY apps/k8s/spark_submitter/index.py /home/spark_submitter/
COPY apps/k8s/spark_submitter/spark_submit_arg_pb2.py /home/spark_submitter/
COPY fueling /home/spark_submitter/fueling

WORKDIR /home/spark_submitter
ENTRYPOINT ["python3", "index.py"]
