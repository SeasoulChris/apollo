FROM ubuntu:18.04

# Basic tools.
RUN apt update -y && \
    apt install -y \
        curl \
        openjdk-8-jre \
        python3-pip \
        software-properties-common && \
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
    boto3 \
    flask \
    Flask-RESTful \
    protobuf \
    pymongo \
    pyspark==2.4.4

# Resource.
# Get dest path with `python -c "import site; print(site.getsitepackages()[0])"`
COPY spark-kubernetes_2.11-2.4.4.jar /usr/local/lib/python3.6/dist-packages/pyspark/jars/
COPY kube.config /root/.kube/config
COPY index.zip /home/spark_submitter/index.zip

WORKDIR /home/spark_submitter
ENTRYPOINT ["python3", "index.zip"]
