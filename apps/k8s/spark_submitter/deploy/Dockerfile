FROM apolloauto/fuel-client:20200720_1535

# Basic tools.
RUN apt update -y && \
    apt install -y curl software-properties-common && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Kubectl.
#RUN curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
#    apt-add-repository "deb http://apt.kubernetes.io/ kubernetes-xenial main" && \
#    apt update -y && \
#    apt install -y kubectl && \
#    apt clean && rm -rf /var/lib/apt/lists/*

# download the latest kubectl binary file: curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl"
# download the file from your host to avoid the network issue
COPY kubectl /usr/local/bin/
RUN chmod +x /usr/local/bin/kubectl

# Resource.
# Get dest path with `python -c "import site; print(site.getsitepackages()[0])"`
COPY spark-kubernetes_2.12-3.0.0.jar \
    /usr/local/miniconda/envs/fuel/lib/python3.6/site-packages/pyspark/jars/
COPY kube.config /root/.kube/config
COPY index.zip /home/spark_submitter/index.zip

WORKDIR /home/spark_submitter
ENTRYPOINT ["bash", "-c", "--", "source /usr/local/miniconda/bin/activate fuel; python index.zip"]
