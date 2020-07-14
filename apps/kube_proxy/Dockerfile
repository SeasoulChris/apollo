FROM ubuntu:18.04
RUN apt update && \
    apt install -y curl
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/v1.18.0/bin/linux/amd64/kubectl
RUN chmod +x ./kubectl
RUN mv ./kubectl /usr/local/bin/kubectl

RUN useradd -u 500 -ms /bin/bash work
USER work
WORKDIR /home/work
RUN mkdir /home/work/.kube
COPY config /home/work/.kube/config
EXPOSE 30001
CMD ["kubectl", "proxy", "--address=0.0.0.0", "--port=30001", "--accept-hosts=.*"]
