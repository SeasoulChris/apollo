FROM ubuntu:18.04

RUN apt update && \
    apt install -y python3-pip && \
    apt clean

RUN pip3 install --no-cache-dir \
        requests \
        absl-py

COPY openservice_regression.zip /home/openservice_regression.zip
ENV PYTHONPATH /home
WORKDIR /home
ENTRYPOINT ["python3", "openservice_regression.zip"]
