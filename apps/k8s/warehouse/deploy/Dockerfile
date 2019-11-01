FROM ubuntu:18.04

RUN apt update && \
    apt install -y netbase python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        absl-py \
        eventlet \
        flask \
        flask-restful \
        flask-socketio \
        gunicorn \
        matplotlib \
        mpld3 \
        protobuf \
        pymongo \
        pytz \
        redis

COPY apps/k8s/warehouse /home/warehouse
COPY fueling /home/warehouse/fueling
COPY py_proto /home/warehouse/

WORKDIR /home/warehouse
ENTRYPOINT ["python3", "index.py"]
