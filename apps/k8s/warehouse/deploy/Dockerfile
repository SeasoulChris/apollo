FROM ubuntu:18.04

RUN apt update && \
    apt install -y netbase python3-pip && \
    apt clean

RUN pip3 install --no-cache-dir \
        absl-py \
        eventlet \
        flask \
        flask-socketio \
        gunicorn \
        matplotlib \
        mpld3 \
        protobuf \
        pymongo \
        pytz \
        redis

COPY index.zip /home/warehouse/index.zip
COPY py_proto /home/warehouse/
ENV PYTHONPATH=/home/warehouse

WORKDIR /home/warehouse
ENTRYPOINT ["python3", "index.zip"]
