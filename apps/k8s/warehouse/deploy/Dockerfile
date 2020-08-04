FROM ubuntu:18.04

RUN apt update && \
    apt install -y netbase python3-pip && \
    apt clean

RUN pip3 install --no-cache-dir \
        absl-py \
        ansi2html \
        eventlet \
        flask \
        flask-socketio \
        gunicorn \
        kubernetes \
        matplotlib \
        mpld3 \
        protobuf \
        pymongo \
        pytz \
        redis

RUN touch /usr/local/lib/python3.6/dist-packages/google/__init__.py
COPY index.zip /home/warehouse/index.zip
ENV PYTHONPATH=/home/warehouse

WORKDIR /home/warehouse
ENTRYPOINT ["python3", "index.zip"]
