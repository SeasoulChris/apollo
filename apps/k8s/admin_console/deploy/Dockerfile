FROM ubuntu:18.04

RUN apt update && \
    apt install -y netbase python3-pip && \
    apt clean

RUN pip3 install --no-cache-dir \
        absl-py \
        ansi2html \
        eventlet \
        requests \
        flask \
        flask-socketio \
        gunicorn \
        kubernetes \
        matplotlib \
        mpld3 \
        protobuf \
        pymongo \
        pyproj \
        pytz \
        redis

COPY index.zip /home/admin_console/index.zip
ENV PYTHONPATH=/home/admin_console

WORKDIR /home/admin_console
ENTRYPOINT ["python3", "index.zip"]
