FROM apolloauto/apollo:dev-x86_64-18.04-20200720_0236

ENV LD_LIBRARY_PATH=/usr/local/fast-rtps/lib

RUN apt update -y && \
    apt install -y \
        iputils-ping \
        libpcap-dev \
        libunwind-dev \
        openjdk-8-jre && \
    apt clean -y

COPY docker/fuel.yaml /tmp/fuel.yaml
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    /usr/local/miniconda/bin/conda env update -f /tmp/fuel.yaml && \
    /usr/local/miniconda/bin/conda clean -a -y && \
    chmod -R 777 /usr/local/miniconda/envs/fuel

COPY default.txt /home/libs/pip-cache/default.txt
RUN /usr/local/miniconda/envs/fuel/bin/pip3 download \
        -r /home/libs/pip-cache/default.txt \
        -d /home/libs/pip-cache && \
    chmod a+r /home/libs/*

COPY docker/bash.rc /home/libs/bash.rc
