FROM apolloauto/apollo:dev-x86_64-18.04-20200708_2316

RUN apt update -y && \
    apt install -y iputils-ping openjdk-8-jre vim \
        libunwind-dev \
        libpcap-dev \
        python3-pip && \
    apt clean -y

COPY docker/fuel.yaml /tmp/fuel.yaml
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    /usr/local/miniconda/bin/conda env update -f /tmp/fuel.yaml && \
    /usr/local/miniconda/bin/conda clean -a -y && \
    chmod -R 777 /usr/local/miniconda/envs/fuel

COPY default.txt /home/libs/pip-cache/default.txt
RUN /usr/bin/pip3 download -r /home/libs/pip-cache/default.txt -d /home/libs/pip-cache

ADD https://github.com/michael4338/rules_python/archive/8e9004ee8360d541abfcbecb60ba8a6902a53047.tar.gz \
    /home/libs/rules_python-8e9004ee8360d541abfcbecb60ba8a6902a53047.tar.gz
RUN chmod a+r /home/libs/*

COPY docker/bash.rc /home/libs/bash.rc
