FROM apolloauto/apollo:dev-18.04-x86_64-20200331_0146

RUN apt update -y && \
    apt install -y iputils-ping openjdk-8-jre vim \
        libunwind-dev \
        libpcap-dev && \
    apt clean -y


COPY fuel.yaml /tmp/fuel.yaml
RUN /usr/local/miniconda/bin/conda env update -f /tmp/fuel.yaml && \
    /usr/local/miniconda/bin/conda clean -a -y && \
    chmod -R 777 /usr/local/miniconda/envs/fuel

COPY bash.rc /home/libs/bash.rc

