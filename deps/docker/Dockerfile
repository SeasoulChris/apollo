FROM apolloauto/apollo:dev-18.04-x86_64-20200316_1910

RUN apt update -y && \
    apt install -y iputils-ping openjdk-8-jre vim \
        libboost-all-dev \
        cmake \
        build-essential \
        libgoogle-glog-dev \
        libbz2-dev \
        libconsole-bridge-dev \
        libpoco-dev \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libyaml-cpp-dev \
        liblz4-dev \
        liblz4-tool \
        libreadline-dev && \
    apt clean -y


COPY fuel.yaml /tmp/fuel.yaml
RUN /usr/local/miniconda/bin/conda env update -f /tmp/fuel.yaml && \
    /usr/local/miniconda/bin/conda clean -a -y && \
    chmod -R 777 /usr/local/miniconda/envs/fuel

COPY py27_adbsdk.yaml /tmp/py27_adbsdk.yaml
RUN /usr/local/miniconda/bin/conda env update -f /tmp/py27_adbsdk.yaml && \
    /usr/local/miniconda/bin/conda clean -a -y && \
    chmod -R 777 /usr/local/miniconda/envs/py27_adbsdk

COPY bash.rc /home/libs/bash.rc

COPY installers/install_adbsdk.sh /tmp/installers/
RUN bash /tmp/installers/install_adbsdk.sh
