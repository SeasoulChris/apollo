FROM apolloauto/apollo:base-18.04-x86_64-20191111_1530

WORKDIR /
# Set up apollo
RUN mkdir -p /root/.cache && touch /root/.cache/.apollo_agreement.txt && \
    rm -rf /apollo && \
    git clone --shallow-since=2019-08-13 --single-branch --branch master https://github.com/ApolloAuto/apollo.git

# Install fuse and bosfs
RUN apt-get update && apt-get -y install autotools-dev automake uuid-dev libssl-dev libcurl4-openssl-dev && \
    wget http://sdk.bce.baidu.com/console-sdk/fuse-2.9.4.tar.gz && tar zxf fuse-2.9.4.tar.gz && \
    wget http://sdk.bce.baidu.com/console-sdk/bosfs-1.0.0.9.tar.gz && tar zxf bosfs-1.0.0.9.tar.gz
WORKDIR /fuse-2.9.4
RUN ./configure && sudo make && sudo make install && fusermount -V
WORKDIR /bosfs-1.0.0
RUN bash build.sh && bosfs -v


WORKDIR /apollo

RUN echo "source /apollo/scripts/apollo_base.sh" > ~/.bash_profile && \
    mkdir /apollo/modules/data/fuel

ADD . /apollo/modules/data/fuel

# Generate required py_proto & cyber lib
RUN bash apollo.sh build_py && \
    bazel build -c opt //cyber/py_wrapper:_cyber_record.so //cyber/py_wrapper:_cyber_record_py3.so && git checkout -- .


# Append conda path and prepare conda package env
ENV PATH $PATH:/apollo/scripts:/usr/local/miniconda/bin
RUN conda config --add channels conda-forge && \
    conda env update --prune -f /apollo/modules/data/fuel/conda/py36.yaml
