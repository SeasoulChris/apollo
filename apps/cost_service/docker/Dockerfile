FROM apolloauto/fuel-client:20200211_1459

WORKDIR /
# Set up apollo
RUN mkdir -p /root/.cache && touch /root/.cache/.apollo_agreement.txt && \
    rm -rf /apollo && \
    git clone --depth 1 --branch bazel2.x https://github.com/ApolloAuto/apollo.git

WORKDIR /apollo
RUN echo "source /apollo/scripts/apollo_base.sh" > ~/.bash_profile && \
    mkdir /fuel

ADD . /fuel
WORKDIR /fuel

RUN cat /home/libs/bash.rc >> ~/.bashrc && \
    ./tools/build_apollo.sh && \
    bazel build //fueling/learning/autotuner/...
