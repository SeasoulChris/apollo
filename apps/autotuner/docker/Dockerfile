FROM apolloauto/fuel-client:20200519_1105

WORKDIR /

# Install xvfb for orca in optuna to plot images
# details: https://github.com/plotly/orca#linux-troubleshooting-headless-server-configuration
RUN apt update -y && \
    apt install -y xvfb && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set up apollo
RUN mkdir -p /root/.cache && touch /root/.cache/.apollo_agreement.txt && \
    rm -rf /apollo && \
    git clone --depth 1 --branch master https://github.com/ApolloAuto/apollo.git

WORKDIR /apollo
RUN echo "source /apollo/scripts/apollo_base.sh" > ~/.bash_profile && \
    mkdir /fuel

ADD . /fuel
WORKDIR /fuel

ARG GITHUB_VERSION
ENV GITHUB_VERSION=$GITHUB_VERSION

RUN cat /home/libs/bash.rc >> ~/.bashrc && \
    ./tools/build.sh //fueling/learning/autotuner/...
