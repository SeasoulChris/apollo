FROM ___FUEL_DOCKER_VERSION__

ARG GITHUB_VERSION
ENV GITHUB_VERSION=$GITHUB_VERSION

WORKDIR /

# Install xvfb for orca in optuna to plot images
# details: https://github.com/plotly/orca#linux-troubleshooting-headless-server-configuration
RUN apt update -y && \
    apt install -y xvfb && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Adding the grpc_health_probe
RUN GRPC_HEALTH_PROBE_VERSION=v0.3.2 && \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /bin/grpc_health_probe

# Set up apollo
RUN mkdir -p /root/.cache && touch /root/.cache/.apollo_agreement.txt && \
    rm -rf /apollo && \
    git clone --depth 1 --branch master https://github.com/ApolloAuto/apollo.git

WORKDIR /apollo
RUN echo "source /apollo/scripts/apollo_base.sh" > ~/.bash_profile && \
    mkdir -p /fuel/local

ADD . /fuel
WORKDIR /fuel

RUN cat /home/libs/bash.rc >> ~/.bashrc && \
    echo "source /apollo/scripts/apollo_base.sh" > ~/.bash_profile && \
    echo "startup --output_user_root=/fuel/local/bazel_cache" > local/user.bazelrc && \
    ./tools/build_local.sh //fueling/learning/autotuner/...
