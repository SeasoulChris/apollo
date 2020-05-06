FROM simengineregistry.azurecr.io/cost_service:20200504_1613

RUN rm -rf /fuel/fueling && \
    rm -rf /fuel/apps
COPY fueling /fuel/fueling
COPY apps /fuel/apps


WORKDIR /fuel

RUN ./tools/build.sh //fueling/learning/autotuner/...
