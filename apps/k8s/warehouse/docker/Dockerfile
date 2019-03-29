FROM apolloauto/apollo:dev-18.04-x86_64-test

COPY py_proto /apollo/py_proto
COPY fueling /apollo/modules/data/fuel/fueling
COPY apps/k8s/warehouse /apollo/modules/data/fuel/apps/k8s/warehouse
RUN /usr/local/miniconda2/bin/conda env update -f \
    /apollo/modules/data/fuel/apps/k8s/warehouse/docker/conda-py36.yaml

ENTRYPOINT [ "/apollo/modules/data/fuel/apps/k8s/warehouse/start-server.sh" ]
