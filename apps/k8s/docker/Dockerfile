FROM apolloauto/apollo:dev-18.04-x86_64-test

COPY apps/k8s/docker/conda-py36.yaml /tmp/conda-py36.yaml
RUN /usr/local/miniconda2/bin/conda env update -f /tmp/conda-py36.yaml

COPY apps/k8s/docker/py_proto /apollo/py_proto
COPY fueling /apollo/modules/data/fuel/fueling
COPY apps/k8s/warehouse /apollo/modules/data/fuel/apps/k8s/warehouse

ENTRYPOINT [ "/apollo/modules/data/fuel/apps/k8s/warehouse/start-server.sh" ]
