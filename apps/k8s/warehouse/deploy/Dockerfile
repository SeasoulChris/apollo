FROM apolloauto/apollo:dev-18.04-x86_64-test

COPY apps/k8s/warehouse/deploy/conda-py37.yaml /tmp/conda-py37.yaml
RUN /usr/local/miniconda2/bin/conda env update -f /tmp/conda-py37.yaml

COPY py_proto /apollo/py_proto
COPY fueling /apollo/modules/data/fuel/fueling
COPY apps/k8s/warehouse /apollo/modules/data/fuel/apps/k8s/warehouse
ENTRYPOINT [ "/apollo/modules/data/fuel/apps/k8s/warehouse/deploy/start-server.sh" ]
