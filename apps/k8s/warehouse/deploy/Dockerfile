FROM apolloauto/apollo:dev-18.04-x86_64-20190822_1616

COPY apps/k8s/warehouse/deploy/conda-py37.yaml /tmp/conda-py37.yaml
RUN /usr/local/miniconda/bin/conda env update -f /tmp/conda-py37.yaml && \
    /usr/local/miniconda/bin/conda clean -a -y

COPY py_proto /apollo/py_proto
COPY fueling /apollo/modules/data/fuel/fueling
COPY apps/k8s/warehouse /apollo/modules/data/fuel/apps/k8s/warehouse
ENTRYPOINT [ "/apollo/modules/data/fuel/apps/k8s/warehouse/deploy/entrypoint.sh" ]
