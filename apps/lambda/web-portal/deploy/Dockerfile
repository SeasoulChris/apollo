FROM apolloauto/fuel-client:20191022_1411

ENV IN_CLIENT_DOCKER=true
ENV SUBMITTER=web-portal

RUN python3 -m pip install \
    absl-py \
    flask \
    flask-restful \
    gunicorn \
    protobuf \
    pyopenssl \
    requests

COPY py_proto /apollo/py_proto
COPY apps/local /apollo/modules/data/fuel/apps/local
COPY fueling/common /apollo/modules/data/fuel/fueling/common
COPY tools /apollo/modules/data/fuel/tools
COPY kube.config /root/.kube/config

COPY apps/lambda/web-portal/deploy/entrypoint.sh /home/bae/entrypoint.sh
ENTRYPOINT [ "/home/bae/entrypoint.sh" ]
