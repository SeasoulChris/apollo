FROM apolloauto/fuel-client:20190821_1718

ENV IN_CLIENT_DOCKER=true
ENV SUBMITTER=bae-proxy

COPY apps/lambda/bae-proxy/deploy/py3-requirement.txt /tmp/py3-requirement.txt
RUN python3 -m pip install -r /tmp/py3-requirement.txt

COPY py_proto /apollo/py_proto
COPY apps/local /apollo/modules/data/fuel/apps/local
COPY fueling/common /apollo/modules/data/fuel/fueling/common
COPY tools /apollo/modules/data/fuel/tools
COPY kube.config /root/.kube/config

COPY apps/lambda/bae-proxy/deploy/entrypoint.sh /home/bae/entrypoint.sh
ENTRYPOINT [ "/home/bae/entrypoint.sh" ]
