FROM apolloauto/apollo:dev-x86_64-20181119_1800

# Add user.
ENV USER=apollo
RUN adduser --disabled-password --force-badname --gecos '' ${USER} && \
    usermod -aG sudo ${USER} && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Add tools and libs.
RUN pip install h5py redis

# Add dependent data.
# Note that --chown argument cannot understand ${USER} env.
COPY --chown=apollo:apollo local_integ /usr/local/apollo/local_integ
COPY --chown=apollo:apollo pod_scripts /home/${USER}/pod_scripts
# Add dynamic-model.
COPY --chown=apollo:apollo apollo-dynamic-model /baidu/adu-lab/apollo-dynamic-model
RUN bash /baidu/adu-lab/apollo-dynamic-model/dynamic_model.sh compile_proto

# Prepare apollo environment.
COPY --chown=apollo:apollo apollo /apollo
RUN rm -fr /apollo/modules/map/data && \
    ln -s /mnt/nfs/job-memory/baidu/adu-lab/apollo-map /apollo/modules/map/data
COPY --chown=apollo:apollo apollo-fuel /apollo/modules/data/fuel

# Disable AINFO/AWARN/AERROR in pipeline.
RUN chown ${USER}:${USER} /apollo && \
    sed -i "s/define AINFO ALOG_MODULE(MODULE_NAME, INFO)/define AINFO ADEBUG_MODULE(MODULE_NAME)/g" /apollo/cyber/common/log.h && \
    sed -i "s/define AWARN ALOG_MODULE(MODULE_NAME, WARN)/define AWARN ADEBUG_MODULE(MODULE_NAME)/g" /apollo/cyber/common/log.h && \
    sed -i "s/define AERROR ALOG_MODULE(MODULE_NAME, ERROR)/define AERROR ADEBUG_MODULE(MODULE_NAME)/g" /apollo/cyber/common/log.h

USER ${USER}
RUN echo "source /apollo/scripts/apollo_base.sh" >> /home/${USER}/.bashrc && \
    mkdir -p /home/${USER}/.cache && touch /home/${USER}/.cache/.apollo_agreement.txt

# Build apollo with minimum targets for the pipeline jobs.
WORKDIR /apollo
RUN sed "s/MACHINE_ARCH/x86_64/g" WORKSPACE.in > WORKSPACE && \
    bazel build -c opt \
        cyber:mainboard \
        cyber/py_wrapper/... \
        cyber/tools/cyber_recorder \
        modules/data/fuel/services:record_io_service \
        modules/data/tools/rosbag_to_record \
        modules/prediction:libprediction_component.so && \
    bash apollo.sh build_py
