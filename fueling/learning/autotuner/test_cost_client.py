import time

from fueling.learning.autotuner.client.cost_computation_client import CostComputationClient
from fueling.learning.autotuner.proto.dynamic_model_info_pb2 import DynamicModel
from fueling.learning.autotuner.proto.git_info_pb2 import GitInfo

CHANNEL_URL = "localhost:50052"

# uncomment me if testing inside a cluster
# CHANNEL_URL="costservice:50052"

# uncomment me if testing for az-staging
# CHANNEL_URL="20.190.24.195:50052"

# uncomment me if testing for bce-staging
# CHANNEL_URL="180.76.157.176:50052"

# uncomment me if testing for bce-platform
# CHANNEL_URL="180.76.242.157:50052"

# uncomment me if testing for bce-debug
# CHANNEL_URL = "180.76.111.129:50052"

# settings
dynamic_model = DynamicModel.ECHO_LINCOLN
git_info = GitInfo(
    repo="https://github.com/ApolloAuto/apollo.git",
    commit_id="09ba18ada1aad062e0ffc3ea6f02de177895ed34"
)

# uncomment me for OWN_MODEL test
# dynamic_model = DynamicModel.OWN_MODEL
# git_info = GitInfo(
#    repo="https://github.com/vlin17/apollo.git",
#    commit_id="ff0756f99998c1ac373a8b4e6b8853fc9765ae4d"
# )

scenario_ids = [11014, 11015, 11016, 11017, 11018, 11019, 11020, 30019,
                30020, 30021, 30022, 30023, 30024, 30025]
configs = {  # map of config_id to path_to_config pairs
    "config_id1": {
        "/apollo/modules/control/conf/config_test1": "config1_pb2",
        "/apollo/modules/control/conf/config_test2": "config1_pb2",
    },
    "config_id2": {
        "/apollo/modules/control/conf/config_test1": "config1_pb2",
    },
}

# method 1:
"""
with CostComputationClient(CHANNEL_URL, commit_id, scenario_ids, dynamic_model) as client:
    iteration_id, score = client.compute_cost(configs)
    print(f"Received score {score} for {iteration_id}")
"""

# method 2:
client = CostComputationClient(CHANNEL_URL)
try:
    tic1 = time.perf_counter()

    client.initialize(git_info, scenario_ids, dynamic_model)
    print(f"Initialization time: {time.perf_counter() - tic1:0.4f} seconds")
    """
    client.set_token("tuner-8f6b804fc84f495280d27e6696330db2")
    """

    iteration = 3
    total_time = 0
    for i in range(iteration):
        tic = time.perf_counter()
        iteration_id, score = client.compute_cost(configs)
        compute_time = time.perf_counter() - tic
        print(f"Received score {score} for {iteration_id}")
        print(f"Compute time: {compute_time:0.4f} seconds")
        total_time += compute_time

    print(f"Average time: {total_time / iteration}")
except Exception as error:
    print(error)
finally:
    client.close()
    print(f"Done. Total time: {time.perf_counter() - tic1:0.4f} seconds")
