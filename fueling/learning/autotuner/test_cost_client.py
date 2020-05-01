import time

from fueling.learning.autotuner.client.cost_computation_client import CostComputationClient
from fueling.learning.autotuner.proto.dynamic_model_info_pb2 import DynamicModel

# uncomment me if testing inside a cluster
# CostComputationClient.set_channel("costservice:50052")

# uncomment me if testing for az-staging
# CostComputationClient.set_channel("40.77.110.196:50052")

# uncomment me if testing for bce-platform
# CostComputationClient.set_channel("180.76.242.157:50052")

# settings
commit_id = "c693dd9e2e7910b041416021fcdb648cc4d8934d"
scenario_ids = [11008, 11010]
dynamic_model = DynamicModel.ECHO_LINCOLN
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
with CostComputationClient(commit_id, scenario_ids, dynamic_model) as client:
    iteration_id, score = client.compute_cost(configs)
    print(f"Received score {score} for {iteration_id}")
"""

# method 2:
client = CostComputationClient()
try:
    tic1 = time.perf_counter()

    client.initialize(commit_id, scenario_ids, dynamic_model)
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
