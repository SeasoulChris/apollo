from fueling.learning.autotuner.client.cost_computation_client import CostComputationClient

# uncomment me if testing inside a cluster
# CostComputationClient.set_channel("costservice:50052")

# uncomment me if testing for az-staging
# CostComputationClient.set_channel("40.77.110.196:50052")

# uncomment me if testing for bce-platform
#CostComputationClient.set_channel("180.76.242.157:50052")

training_id, score = CostComputationClient.compute_mrac_cost(
    # commit id
    "c693dd9e2e7910b041416021fcdb648cc4d8934d",
    # map of config_id to path_to_config pairs
    {
        "config_id1": {
            "/apollo/modules/control/conf/config_test1": "config1_pb2",
            "/apollo/modules/control/conf/config_test2": "config1_pb2",
        },
        "config_id2": {
            "/apollo/modules/control/conf/config_test1": "config1_pb2",
        },
    },
    # scenario id list
    [11008, 11010]
)

print(f"Received score {score} for {training_id}")
