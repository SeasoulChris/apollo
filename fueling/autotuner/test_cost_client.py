from fueling.autotuner.client.cost_computation_client import CostComputationClient

result = CostComputationClient.compute_mrac_cost(
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
)

print(f"Cost={result}")
