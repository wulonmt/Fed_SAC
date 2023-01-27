from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from stable_baselines3 import SAC
from airsim_client import AirsimClient
from flwr.common.typing import Parameters
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import GetParametersIns
from flwr.server.utils.tensorboard import tensorboard
from utils.CustomFedAdam import CustomFedAdam
from flwr.server.strategy import FedAvg, FedAdam


def main():
    init_model = AirsimClient()
    initial_parameters = init_model.get_parameters(config={})
    initial_parameters = ndarrays_to_parameters(initial_parameters)
                                        
    # Decorated strategy
    strategy = tensorboard("Server_log")(CustomFedAdam)(initial_parameters = initial_parameters,
                                                min_fit_clients=2,
                                                min_evaluate_clients=2,
                                                min_available_clients=2,)
    
    # Start Flower server
    fl.server.start_server(
        server_address="192.168.1.187:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
    
if __name__ == "__main__":
    main()
