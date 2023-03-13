from flwr.server.strategy import FedAdam, FedOpt, FedAvg
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.client_manager import ClientManager
import flwr as fl

import numpy as np
import os

from flwr.common import (
    FitRes,
    FitIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from utils.Ptime import Ptime

class CustomFedAdam(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-8,
        total_rounds: int,
        model_path:str = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.eta = eta
        self.eta_l = eta_l
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tau = tau
        self.initial = True
        self.m_t: Optional[NDArrays] = None
        self.v_t: Optional[NDArrays] = None
        self.total_rounds = total_rounds
        self.ptime = Ptime()
        self.ptime.set_time_now()
        self.model_path = model_path
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        #Aggregate fit results using weighted average.
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        
        fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)
        if self.initial:
            self.current_weights = fedavg_weights_aggregate
            self.initial = False
        
        # Adam
        delta_t: NDArrays = [
            x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
        ]

        # m_t
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            (np.multiply(self.beta_1, x) + (1 - self.beta_1) * y) / (1 - self.beta_1) #add bias modified
            for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            (self.beta_2 * x + (1 - self.beta_2) * np.multiply(y, y)) / (1 - self.beta_2) #add bias modified
            for x, y in zip(self.v_t, delta_t)
        ]

        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
        ]

        self.current_weights = new_weights
        
        #Saving model at the end
        if server_round == self.total_rounds:
            aggregated_ndarrays: List[np.ndarray] = self.current_weights
            print(f"Saving round {server_round} aggregated_ndarrays...")
            if not os.path.isdir('result_model'):
                os.mkdir('result_model')
            np.savez(f"result_model/{self.ptime.get_time()}-weights_{self.model_path}.npz", *aggregated_ndarrays)

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated
        
        #return fedavg_weights_aggregate, metrics_aggregated
        
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        config["learning_rate"] = self.eta_l
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
