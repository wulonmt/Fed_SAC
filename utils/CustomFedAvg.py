from flwr.server.strategy import FedAvg
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

class CustomFedAvg(FedAvg):
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
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
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
        self.ptime = Ptime()
        self.ptime.set_time_now()
        self.model_path = model_path
        self.total_rounds = total_rounds
        
    def aggregate_fit(self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        #Saving model at the end
        if server_round == self.total_rounds:
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(parameters_aggregated)
            print(f"Saving round {server_round} aggregated_ndarrays...")
            if not os.path.isdir('result_model'):
                os.mkdir('result_model')
            np.savez(f"result_model/{self.ptime.get_time()}-weights_{self.model_path}.npz", *aggregated_ndarrays)

        return parameters_aggregated, metrics_aggregated
        
        
