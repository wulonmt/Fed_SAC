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

from functools import reduce
from flwr.common.logger import log
from logging import WARNING

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
        alpha_fed = False,
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
        self.alpha_fed = alpha_fed
        
    def alpha_aggregate(self, results: List[Tuple[NDArrays, int, float]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples, _ in results])
        ent_coefs_total = sum([coefs for _, _, coefs  in results])

        #Weighted by entropy coefficient value
        if self.alpha_fed:
            print("ent coefs: ", [coefs for _, _, coefs in results])
            weighted_weights = [
            [layer * coefs for layer in weights] for weights, _, coefs in results
            ]
                    # Compute average weights of each layer
            weights_prime: NDArrays = [
                reduce(np.add, layer_updates) / ent_coefs_total
                for layer_updates in zip(*weighted_weights)
            ]
            return weights_prime
            
        # Create a list of weights, each multiplied by the related number of examples
        else:
            weighted_weights = [
                [layer * num_examples for layer in weights] for weights, num_examples, _ in results
            ]
            # Compute average weights of each layer
            weights_prime: NDArrays = [
                reduce(np.add, layer_updates) / num_examples_total
                for layer_updates in zip(*weighted_weights)
            ]
            return weights_prime
        
    def aggregate_fit(self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics["ent_coef"])
            for _, fit_res in results
        ]
        
        parameters_aggregated = ndarrays_to_parameters(self.alpha_aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            
        #Saving model at the end
        if server_round == self.total_rounds:
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(parameters_aggregated)
            print(f"Saving round {server_round} aggregated_ndarrays...")
            if not os.path.isdir('result_model'):
                os.mkdir('result_model')
            np.savez(f"result_model/{self.ptime.get_time()}-weights_{self.model_path}.npz", *aggregated_ndarrays)

        return parameters_aggregated, metrics_aggregated
        
        
