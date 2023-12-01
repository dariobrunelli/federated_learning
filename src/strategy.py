import torch
from collections import OrderedDict
from typing import Tuple, Dict, List
from flwr.common import Metrics, Scalar

from tools import test


def get_evaluate_fn(model, testloader, config):
    """This is a function that returns a function. The returned
    function (i.e. `evaluate_fn`) will be executed by the strategy
    at the end of each round to evaluate the stat of the global
    model."""
    
    def evaluate_fn(server_round: int, parameters, _):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""
        
        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        torch.save(model.state_dict(), f"./trainings/aggregated_model_{server_round}round.pth")
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        processed_dict = {}
        for idx, (k, v) in enumerate(model.state_dict().items()):
            if not "num_batches_tracked" in k:
                id = list(state_dict.keys()).index(f"{k}")
                processed_dict[f"{k}"] = state_dict[list(state_dict.keys())[id]]

        model.load_state_dict(processed_dict, strict=False)
        # strict here is important since the heads layers are missing from the state,
        # we don't want this line to raise an error but load the present keys anyway.

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)  # send model to device
        # call test
        loss, mse_list = test(model, testloader, config)
        return loss, mse_list
    return evaluate_fn  


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # Number of local epochs done by clients
        "lr": 0.01,  # Learning rate to use by clients during fit()
    }
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
