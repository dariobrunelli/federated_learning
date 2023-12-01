import torch 
import flwr as fl

from typing import Dict
from flwr.common import Scalar
from collections import OrderedDict

from tools import train


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.trainloader = trainloader
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_parameters(self, parameters):
        """With the model paramters received from the server,
        overwrite the uninitialise model in this class with them."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        processed_dict = {}
        for idx, (k, v) in enumerate(self.model.state_dict().items()):
            if not "num_batches_tracked" in k:
                id = list(state_dict.keys()).index(f"{k}")
                processed_dict[f"{k}"] = state_dict[list(state_dict.keys())[id]]# now replace the parameters
        self.model.load_state_dict(processed_dict, strict=True)
        # self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract all model parameters and conver them to a list of
        NumPy arryas. The server doesn't work with PyTorch/TF/etc."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()] 

    def fit(self, parameters, config):
        """This method train the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # read from config
        lr, epochs = config["lr"], config["epochs"]

        # Define the optimizer
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        # do local training
        train(self.model, self.trainloader, optim, epochs, self.device, self.cfg)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return self.get_parameters({}), len(self.trainloader), {}

    # def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
    #     """Evaluate the model sent by the server on this client's
    #     local validation set. Then return performance metrics."""

    #     self.set_parameters(parameters)
    #     nme = test(self.model, self.valloader, self.cfg)      # FIXME: usiamo nme per loss e metrica in test
    #     # send statistics back to the server
    #     nmes = ["nme_eyes", "nme_mouth", "nme_nose", "nme_eyebrows", "nme_chin", "nme_dbox"]
    #     nme_dict = {nmes[i]: nme[i] for i in range(len(nmes))}
    #     print(nme_dict)
    #     return nme, len(self.valloader), {nme_dict}

def generate_client_fn(model, trainloaders, cfg):
    def client_fn(cid: str):
        """Returns a FlowerClient containing the cid-th data partition"""

        return FlowerClient(
            model=model, trainloader=trainloaders[int(cid)], cfg=cfg)

    return client_fn