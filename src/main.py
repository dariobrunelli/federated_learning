import json
import argparse
import flwr as fl
import matplotlib.pyplot as plt

from utils.dataset import load_dataset
from clients import generate_client_fn
from model_hrnet import get_face_alignment_net
from utils.defaults import _C as config, update_config
from strategy import fit_config, weighted_average, get_evaluate_fn


NUM_CLIENTS = 50
NUM_ROUNDS = 10

# parse arguments and configurations
parser = argparse.ArgumentParser(description='Train Face Alignment')
parser.add_argument('--cfg', default="./experiments/toronto/face_alignment_toronto_hrnet_w18.yaml")
parser.add_argument('--num_rounds', default="None")
parser.add_argument('--num_clients', default="None")
args = parser.parse_args()
update_config(config, args)

num_rounds = args.num_rounds if args.num_rounds != "None" else config.NUM_ROUNDS
num_clients = args.num_clients if args.num_clients != "None" else config.NUM_CLIENTS
# num_clients = NUM_CLIENTS
# num_rounds = NUM_ROUNDS


# get dataset
train_loaders, val_loader = load_dataset(config, num_clients)

# get model
model = get_face_alignment_net(config)

# INSPECT ONLY get first partition
# train_partition = trainloaders[0].dataset
# partition_indices = train_partition.indices

# define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,                                   # Sample 10% of available clients for training
    fraction_evaluate=0.05,                             # Sample 5% of available clients for evaluation
    min_fit_clients=10,                                 # Never sample less than 10 clients for training
    min_evaluate_clients=5,                             # Never sample less than 5 clients for evaluation
    min_available_clients=int(num_clients * 0.75),      # Wait until at least 75% of the clients are available
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,   # aggregates federated metrics
    evaluate_fn= get_evaluate_fn(model, val_loader, config), # per Neuroface non c'è un test loader quindi facciamo train e valid
)

# define client function
client_fn_callback = generate_client_fn(model, train_loaders, config)


# With a dictionary, you tell Flower's VirtualClientEngine that each
# client needs exclusive access to these many resources in order to run
client_resources = {"num_cpus": 1, "num_gpus": 1.0}


history = fl.simulation.start_simulation(
    client_fn=client_fn_callback,                  # a callback to construct a client
    num_clients=num_clients,                       # total number of clients in the experiment
    config=fl.server.ServerConfig(num_rounds),     # let's run for 10 rounds
    strategy=strategy,                             # the strategy that will orchestrate the whole FL pipeline
    client_resources=client_resources)


print(f"{history.metrics_centralized = }")


with open('./trainings/history_metrics_centralized.json', 'w') as fp:
    json.dump(list(history.metrics_centralized.values()), fp)
with open('./trainings/history_losses_centralized.json', 'w') as fp:
    json.dump(list(history.losses_centralized), fp)

last_mses = []
for mse_list in history.metrics_centralized.values():
    last_mses.append(mse_list[-1][-1])

global_accuracy_centralised = sum(last_mses)/len(last_mses)
mets = history.metrics_centralized.keys()      # ["nme_eyes", "nme_mouth", ...]
vals = history.metrics_centralized.values()
for idx, met in enumerate(mets):
    round = [data[0] for data in history.metrics_centralized[f"{met}"]]
    nme = [100.0 * data[1] for data in history.metrics_centralized[f"{met}"]]
    plt.plot(round, nme)
    plt.grid()
    plt.ylabel("NME (%)")
    plt.xlabel("Round")
    plt.title(f"NEUROFACE - {met} - {num_clients} clients with {num_clients//num_rounds} clients per round")
    plt.savefig(f"."
                f"/trainings/fedlearn_neuroface{met}_{num_clients}cl_{num_rounds}rounds.png")


# model.double()
# # batch = train_loaders[0].dataset
# for data in train_loaders[0].dataset:
#     img = torch.from_numpy(data[0]).double()
#     target = data[1].double()
#     img = torch.unsqueeze(img, 0)
#     target = torch.unsqueeze(target, 0)
#     pred = model(img)
#     print(img.size(), pred.size(), target.size())
# # for k, v in model.state_dict().items():
# #     print(k, v)